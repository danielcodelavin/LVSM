import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils 
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer
from diffusers import DDPMScheduler
import math

import torch.nn.functional as F


class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()

        self._init_time_embedding()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        self._init_noise_predictor()

        self.init_noise_scheduler()

        # Initialize loss computer
        self.loss_computer = LossComputer(config)


    def _init_time_embedding(self):
        time_embed_dim = self.config.model.transformer.d * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(self.config.model.transformer.d, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.config.model.transformer.d),
        )
        self.time_embedding.apply(init_weights)

    def get_timestep_embedding(self, time_step):
        half_dim = self.config.model.transformer.d // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time_step.device) * -emb)
        emb = time_step[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.time_embed(emb)
    
    def _init_noise_predictor(self):
        self.noise_predictor = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(self.config.model.transformer.d, (self.config.model.target_pose_tokenizer.patch_size**2) * 3, bias=False,))
        self.noise_predictor.apply(init_weights)
        
    def _init_noise_scheduler(self):
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.diffusion.num_train_timesteps,
            beta_schedule=self.config.diffusion.beta_schedule,
            prediction_type="epsilon",  # predict noise
        )

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.n_layer)
        ]
        
        # Apply special initialization if configured
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()


    
    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(self.transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in self.transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens
            


    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
    
    
    def forward(self, data_batch, has_target_image=True):
        input, target = self.process_data(data_batch, has_target_image=has_target_image, 
                                        target_has_input=self.config.training.target_has_input, 
                                        compute_rays=True)
        
        # Process input images 
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)
        
        # Process target pose 
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)
        
        
        if has_target_image:
            # Tokenize the target images
            target_image_tokens = self.image_tokenizer(target.image * 2.0 - 1.0)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (b * v_target,), device=target_image_tokens.device
            )
            
            # Add noise to target image tokens according to the timesteps
            noise = torch.randn_like(target_image_tokens)
            noisy_target_tokens = self.noise_scheduler.add_noise(
                target_image_tokens, noise, timesteps
            )
            
            # Create timestep embeddings
            timestep_embeddings = self.get_timestep_embedding(timesteps)
            timestep_embeddings = timestep_embeddings.reshape(b * v_target, 1, d)
            timestep_embeddings = timestep_embeddings.expand(-1, n_patches, -1)
            
            # Add timestep embeddings to target pose tokens
            target_pose_tokens = target_pose_tokens + timestep_embeddings
        
        # Repeat input tokens and concatenate with target tokens 
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target, np=n_patches * v_input
        )
        
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        # Process through transformer 
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(concat_img_tokens, 
                                                gradient_checkpoint=True, 
                                                checkpoint_every=checkpoint_every)
        
        # Get only target token outputs
        _, target_tokens_output = transformer_output_tokens.split(
            [v_input * n_patches, n_patches], dim=1
        )
        
        # Predict noise instead of directly predicting the image
        predicted_noise = self.noise_predictor(target_tokens_output)
        
        # For training, calculate diffusion loss
        if has_target_image:
            loss_metrics = {
                "loss": F.mse_loss(predicted_noise, noise),
            }
        else:
            loss_metrics = None
        
        render = None
        
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=render,
            # Store these for use in other methods
            timesteps=timesteps if has_target_image else None,
            noise=noise if has_target_image else None,
            predicted_noise=predicted_noise if has_target_image else None,
        )
        
        return result


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """Render a video using diffusion sampling"""
        # Get input and target data
        if data_batch.input is None:
            input, target = self.process_data(
                data_batch, 
                has_target_image=False, 
                target_has_input=self.config.training.target_has_input, 
                compute_rays=True
            )
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target

        # Set up camera trajectories (using existing method from original implementation)
        rendering_ray_o, rendering_ray_d = self.setup_camera_trajectories(
            traj_type=traj_type, num_frames=num_frames, loop_video=loop_video, order_poses=order_poses
        )

        # Prepare input tokens (as in original implementation)
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()
        input_img_tokens = self.image_tokenizer(posed_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)

        # Compute rays for rendering (using existing method from original implementation)
        # For demonstration, we assume your original code computes refined rays for rendering
        rendering_ray_o, rendering_ray_d = self.compute_rendering_rays(rendering_ray_o, rendering_ray_d)

        # Get pose conditioning for target views (as in original implementation)
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )

        _, num_views, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)
        _, n_patches, d = target_pose_tokens.size()

        # Process views in chunks (using approach from original implementation)
        view_chunk_size = 4
        video_rendering_list = []

        # Loop through view chunks
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # Get current chunk of pose tokens
            start_idx = cur_chunk * n_patches
            end_idx = (cur_chunk + cur_view_chunk_size) * n_patches
            cur_target_pose_tokens = rearrange(
                target_pose_tokens[:, start_idx:end_idx, :], 
                "b (v_chunk p) d -> (b v_chunk) p d", 
                v_chunk=cur_view_chunk_size, p=n_patches
            )

            # Repeat input tokens for each target view
            repeated_input_img_tokens = repeat(
                input_img_tokens.detach(), "b np d -> (b chunk) np d", 
                chunk=cur_view_chunk_size, np=n_patches * v_input
            )

            # Initialize with random noise for diffusion
            sample = torch.randn(
                (bs * cur_view_chunk_size, n_patches, (self.config.model.target_pose_tokenizer.patch_size ** 2) * 3),
                device=input_img_tokens.device
            )

            # Diffusion sampling loop
            for t in self.noise_scheduler.timesteps:
                # Create timestep tensor
                timesteps = torch.full(
                    (bs * cur_view_chunk_size,), t, 
                    device=input_img_tokens.device, dtype=torch.long
                )

                # Get timestep embeddings
                timestep_embeddings = self.get_timestep_embedding(timesteps)
                timestep_embeddings = timestep_embeddings.reshape(bs * cur_view_chunk_size, 1, d)
                timestep_embeddings = timestep_embeddings.expand(-1, n_patches, -1)

                # Add timestep embeddings to pose tokens
                time_conditioned_pose_tokens = cur_target_pose_tokens + timestep_embeddings

                # Concatenate input tokens with pose tokens
                cur_concat_input_tokens = torch.cat(
                    (repeated_input_img_tokens, time_conditioned_pose_tokens), dim=1
                )
                cur_concat_input_tokens = self.transformer_input_layernorm(cur_concat_input_tokens)

                # Process through transformer (as in original implementation)
                transformer_output_tokens = self.pass_layers(
                    cur_concat_input_tokens, gradient_checkpoint=False
                )

                # Extract target tokens output
                _, target_tokens_output = transformer_output_tokens.split(
                    [v_input * n_patches, n_patches], dim=1
                )

                # Predict noise
                model_output = self.noise_predictor(target_tokens_output)

                # Perform denoising step using the DDPM scheduler
                sample = self.noise_scheduler.step(
                    model_output=model_output, 
                    timestep=t, 
                    sample=sample
                ).prev_sample

            # Decode the final denoised tokens to images
            decoded_images = torch.sigmoid((sample + 1.0) / 2.0)  # Rescale from [-1,1] to [0,1]

            # Reshape to proper image dimensions
            video_rendering = rearrange(
                decoded_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=target.image_h_w[0] // self.config.model.target_pose_tokenizer.patch_size, 
                w=target.image_h_w[1] // self.config.model.target_pose_tokenizer.patch_size, 
                p1=self.config.model.target_pose_tokenizer.patch_size, 
                p2=self.config.model.target_pose_tokenizer.patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)

        # Concatenate all chunks and return
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering

        return data_batch


    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0



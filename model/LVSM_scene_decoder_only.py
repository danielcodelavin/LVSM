import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils 
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer

# New imports for diffusion
from diffusers import DDPMScheduler


class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize loss computer for non-diffusion path
        self.loss_computer = LossComputer(config)


        if self.config.training.get("use_diffusion", False):
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.config.diffusion.num_train_timesteps,
                beta_start=self.config.diffusion.beta_start,
                beta_end=self.config.diffusion.beta_end,
                beta_schedule=self.config.diffusion.beta_schedule,)

            d_model = self.config.model.transformer.d

            self.time_proj = nn.Linear(1, d_model)
            self.time_proj.apply(lambda module: init_weights(module, 0.02))

            # This is the main embedding network, which now correctly receives
            # a d_model-dimensional input from the projection layer.
            self.time_embedding = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.time_embedding.apply(lambda module: init_weights(module, 0.02))
            print("INFO: Diffusion components initialized.")
        


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
        # Image tokenizer (expects 9 channels: 3 image + 6 pose)
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer (expects 6 channels: pose only)
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
            # sigmoid is now applied conditionally
            # in the forward_direct path, as noise prediction should be unbounded.
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(config.d, config.d_head, use_qk_norm=use_qk_norm) 
            for _ in range(config.n_layer)
        ]
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                weight_init_std = 0.02 / (2 * (idx + 1))**0.5 if config.depth_init else 0.02 / (2 * config.n_layer)**0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        if hasattr(self, 'loss_computer'):
            self.loss_computer.eval()


    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing."""
        num_layers = len(self.transformer_blocks)
        if not gradient_checkpoint:
            for layer in self.transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
        def _process_layer_group(tokens, start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_blocks[idx](tokens)
            return tokens
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group, input_tokens, start_idx, end_idx, use_reentrant=False
            )
        return input_tokens
            

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        """Creates posed input by concatenating image channels and pose channels."""
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        pose_cond = torch.cat([o_cross_d, ray_d], dim=2)
        if images is None:
            return pose_cond
        else:
            # When not using diffusion, input images are in [0, 1] and need to be scaled to [-1, 1].
            # When using diffusion, the input (noisy) images are already in the [-1, 1] range.
            if not self.config.training.get("use_diffusion", False):
                 images = images * 2.0 - 1.0
            return torch.cat([images, pose_cond], dim=2)
    
    
    def forward(self, data_batch, has_target_image=True):
        """Main forward pass that toggles between direct and diffusion paths."""
        use_diffusion = self.config.training.get("use_diffusion", False)
        if use_diffusion:
            return self.forward_diffusion(data_batch)
        else:
            return self.forward_direct(data_batch, has_target_image)

    def forward_direct(self, data_batch, has_target_image=True):
        """The original, direct prediction logic."""
        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)
        posed_input_images = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        b, v_input, _, _, _ = posed_input_images.size()
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)
        
        # FIX: Corrected typo from target.ray_oxq to target.ray_o
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, _, _, _ = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)
        
        repeated_input_img_tokens = repeat(input_img_tokens, 'b np d -> (b v_target) np d', v_target=v_target, np=n_patches * v_input)
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(concat_img_tokens, gradient_checkpoint=self.training, checkpoint_every=checkpoint_every)
        
        _, target_image_tokens = transformer_output_tokens.split([v_input * n_patches, n_patches], dim=1)
        
        # Apply sigmoid ONLY in the direct path to map outputs to [0,1]
        rendered_pixels = torch.sigmoid(self.image_token_decoder(target_image_tokens))
        
        height, width, patch_size = target.image_h_w[0], target.image_h_w[1], self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(rendered_pixels, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)", v=v_target, h=height//patch_size, w=width//patch_size, p1=patch_size, p2=patch_size, c=3)
        
        loss_metrics = self.loss_computer(rendered_images, target.image) if has_target_image else None
        
        return edict(input=input, target=target, loss_metrics=loss_metrics, render=rendered_images)

    def forward_diffusion(self, data_batch):
       
        input, target = self.process_data(data_batch, has_target_image=True, target_has_input=self.config.training.target_has_input, compute_rays=True)
        device, b, v_target = target.image.device, target.image.shape[0], target.image.shape[1]
        
        gt_images = rearrange(target.image * 2.0 - 1.0, "b v c h w -> (b v) c h w")
        noise = torch.randn_like(gt_images)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (gt_images.shape[0],), device=device).long()
        noisy_images = self.scheduler.add_noise(gt_images, noise, timesteps)
        
        posed_input_images = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        _, v_input, _, _, _ = posed_input_images.size()
        input_img_tokens = self.image_tokenizer(posed_input_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)
        
        noisy_images_reshaped = rearrange(noisy_images, "(b v) c h w -> b v c h w", b=b)
        posed_noisy_target = self.get_posed_input(images=noisy_images_reshaped, ray_o=target.ray_o, ray_d=target.ray_d)
        
        target_tokens = self.image_tokenizer(posed_noisy_target)
        
        alphas = self.scheduler.alphas_cumprod[timesteps].to(device).float().view(-1, 1)
        time_proj_emb = self.time_proj(alphas)
        time_emb = self.time_embedding(time_proj_emb).unsqueeze(1).expand(-1, n_patches, -1)
        target_tokens_with_time = target_tokens + time_emb

        repeated_input_img_tokens = repeat(input_img_tokens, 'b np d -> (b v_target) np d', v_target=v_target)
        transformer_input = torch.cat((repeated_input_img_tokens, target_tokens_with_time), dim=1)
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(concat_img_tokens, gradient_checkpoint=self.training, checkpoint_every=checkpoint_every)
        
        _, predicted_noise_tokens = transformer_output_tokens.split([v_input * n_patches, n_patches], dim=1)
        predicted_noise = self.image_token_decoder(predicted_noise_tokens)
        
        height, width, patch_size = target.image_h_w[0], target.image_h_w[1], self.config.model.target_pose_tokenizer.patch_size
        predicted_noise = rearrange(predicted_noise, "(b v) (h w) (p1 p2 c) -> (b v) c (h p1) (w p2)", b=b, v=v_target, h=height//patch_size, w=width//patch_size, p1=patch_size, p2=patch_size, c=3)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].to(device).float().view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (noisy_images - beta_prod_t.sqrt() * predicted_noise) / alpha_prod_t.sqrt()
        
        pred_x0_reshaped = rearrange(pred_x0, "(b v) c h w -> b v c h w", b=b)
        rendered_images = torch.clamp(pred_x0_reshaped, -1.0, 1.0) / 2.0 + 0.5

        if self.config.training.get("use_regular_mse_for_diffusion", True):
            loss = F.mse_loss(predicted_noise, noise)
            loss_metrics = edict(loss=loss, mse_loss=loss.detach())
        else:
            loss_metrics = self.loss_computer(rendered_images, target.image) if has_target_image else None
        
        return edict(input=input, target=target, loss_metrics=loss_metrics, render=rendered_images)


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """Dispatches to the correct video rendering method based on config."""
        if self.config.training.get("use_diffusion", False):
             return self.render_video_diffusion(data_batch, traj_type, num_frames, loop_video, order_poses)
        else:
             return self.render_video_direct(data_batch, traj_type, num_frames, loop_video, order_poses)

    @torch.no_grad()
    def render_video_direct(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        if not hasattr(data_batch, 'input') or data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        posed_images = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        bs, v_input, _, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)

        if traj_type == "interpolate":
            c2ws, fxfycxcy, device = input.c2w, input.fxfycxcy, input.c2w.device
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device)
            intrinsics[:, :, 0, 0], intrinsics[:, :, 1, 1], intrinsics[:, :, 0, 2], intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 0], fxfycxcy[:, :, 1], fxfycxcy[:, :, 2], fxfycxcy[:, :, 3]
            if loop_video:
                c2ws, intrinsics = torch.cat([c2ws, c2ws[:, [0], :]], dim=1), torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)
            all_c2ws, all_intrinsics = [], []
            for b_idx in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(c2ws[b_idx, :, :3, :4], intrinsics[b_idx], num_frames, order_poses=order_poses)
                all_c2ws.append(cur_c2ws.to(device)); all_intrinsics.append(cur_intrinsics.to(device))
            all_c2ws, all_intrinsics = torch.stack(all_c2ws, dim=0), torch.stack(all_intrinsics, dim=0)
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0], all_fxfycxcy[:, :, 1], all_fxfycxcy[:, :, 2], all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 0, 0], all_intrinsics[:, :, 1, 1], all_intrinsics[:, :, 0, 2], all_intrinsics[:, :, 1, 2]

        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device)
        target_pose_cond = self.get_posed_input(ray_o=rendering_ray_o.to(input.image.device), ray_d=rendering_ray_d.to(input.image.device))
                
        _, num_views, _, _, _ = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)
        _, n_patches, d = target_pose_tokens.size()
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)

        video_rendering_list, view_chunk_size = [], 4
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)
            repeated_input_img_tokens = repeat(input_img_tokens.detach(), 'b np d -> (b chunk) np d', chunk=cur_view_chunk_size, np=v_input * n_patches)
            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches            
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], "b (v_chunk p) d -> (b v_chunk) p d", v_chunk=cur_view_chunk_size, p=n_patches)
            cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1)
            cur_concat_input_tokens = self.transformer_input_layernorm(cur_concat_input_tokens)
            transformer_output_tokens = self.pass_layers(cur_concat_input_tokens, gradient_checkpoint=False)
            _, pred_target_image_tokens = transformer_output_tokens.split([v_input * n_patches, n_patches], dim=1)
            
            video_rendering = torch.sigmoid(self.image_token_decoder(pred_target_image_tokens))
            
            height, width, patch_size = target.image_h_w[0], target.image_h_w[1], self.config.model.target_pose_tokenizer.patch_size
            video_rendering = rearrange(video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)", v=cur_view_chunk_size, h=height//patch_size, w=width//patch_size, p1=patch_size, p2=patch_size, c=3).cpu()
            video_rendering_list.append(video_rendering)
        
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering
        return data_batch

    @torch.no_grad()
    def render_video_diffusion(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """Renders a video using the DDPM sampling loop."""
        if not hasattr(data_batch, 'input') or data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
        else:
            input, target = data_batch.input, data_batch.target

        posed_images = self.get_posed_input(images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        bs, v_input, _, h, w = posed_images.size()
        input_img_tokens = self.image_tokenizer(posed_images)
        _, n_patches, d = input_img_tokens.size()
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)

        if traj_type == "interpolate":
            c2ws, fxfycxcy, device = input.c2w, input.fxfycxcy, input.c2w.device
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device)
            intrinsics[:, :, 0, 0], intrinsics[:, :, 1, 1], intrinsics[:, :, 0, 2], intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 0], fxfycxcy[:, :, 1], fxfycxcy[:, :, 2], fxfycxcy[:, :, 3]
            if loop_video:
                c2ws, intrinsics = torch.cat([c2ws, c2ws[:, [0], :]], dim=1), torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)
            all_c2ws, all_intrinsics = [], []
            for b_idx in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(c2ws[b_idx, :, :3, :4], intrinsics[b_idx], num_frames, order_poses=order_poses)
                all_c2ws.append(cur_c2ws.to(device)); all_intrinsics.append(cur_intrinsics.to(device))
            all_c2ws, all_intrinsics = torch.stack(all_c2ws, dim=0), torch.stack(all_intrinsics, dim=0)
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0], all_fxfycxcy[:, :, 1], all_fxfycxcy[:, :, 2], all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 0, 0], all_intrinsics[:, :, 1, 1], all_intrinsics[:, :, 0, 2], all_intrinsics[:, :, 1, 2]

        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device)
        
        noisy_frames = torch.randn((bs, num_frames, 3, h, w), device=device)
        self.scheduler.set_timesteps(self.config.diffusion.num_train_timesteps)

        for t in self.scheduler.timesteps:
            print(f"Sampling timestep {t}/{self.scheduler.config.num_train_timesteps}", end='\r')
            posed_noisy_frames = self.get_posed_input(images=noisy_frames, ray_o=rendering_ray_o, ray_d=rendering_ray_d)
            
            target_tokens = self.image_tokenizer(posed_noisy_frames)
            
            timesteps_tensor = torch.tensor([t] * target_tokens.shape[0], device=device)
            alphas = self.scheduler.alphas_cumprod[timesteps_tensor].to(device).float().view(-1, 1)
            time_proj_emb = self.time_proj(alphas)
            time_emb = self.time_embedding(time_proj_emb).unsqueeze(1).expand(-1, n_patches, -1)
            target_tokens_with_time = target_tokens + time_emb

            repeated_input_img_tokens = repeat(input_img_tokens, 'b np d -> (b v_target) np d', v_target=num_frames)
            transformer_input = torch.cat((repeated_input_img_tokens, target_tokens_with_time), dim=1)
            concat_img_tokens = self.transformer_input_layernorm(transformer_input)

            transformer_output_tokens = self.pass_layers(concat_img_tokens, gradient_checkpoint=False)
            _, predicted_noise_tokens = transformer_output_tokens.split([v_input * n_patches, n_patches], dim=1)

            predicted_noise = self.image_token_decoder(predicted_noise_tokens)
            
            # --- FIX: Replaced hardcoded patch size with value from config ---
            patch_size = self.config.model.target_pose_tokenizer.patch_size
            predicted_noise = rearrange(predicted_noise, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)", b=bs, v=num_frames, h=h//patch_size, w=w//patch_size, p1=patch_size, p2=patch_size, c=3)
            
            # Here, t is a scalar, so scheduler.step works correctly.
            noisy_frames = self.scheduler.step(predicted_noise.squeeze(0), t, noisy_frames.squeeze(0)).prev_sample.unsqueeze(0)
            
        video_rendering = torch.clamp(noisy_frames, -1.0, 1.0) / 2.0 + 0.5
        data_batch.video_rendering = video_rendering.cpu()
        print("\nVideo rendering complete.")
        return data_batch


    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            if not ckpt_names:
                print(f"No .pt files found in directory {load_path}")
                return None
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
            
            # --- FIX: Intelligently find the model state dictionary ---
            # Check for common key names for the model's state dictionary
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # If no known key is found, assume the checkpoint itself is the state_dict
                state_dict = checkpoint

            # Load the found state dictionary
            status = self.load_state_dict(state_dict, strict=False)
            print("Model load status:", status)
            # --- END OF FIX ---
            
            print(f"Successfully loaded checkpoint from {ckpt_paths[-1]}")
            return 0
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to load checkpoint from {load_path}: {e}")
            return None
"""
Inference script with multi-step diffusion recycling
Supports two schedules via command line:
  --schedule fast    : 10 steps (10_steps_mixed baseline) - FAST
  --schedule best    : 60 steps (60_ultra_high) - BEST QUALITY
"""

import importlib
import os
import time
import torch
import argparse
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
import litdata as ld
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
from einops import rearrange, repeat
from easydict import EasyDict as edict


# Define the two recycling schedules
SCHEDULES = {
    'fast': {
        'name': '10_steps_mixed',
        'steps': [999, 990, 975, 950, 920, 880, 820, 730, 600, 400],
        'description': '10 steps',
    },
    'best': {
        'name': '60_ultra_high',
        'steps': [999] * 8 + [995] * 8 + [990] * 8 + [985] * 6 + list(range(980, -1, -28)),
        'description': '60 steps',
    },
    'superlong_linear': {
        'name': 'superlong_linear',
       'steps': [999] * 8 + [995] * 8 + [990] * 8 + [985] * 8 + [970]*10 +list(range(950, -1, -14)),
        'description': '60 steps',
    },
    'superfast':{
        'name': 'superfast',
       'steps': [999] * 16 + [995] * 16 + [990] * 16 + [985] * 16 + [970]*10 +list(range(950, -1, -14)),
        'description': 'not superfast',
    },
    'linear_best': {
        'name': 'linear_best',
        'steps': list(range(999, 939, -1)) + list(range(935, -1, -15)),  # 60 unique decreasing values
        'description': '60 steps - always decreasing',
    },
    'repeat_complete':{
        'name': 'repeat_complete',
        'steps': [999] + [990] +list(range(980, -1, -14)),
        'description': '60 steps - repeating complete schedule',


    }
}



class LitDataCollateInference:
    """Collate function for inference."""
    def __init__(self, config):
        self.config = config
        self.target_size = self.config.model.image_tokenizer.image_size

    def __call__(self, batch):
        min_views_required = self.config.training.num_views
        
        valid_samples = [s for s in batch if s and 'image' in s and s['image'].shape[0] >= min_views_required]

        if not valid_samples:
            return None

        processed_batch = {k: [] for k in valid_samples[0].keys()}
        num_views_to_sample = self.config.training.num_views
        random_behavior = self.config.training.get("random_sample_views", False)

        for sample in valid_samples:
            num_available_views = sample['image'].shape[0]
            
            if not random_behavior:
                view_selector_config = self.config.training.view_selector
                min_frame_dist = view_selector_config.get("min_frame_dist", 25)
                max_frame_dist = min(num_available_views - 1, view_selector_config.get("max_frame_dist", 100))
                
                if max_frame_dist <= min_frame_dist:
                    image_indices = random.sample(range(num_available_views), num_views_to_sample)
                else:
                    frame_dist = random.randint(min_frame_dist, max_frame_dist)
                    start_frame = random.randint(0, num_available_views - frame_dist - 1)
                    end_frame = start_frame + frame_dist
                    num_intermediate_views = num_views_to_sample - 2
                    
                    if end_frame > start_frame + 1 and (end_frame - start_frame - 1) >= num_intermediate_views:
                        sampled_frames = random.sample(range(start_frame + 1, end_frame), num_intermediate_views)
                        image_indices = [start_frame, end_frame] + sampled_frames
                    else:
                        image_indices = random.sample(range(num_available_views), num_views_to_sample)
            else:
                image_indices = random.sample(range(num_available_views), num_views_to_sample)
            
            for key, value in sample.items():
                if key == 'image':
                    original_images = value[image_indices]
                    resized_images = torch.stack(
                        [TF.resize(img, [self.target_size, self.target_size], antialias=True) for img in original_images]
                    )
                    processed_batch[key].append(resized_images)
                elif isinstance(value, torch.Tensor):
                    processed_batch[key].append(value[image_indices])
                elif key == "scene_name":
                    processed_batch[key].append(value)
        
        if not processed_batch.get("image"):
            return None

        final_batch = {}
        for key, value_list in processed_batch.items():
            if value_list and isinstance(value_list[0], torch.Tensor):
                final_batch[key] = torch.stack(value_list)
            else:
                final_batch[key] = value_list
        return final_batch


def print_rank0(message):
    """Print only from rank 0 process."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message)
    else:
        print(message)


def denoise_single_step(model_module, input_img_tokens, target_data, timestep, config, device, n_patches, v_input, v_target, current_images):
    """Perform a single denoising step at the given timestep."""
    b = current_images.shape[0]
    h, w = target_data.image_h_w[0], target_data.image_h_w[1]
    
    # Encode current images with pose
    posed_noisy_target = model_module.get_posed_input(
        images=current_images, 
        ray_o=target_data.ray_o, 
        ray_d=target_data.ray_d
    )
    target_tokens = model_module.image_tokenizer(posed_noisy_target)
    
    # Add time embedding
    timesteps_tensor = torch.full((target_tokens.shape[0],), timestep, device='cpu', dtype=torch.long)
    alphas = model_module.scheduler.alphas_cumprod[timesteps_tensor].to(device).float().view(-1, 1)
    time_proj_emb = model_module.time_proj(alphas)
    time_emb = model_module.time_embedding(time_proj_emb).unsqueeze(1).expand(-1, n_patches, -1)
    target_tokens_with_time = target_tokens + time_emb
    
    # Prepare transformer input
    if config.training.get("true_cross_attention", False):
        target_tokens_with_time_flat = rearrange(target_tokens_with_time, '(b v) p d -> b (v p) d', b=b)
        transformer_input = torch.cat((input_img_tokens, target_tokens_with_time_flat), dim=1)
    else:
        repeated_input_img_tokens = repeat(input_img_tokens, 'b np d -> (b v_target) np d', v_target=v_target)
        transformer_input = torch.cat((repeated_input_img_tokens, target_tokens_with_time), dim=1)
    
    concat_img_tokens = model_module.transformer_input_layernorm(transformer_input)
    
    # Prepare kwargs
    block_type = type(model_module.transformer_blocks[0])
    kwargs = {}
    if block_type.__name__ == 'AlternatingAttentionBlock':
        kwargs = {'num_frames': v_input + v_target}
    elif block_type.__name__ == 'SourceTargetAttentionBlock':
        kwargs = {'source_token_len': v_input * n_patches}
    
    # Forward through transformer
    transformer_output_tokens = model_module.pass_layers(
        concat_img_tokens, 
        gradient_checkpoint=False, 
        **kwargs
    )
    
    # Extract target tokens
    if config.training.get("true_cross_attention", False):
        _, predicted_noise_tokens = transformer_output_tokens.split([v_input * n_patches, v_target * n_patches], dim=1)
        predicted_noise_tokens = rearrange(predicted_noise_tokens, 'b (v p) d -> (b v) p d', v=v_target)
    else:
        _, predicted_noise_tokens = transformer_output_tokens.split([v_input * n_patches, n_patches], dim=1)
    
    # Decode noise prediction
    predicted_noise = model_module.image_token_decoder(predicted_noise_tokens)
    
    patch_size = config.model.target_pose_tokenizer.patch_size
    predicted_noise = rearrange(
        predicted_noise, 
        "(b v) (h w) (p1 p2 c) -> (b v) c (h p1) (w p2)", 
        b=b, v=v_target, h=h//patch_size, w=w//patch_size, p1=patch_size, p2=patch_size, c=3
    )
    
    # Reconstruct x0
    current_images_flat = rearrange(current_images, "b v c h w -> (b v) c h w")
    timesteps = torch.full((current_images_flat.shape[0],), timestep, device=device, dtype=torch.long)
    alpha_prod_t = model_module.scheduler.alphas_cumprod[timesteps].to(device).float().view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (current_images_flat - beta_prod_t.sqrt() * predicted_noise) / alpha_prod_t.sqrt()
    
    pred_x0_reshaped = rearrange(pred_x0, "(b v) c h w -> b v c h w", b=b)
    
    return pred_x0_reshaped


def run_inference_diffusion_multistep(model, batch, config, device, amp_dtype, timestep_schedule):
    """
    Multi-step diffusion inference with recycling.
    Uses the validated schedule from experiments.
    THIS IS COPIED EXACTLY FROM THE WORKING EXPERIMENT FILE.
    """
    model_module = model.module if isinstance(model, DDP) else model
    
    # Process data
    input_data, target_data = model_module.process_data(
        batch, 
        has_target_image=True, 
        target_has_input=config.training.target_has_input, 
        compute_rays=True
    )
    
    b, v_target = target_data.image.shape[0], target_data.image.shape[1]
    h, w = target_data.image_h_w[0], target_data.image_h_w[1]
    
    # Encode input images
    posed_input_images = model_module.get_posed_input(
        images=input_data.image, 
        ray_o=input_data.ray_o, 
        ray_d=input_data.ray_d
    )
    _, v_input, _, _, _ = posed_input_images.size()
    input_img_tokens = model_module.image_tokenizer(posed_input_images)
    _, n_patches, d = input_img_tokens.size()
    input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)
    
    # Start with pure noise
    pure_noise = torch.randn((b, v_target, 3, h, w), device=device)
    
    # Initialize at highest timestep
    first_timestep = timestep_schedule[0]
    timesteps = torch.full((b * v_target,), first_timestep, device=device, dtype=torch.long)
    zero_images = torch.zeros((b * v_target, 3, h, w), device=device)
    pure_noise_flat = rearrange(pure_noise, "b v c h w -> (b v) c h w")
    current_images_flat = model_module.scheduler.add_noise(zero_images, pure_noise_flat, timesteps)
    current_images = rearrange(current_images_flat, "(b v) c h w -> b v c h w", b=b, v=v_target)
    
    # Recycling loop - EXACTLY AS IN THE WORKING EXPERIMENT FILE
    num_recycles = len(timestep_schedule)
    for recycle_idx, timestep in enumerate(timestep_schedule):
        denoised = denoise_single_step(
            model_module, input_img_tokens, target_data, timestep, 
            config, device, n_patches, v_input, v_target, current_images
        )
        
        # If not last step, add noise for next timestep
        if recycle_idx < num_recycles - 1:
            next_timestep = timestep_schedule[recycle_idx + 1]
            # Re-noise to next timestep
            denoised_flat = rearrange(denoised, "b v c h w -> (b v) c h w")
            next_timesteps = torch.full((denoised_flat.shape[0],), next_timestep, device=device, dtype=torch.long)
            noise_to_add = torch.randn_like(denoised_flat)
            current_images_flat = model_module.scheduler.add_noise(denoised_flat, noise_to_add, next_timesteps)
            current_images = rearrange(current_images_flat, "(b v) c h w -> b v c h w", b=b, v=v_target)
        else:
            current_images = denoised
    
   
    rendered_images = torch.clamp(current_images, -1.0, 1.0) / 2.0 + 0.5
    
    # Compute metrics
    loss_metrics = model_module.loss_computer(rendered_images, target_data.image)
    
    return edict(input=input_data, target=target_data, loss_metrics=loss_metrics, render=rendered_images)

def run_inference(config, model, dataloader, device, output_dir, schedule_name='fast'):
    """
    Run inference on entire dataset and compute metrics.
    
    Args:
        config: Configuration object
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        output_dir: Directory to save results
        schedule_name: 'fast' or 'best' schedule
    """
    model.eval()
    
    # Get the selected schedule
    schedule_config = SCHEDULES[schedule_name]
    timestep_schedule = schedule_config['steps']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print_rank0(f"Output directory: {os.path.abspath(output_dir)}")
    
    compute_metrics = config.inference.get("compute_metrics", True)
    render_video = config.inference.get("render_video", False)
    use_diffusion = config.training.get("use_diffusion", False)
    
    # Get AMP settings
    amp_dtype_mapping = {"f16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, 'tf32': torch.float32}
    use_amp = config.training.get("use_amp", False)
    amp_dtype = amp_dtype_mapping[config.training.get("amp_dtype", "f16")]
    
    total_samples = 0
    dataloader_iter = iter(dataloader)
    
    
    with torch.no_grad():
       
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(total=len(dataloader.dataset), disable=not is_main, desc=f"Inference ({schedule_name})")
        
        while True:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            
            if batch is None:
                continue
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with AMP
            with torch.autocast(enabled=use_amp, device_type="cuda", dtype=amp_dtype):
                if use_diffusion:
                    # Use multi-step recycling
                    result = run_inference_diffusion_multistep(
                        model, batch, config, device, amp_dtype, timestep_schedule
                    )
                else:
                   
                    result = model(batch, has_target_image=True)
            
            # Optionally render video
            if render_video:
                render_config = config.inference.get("render_video_config", {})
                result = model.module.render_video(
                    result,
                    traj_type=render_config.get("traj_type", "interpolate"),
                    num_frames=render_config.get("num_frames", 60),
                    loop_video=render_config.get("loop_video", False),
                    order_poses=render_config.get("order_poses", False)
                ) if isinstance(model, DDP) else model.render_video(
                    result,
                    traj_type=render_config.get("traj_type", "interpolate"),
                    num_frames=render_config.get("num_frames", 60),
                    loop_video=render_config.get("loop_video", False),
                    order_poses=render_config.get("order_poses", False)
                )
            
            # Export results
            export_results(
                result,
                output_dir,
                compute_metrics=compute_metrics
            )
            
            batch_size = batch['image'].shape[0] if isinstance(batch['image'], torch.Tensor) else len(batch['image'])
            total_samples += batch_size
            pbar.update(batch_size)
            
            # Clear cache periodically
            if total_samples % 100 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
    
    print_rank0(f"\nInference complete! Processed {total_samples} samples.")
    
    # Summarize metrics if computed
    if compute_metrics and (not dist.is_initialized() or dist.get_rank() == 0):
        print_rank0("\nComputing summary statistics...")
        summarize_evaluation(output_dir)
    
    return total_samples


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference with diffusion recycling')
    parser.add_argument('--schedule', type=str, default='best', choices=['fast', 'superlong_linear', 'superlong_linear', 'superfast', 'linear_best', 'repeat_complete'],
                       help='Recycling schedule: fast (10 steps) or best (60 steps)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional, uses default if not specified)')
    args, unknown = parser.parse_known_args()  # Allow other args to pass through to config
    
  
    config = init_config()
    
    
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    
   
    ddp_info = init_distributed(seed=777)
    
    if dist.is_initialized():
        dist.barrier()
    
    
    torch.backends.cuda.matmul.allow_tf32 = config.training.get("use_tf32", True)
    torch.backends.cudnn.allow_tf32 = config.training.get("use_tf32", True)
    
    # Get output directory from config
    output_dir = config.inference.get("inference_out_dir", config.get("inference_out_dir", "./inference_output"))
    
    
    
   
    print_rank0("Initializing LitData StreamingDataset...")
    job_id = os.environ.get("SLURM_JOB_ID", f"inference_{int(time.time())}")
    cache_dir = f"/dev/shm/litdata_cache_inference_{job_id}"
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(cache_dir, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    print_rank0(f"Using RAM Disk cache directory: {cache_dir}")
    
    dataset = ld.StreamingDataset(
        input_dir=config.training.dataset_path,
        cache_dir=cache_dir,
        max_cache_size="50GB",
        shuffle=False,
    )
    
    if len(dataset) == 0:
        print_rank0(f"\nFATAL ERROR: LitData dataset is empty. Check path: {os.path.abspath(config.training.dataset_path)}")
        if dist.is_initialized():
            dist.barrier()
        exit(1)
    
    collate_fn = LitDataCollateInference(config)
    batch_size_per_gpu = config.training.get("batch_size_per_gpu", 1)
    
    dataloader = ld.StreamingDataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        num_workers=config.training.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    print_rank0(f"Dataloader ready. Dataset size: {len(dataset)}")
    
    # Initialize model
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)
    
   
    checkpoint_path = config.training.get("checkpoint_dir", "./checkpoints")
    print_rank0(f"Loading checkpoint from: {checkpoint_path}")
    
    load_result = model.load_ckpt(checkpoint_path)
    if load_result is None:
        print_rank0("ERROR: Failed to load checkpoint!")
        if dist.is_initialized():
            dist.barrier()
        exit(1)
    
   
    if dist.is_initialized() and ddp_info.world_size > 1:
        model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)
        print_rank0(f"Using DDP with world size: {ddp_info.world_size}")
    
    if dist.is_initialized():
        dist.barrier()
 
    start_time = time.time()
    total_samples = run_inference(
        config=config,
        model=model,
        dataloader=dataloader,
        device=ddp_info.device,
        output_dir=output_dir,
        schedule_name=args.schedule
    )
    elapsed_time = time.time() - start_time
    
    print_rank0(f"\n{'='*70}")
    print_rank0(f"INFERENCE COMPLETED SUCCESSFULLY")
    print_rank0(f"{'='*70}")
    print_rank0(f"Schedule used: {SCHEDULES[args.schedule]['name']}")
    print_rank0(f"Total samples processed: {total_samples}")
    print_rank0(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
    print_rank0(f"Average time per sample: {elapsed_time/total_samples:.2f}s")
    print_rank0(f"Results saved to: {os.path.abspath(output_dir)}")
    print_rank0(f"{'='*70}\n")
    
    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
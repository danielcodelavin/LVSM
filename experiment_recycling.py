"""
Comprehensive Diffusion Schedule Testing Script
Tests multiple schedule variations to find optimal denoising strategy
Based on experimental results: superfast (PSNR 17) > best (PSNR 15)
"""

import importlib
import os
import time
import torch
import argparse
from rich import print
from rich.table import Table
from rich.console import Console
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
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ============================================================================
# SCHEDULE DEFINITIONS - Strategic Variations
# ============================================================================

def generate_test_schedules():
    """
    Generate strategic schedule variations based on experimental findings.
    
    Key insights from your results:
    - superfast (fewer steps) outperforms best (more steps)
    - Pattern: [999, 990] + linear descent works well
    
    Strategy: Explore variations in:
    1. Number of initial high-timestep repeats
    2. Linear descent step size
    3. Starting point of linear descent
    4. Total number of steps
    """
    
    schedules = {}
    
    # BASELINE - Your current best performers
    schedules['superfast_baseline'] = {
        'name': 'superfast_baseline',
        'steps': [999] + [990] + list(range(980, -1, -28)),
        'description': 'Original superfast (PSNR 17)',
    }
    
    schedules['best_baseline'] = {
        'name': 'best_baseline',
        'steps': [999] * 8 + [995] * 8 + [990] * 8 + [985] * 6 + list(range(980, -1, -28)),
        'description': 'Original best (PSNR 15)',
    }
    
    # VARIATION 1: Even fewer initial steps, faster descent
    schedules['ultrafast_steep'] = {
        'name': 'ultrafast_steep',
        'steps': [999, 990] + list(range(975, -1, -35)),
        'description': '2 high + steep descent (-35)',
    }
    
    # VARIATION 2: Single high step, aggressive descent
    schedules['minimal_aggressive'] = {
        'name': 'minimal_aggressive',
        'steps': [999] + list(range(985, -1, -30)),
        'description': '1 high + aggressive (-30)',
    }
    
    # VARIATION 3: More gradual near top, then steep
    schedules['gradual_top_steep'] = {
        'name': 'gradual_top_steep',
        'steps': [999, 995, 990, 985] + list(range(975, -1, -35)),
        'description': '4 gradual top + steep descent',
    }
    
    # VARIATION 4: Logarithmic-like spacing (denser at high timesteps)
    schedules['log_spaced'] = {
        'name': 'log_spaced',
        'steps': [999, 995, 990, 985, 980, 970, 955, 935] + list(range(900, -1, -50)),
        'description': 'Dense high, sparse low',
    }
    
    # VARIATION 5: Two-phase: slow then fast
    schedules['two_phase_slow_fast'] = {
        'name': 'two_phase_slow_fast',
        'steps': [999, 990] + list(range(985, 800, -15)) + list(range(780, -1, -40)),
        'description': 'Slow descent then fast',
    }
    
    # VARIATION 6: Three initial, medium descent
    schedules['triple_medium'] = {
        'name': 'triple_medium',
        'steps': [999, 995, 990] + list(range(980, -1, -25)),
        'description': '3 high + medium descent (-25)',
    }
    
    # VARIATION 7: Extended superfast with finer steps
    schedules['superfast_fine'] = {
        'name': 'superfast_fine',
        'steps': [999, 990] + list(range(980, -1, -22)),
        'description': 'Superfast pattern + finer (-22)',
    }
    
    # VARIATION 8: Minimal with very steep
    schedules['minimal_verysteep'] = {
        'name': 'minimal_verysteep',
        'steps': [999] + list(range(980, -1, -40)),
        'description': '1 high + very steep (-40)',
    }
    
    # VARIATION 9: Five high steps, aggressive after
    schedules['five_high_aggressive'] = {
        'name': 'five_high_aggressive',
        'steps': [999, 997, 995, 992, 990] + list(range(980, -1, -32)),
        'description': '5 high gradual + aggressive',
    }
    
    # VARIATION 10: Exponential-like decay
    timesteps_exp = [999]
    current = 999
    decay_rate = 0.92
    while current > 10:
        current = int(current * decay_rate)
        if current not in timesteps_exp:
            timesteps_exp.append(current)
    timesteps_exp.append(0)
    schedules['exponential_decay'] = {
        'name': 'exponential_decay',
        'steps': timesteps_exp,
        'description': 'Exponential decay (0.92)',
    }
    
    # VARIATION 11: Hybrid: few high + medium steps + aggressive
    schedules['hybrid_balanced'] = {
        'name': 'hybrid_balanced',
        'steps': [999, 990] + list(range(980, 600, -20)) + list(range(590, -1, -40)),
        'description': 'Hybrid: balanced then aggressive',
    }
    
    # VARIATION 12: Very minimal (testing lower bound)
    schedules['ultra_minimal'] = {
        'name': 'ultra_minimal',
        'steps': [999] + list(range(970, -1, -50)),
        'description': 'Ultra minimal (1+sparse)',
    }
    
    # VARIATION 13: Denser around middle timesteps
    schedules['middle_dense'] = {
        'name': 'middle_dense',
        'steps': [999, 990] + list(range(980, 700, -20)) + list(range(690, 400, -10)) + list(range(390, -1, -30)),
        'description': 'Dense in middle range',
    }
    
    # VARIATION 14: Four high + optimized descent (between superfast and best)
    schedules['four_high_optimized'] = {
        'name': 'four_high_optimized',
        'steps': [999, 995, 990, 985] + list(range(980, -1, -28)),
        'description': '4 high + superfast descent',
    }
    
    # VARIATION 15: Reverse engineering - trying to beat superfast
    schedules['beat_superfast'] = {
        'name': 'beat_superfast',
        'steps': [999, 992] + list(range(985, -1, -26)),
        'description': 'Optimized to beat superfast',
    }
    
    return schedules


# ============================================================================
# INFERENCE CODE (Same as original)
# ============================================================================

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
    
    posed_noisy_target = model_module.get_posed_input(
        images=current_images, 
        ray_o=target_data.ray_o, 
        ray_d=target_data.ray_d
    )
    target_tokens = model_module.image_tokenizer(posed_noisy_target)
    
    timesteps_tensor = torch.full((target_tokens.shape[0],), timestep, device='cpu', dtype=torch.long)
    alphas = model_module.scheduler.alphas_cumprod[timesteps_tensor].to(device).float().view(-1, 1)
    time_proj_emb = model_module.time_proj(alphas)
    time_emb = model_module.time_embedding(time_proj_emb).unsqueeze(1).expand(-1, n_patches, -1)
    target_tokens_with_time = target_tokens + time_emb
    
    if config.training.get("true_cross_attention", False):
        target_tokens_with_time_flat = rearrange(target_tokens_with_time, '(b v) p d -> b (v p) d', b=b)
        transformer_input = torch.cat((input_img_tokens, target_tokens_with_time_flat), dim=1)
    else:
        repeated_input_img_tokens = repeat(input_img_tokens, 'b np d -> (b v_target) np d', v_target=v_target)
        transformer_input = torch.cat((repeated_input_img_tokens, target_tokens_with_time), dim=1)
    
    concat_img_tokens = model_module.transformer_input_layernorm(transformer_input)
    
    block_type = type(model_module.transformer_blocks[0])
    kwargs = {}
    if block_type.__name__ == 'AlternatingAttentionBlock':
        kwargs = {'num_frames': v_input + v_target}
    elif block_type.__name__ == 'SourceTargetAttentionBlock':
        kwargs = {'source_token_len': v_input * n_patches}
    
    transformer_output_tokens = model_module.pass_layers(
        concat_img_tokens, 
        gradient_checkpoint=False, 
        **kwargs
    )
    
    if config.training.get("true_cross_attention", False):
        _, predicted_noise_tokens = transformer_output_tokens.split([v_input * n_patches, v_target * n_patches], dim=1)
        predicted_noise_tokens = rearrange(predicted_noise_tokens, 'b (v p) d -> (b v) p d', v=v_target)
    else:
        _, predicted_noise_tokens = transformer_output_tokens.split([v_input * n_patches, n_patches], dim=1)
    
    predicted_noise = model_module.image_token_decoder(predicted_noise_tokens)
    
    patch_size = config.model.target_pose_tokenizer.patch_size
    predicted_noise = rearrange(
        predicted_noise, 
        "(b v) (h w) (p1 p2 c) -> (b v) c (h p1) (w p2)", 
        b=b, v=v_target, h=h//patch_size, w=w//patch_size, p1=patch_size, p2=patch_size, c=3
    )
    
    current_images_flat = rearrange(current_images, "b v c h w -> (b v) c h w")
    timesteps = torch.full((current_images_flat.shape[0],), timestep, device=device, dtype=torch.long)
    alpha_prod_t = model_module.scheduler.alphas_cumprod[timesteps].to(device).float().view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (current_images_flat - beta_prod_t.sqrt() * predicted_noise) / alpha_prod_t.sqrt()
    
    pred_x0_reshaped = rearrange(pred_x0, "(b v) c h w -> b v c h w", b=b)
    
    return pred_x0_reshaped


def run_inference_diffusion_multistep(model, batch, config, device, amp_dtype, timestep_schedule):
    """Multi-step diffusion inference with recycling."""
    model_module = model.module if isinstance(model, DDP) else model
    
    input_data, target_data = model_module.process_data(
        batch, 
        has_target_image=True, 
        target_has_input=config.training.target_has_input, 
        compute_rays=True
    )
    
    b, v_target = target_data.image.shape[0], target_data.image.shape[1]
    h, w = target_data.image_h_w[0], target_data.image_h_w[1]
    
    posed_input_images = model_module.get_posed_input(
        images=input_data.image, 
        ray_o=input_data.ray_o, 
        ray_d=input_data.ray_d
    )
    _, v_input, _, _, _ = posed_input_images.size()
    input_img_tokens = model_module.image_tokenizer(posed_input_images)
    _, n_patches, d = input_img_tokens.size()
    input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)
    
    pure_noise = torch.randn((b, v_target, 3, h, w), device=device)
    
    first_timestep = timestep_schedule[0]
    timesteps = torch.full((b * v_target,), first_timestep, device=device, dtype=torch.long)
    zero_images = rearrange(target_data.image * 2.0 - 1.0, "b v c h w -> (b v) c h w")
    pure_noise_flat = rearrange(pure_noise, "b v c h w -> (b v) c h w")
    current_images_flat = model_module.scheduler.add_noise(zero_images, pure_noise_flat, timesteps)
    current_images = rearrange(current_images_flat, "(b v) c h w -> b v c h w", b=b, v=v_target)
    
    num_recycles = len(timestep_schedule)
    for recycle_idx, timestep in enumerate(timestep_schedule):
        denoised = denoise_single_step(
            model_module, input_img_tokens, target_data, timestep, 
            config, device, n_patches, v_input, v_target, current_images
        )
        
        if recycle_idx < num_recycles - 1:
            next_timestep = timestep_schedule[recycle_idx + 1]
            denoised_flat = rearrange(denoised, "b v c h w -> (b v) c h w")
            next_timesteps = torch.full((denoised_flat.shape[0],), next_timestep, device=device, dtype=torch.long)
            noise_to_add = torch.randn_like(denoised_flat)
            current_images_flat = model_module.scheduler.add_noise(denoised_flat, noise_to_add, next_timesteps)
            current_images = rearrange(current_images_flat, "(b v) c h w -> b v c h w", b=b, v=v_target)
        else:
            current_images = denoised
    
    rendered_images = torch.clamp(current_images, -1.0, 1.0) / 2.0 + 0.5
    
    loss_metrics = model_module.loss_computer(rendered_images, target_data.image)
    
    return edict(input=input_data, target=target_data, loss_metrics=loss_metrics, render=rendered_images)


def run_single_schedule_inference(config, model, dataloader, device, output_dir, schedule_name, schedule_info):
    """Run inference with a single schedule and return metrics."""
    model.eval()
    
    timestep_schedule = schedule_info['steps']
    
    os.makedirs(output_dir, exist_ok=True)
    
    compute_metrics = config.inference.get("compute_metrics", True)
    render_video = config.inference.get("render_video", False)
    use_amp = config.training.get("use_amp", False)
    amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, 'tf32': torch.float32}
    amp_dtype = amp_dtype_mapping[config.training.get("amp_dtype", "bf16")]
    
    total_samples = 0
    dataloader_iter = iter(dataloader)
    
    with torch.no_grad():
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(total=len(dataloader.dataset), disable=not is_main, desc=f"Testing {schedule_name}")
        
        while True:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break
            
            if batch is None:
                continue
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.autocast(enabled=use_amp, device_type="cuda", dtype=amp_dtype):
                result = run_inference_diffusion_multistep(
                    model, batch, config, device, amp_dtype, timestep_schedule
                )
            
            # Optionally render video (same as original)
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
            
            # Export results - EXACTLY as in original
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
    
    print_rank0(f"\nInference complete for {schedule_name}! Processed {total_samples} samples.")
    
    # NOW compute summary statistics - EXACTLY as in original
    metrics = {}
    if compute_metrics and (not dist.is_initialized() or dist.get_rank() == 0):
        print_rank0("Computing summary statistics...")
        summary_result = summarize_evaluation(output_dir)
        
        # summarize_evaluation might return metrics or save to file
        # Try to capture return value, or read from summary file
        if summary_result is not None:
            metrics = summary_result
        else:
            # Try reading from metrics.json or summary.json if it exists
            summary_files = ['metrics_summary.json', 'summary.json', 'metrics.json']
            for summary_file in summary_files:
                summary_path = os.path.join(output_dir, summary_file)
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        metrics = json.load(f)
                    break
    
    return metrics, total_samples


# ============================================================================
# COMPARISON AND VISUALIZATION
# ============================================================================

def create_comparison_plots(results, output_dir):
    """Create comprehensive comparison plots."""
    console = Console()
    
    if not results:
        console.print("[red]No results to plot![/red]")
        return
    
    # Extract data
    schedule_names = [r['name'] for r in results]
    psnr_values = [r['metrics'].get('psnr', 0) for r in results]
    ssim_values = [r['metrics'].get('ssim', 0) for r in results]
    lpips_values = [r['metrics'].get('lpips', 1) for r in results]
    num_steps = [len(r['steps']) for r in results]
    inference_times = [r['time'] for r in results]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. PSNR comparison
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(range(len(schedule_names)), psnr_values, color='skyblue', edgecolor='navy')
    ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('PSNR Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(schedule_names)))
    ax1.set_xticklabels(schedule_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    # Highlight best
    best_idx = psnr_values.index(max(psnr_values))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    ax1.axhline(y=max(psnr_values), color='red', linestyle='--', alpha=0.5, label=f'Best: {max(psnr_values):.2f}')
    ax1.legend()
    
    # 2. SSIM comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(range(len(schedule_names)), ssim_values, color='lightgreen', edgecolor='darkgreen')
    ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax2.set_title('SSIM Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(schedule_names)))
    ax2.set_xticklabels(schedule_names, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    best_idx = ssim_values.index(max(ssim_values))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # 3. LPIPS comparison (lower is better)
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(range(len(schedule_names)), lpips_values, color='lightcoral', edgecolor='darkred')
    ax3.set_ylabel('LPIPS', fontsize=12, fontweight='bold')
    ax3.set_title('LPIPS Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(schedule_names)))
    ax3.set_xticklabels(schedule_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    best_idx = lpips_values.index(min(lpips_values))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # 4. Steps vs PSNR scatter
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(num_steps, psnr_values, c=psnr_values, cmap='viridis', s=100, edgecolor='black')
    ax4.set_xlabel('Number of Steps', fontsize=12, fontweight='bold')
    ax4.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax4.set_title('Steps vs PSNR', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='PSNR')
    # Annotate best
    best_idx = psnr_values.index(max(psnr_values))
    ax4.annotate(schedule_names[best_idx], (num_steps[best_idx], psnr_values[best_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 5. Inference time comparison
    ax5 = plt.subplot(2, 3, 5)
    bars = ax5.bar(range(len(schedule_names)), inference_times, color='plum', edgecolor='purple')
    ax5.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax5.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(len(schedule_names)))
    ax5.set_xticklabels(schedule_names, rotation=45, ha='right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Efficiency (PSNR per second)
    ax6 = plt.subplot(2, 3, 6)
    efficiency = [psnr / time if time > 0 else 0 for psnr, time in zip(psnr_values, inference_times)]
    bars = ax6.bar(range(len(schedule_names)), efficiency, color='lightyellow', edgecolor='orange')
    ax6.set_ylabel('PSNR / Time', fontsize=12, fontweight='bold')
    ax6.set_title('Efficiency (PSNR per Second)', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(schedule_names)))
    ax6.set_xticklabels(schedule_names, rotation=45, ha='right', fontsize=8)
    ax6.grid(axis='y', alpha=0.3)
    best_idx = efficiency.index(max(efficiency))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'schedule_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Comparison plots saved to: {plot_path}[/green]")
    plt.close()
    
    # Create schedule visualization
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
        ax = axes[idx]
        steps = result['steps']
        ax.plot(range(len(steps)), steps, marker='o', linewidth=2, markersize=4)
        ax.set_title(f"{result['name']}\nPSNR: {result['metrics'].get('psnr', 0):.2f}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Step Index')
        ax.set_ylabel('Timestep')
        ax.grid(alpha=0.3)
        ax.set_ylim([-50, 1050])
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    schedule_viz_path = os.path.join(output_dir, 'schedule_visualization.png')
    plt.savefig(schedule_viz_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Schedule visualization saved to: {schedule_viz_path}[/green]")
    plt.close()


def print_results_table(results):
    """Print comprehensive results table."""
    console = Console()
    
    table = Table(title="Schedule Comparison Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Schedule", style="green")
    table.add_column("Steps", justify="center")
    table.add_column("PSNR ↑", justify="center")
    table.add_column("SSIM ↑", justify="center")
    table.add_column("LPIPS ↓", justify="center")
    table.add_column("Time (s)", justify="center")
    table.add_column("Efficiency", justify="center")
    
    # Sort by PSNR
    sorted_results = sorted(results, key=lambda x: x['metrics'].get('psnr', 0), reverse=True)
    
    for rank, result in enumerate(sorted_results, 1):
        metrics = result['metrics']
        psnr = metrics.get('psnr', 0)
        ssim = metrics.get('ssim', 0)
        lpips = metrics.get('lpips', 1)
        time_val = result['time']
        efficiency = psnr / time_val if time_val > 0 else 0
        
        style = "bold yellow" if rank == 1 else ""
        
        table.add_row(
            f"#{rank}",
            result['name'],
            str(len(result['steps'])),
            f"{psnr:.3f}",
            f"{ssim:.4f}",
            f"{lpips:.4f}",
            f"{time_val:.1f}",
            f"{efficiency:.3f}",
            style=style
        )
    
    console.print(table)
    
    # Print winner details
    best = sorted_results[0]
    console.print(f"\n[bold green]🏆 WINNER: {best['name']}[/bold green]")
    console.print(f"[yellow]   Steps: {best['steps'][:10]}{'...' if len(best['steps']) > 10 else ''}[/yellow]")
    console.print(f"[yellow]   Description: {best['description']}[/yellow]")


# ============================================================================
# MAIN TESTING LOOP
# ============================================================================

def main():
    console = Console()
    
    # Parse command line arguments for config path
    parser = argparse.ArgumentParser(description='Comprehensive schedule testing')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (e.g., configs/my_config.yaml)')
    args = parser.parse_args()
    
    config = init_config()
    
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    ddp_info = init_distributed(seed=777)
    
    if dist.is_initialized():
        dist.barrier()
    
    torch.backends.cuda.matmul.allow_tf32 = config.training.get("use_tf32", True)
    torch.backends.cudnn.allow_tf32 = config.training.get("use_tf32", True)
    
    base_output_dir = "./schedule_comparison_output"
    os.makedirs(base_output_dir, exist_ok=True)
    
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]COMPREHENSIVE SCHEDULE TESTING[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")
    
    # Initialize dataloader (once)
    print_rank0("Setting up dataloader...")
    job_id = os.environ.get("SLURM_JOB_ID", f"schedule_test_{int(time.time())}")
    cache_dir = f"/dev/shm/litdata_cache_schedule_test_{job_id}"
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(cache_dir, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    dataset = ld.StreamingDataset(
        input_dir=config.training.dataset_path,
        cache_dir=cache_dir,
        max_cache_size="50GB",
        shuffle=False,
    )
    
    collate_fn = LitDataCollateInference(config)
    dataloader = ld.StreamingDataLoader(
        dataset,
        batch_size=config.training.get("batch_size_per_gpu", 1),
        num_workers=config.training.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    # Initialize model (once)
    print_rank0("Loading model...")
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)
    
    checkpoint_path = config.training.get("checkpoint_dir", "./checkpoints")
    load_result = model.load_ckpt(checkpoint_path)
    if load_result is None:
        print_rank0("[red]ERROR: Failed to load checkpoint![/red]")
        exit(1)
    
    if dist.is_initialized() and ddp_info.world_size > 1:
        model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    # Generate test schedules
    test_schedules = generate_test_schedules()
    
    console.print(f"[green]Testing {len(test_schedules)} different schedules...[/green]\n")
    
    # Run tests
    all_results = []
    
    for schedule_name, schedule_info in test_schedules.items():
        console.print(f"\n[bold blue]Testing: {schedule_name}[/bold blue]")
        console.print(f"[yellow]Description: {schedule_info['description']}[/yellow]")
        console.print(f"[yellow]Steps: {len(schedule_info['steps'])} total[/yellow]")
        
        schedule_output_dir = os.path.join(base_output_dir, schedule_name)
        
        start_time = time.time()
        metrics, num_samples = run_single_schedule_inference(
            config, model, dataloader, ddp_info.device, 
            schedule_output_dir, schedule_name, schedule_info
        )
        elapsed_time = time.time() - start_time
        
        all_results.append({
            'name': schedule_name,
            'description': schedule_info['description'],
            'steps': schedule_info['steps'],
            'metrics': metrics,
            'time': elapsed_time,
            'num_samples': num_samples
        })
        
        print_rank0(f"[green]✓ Completed {schedule_name} in {elapsed_time:.1f}s[/green]")
        print_rank0(f"[cyan]  PSNR: {metrics.get('psnr', 0):.3f}, SSIM: {metrics.get('ssim', 0):.4f}, LPIPS: {metrics.get('lpips', 1):.4f}[/cyan]")
        
        # Reset dataloader for next schedule
        dataloader_iter = iter(dataloader)
        torch.cuda.empty_cache()
    
    # Save results to JSON
    results_path = os.path.join(base_output_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[green]✓ Results saved to: {results_path}[/green]")
    
    # Print comparison table
    console.print("\n")
    print_results_table(all_results)
    
    # Create visualizations
    if not dist.is_initialized() or dist.get_rank() == 0:
        console.print("\n[cyan]Generating comparison plots...[/cyan]")
        create_comparison_plots(all_results, base_output_dir)
    
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold green]TESTING COMPLETE![/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")
    
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
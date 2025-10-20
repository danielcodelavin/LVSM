import importlib
import os
import time
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
import litdata as ld
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random


class LitDataCollateInference:
    """
    Collate function for inference - similar to training but without view sampling.
    Processes all available views in the sample.
    """
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
            
            # View selection logic (same as training)
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


def run_inference(config, model, dataloader, device, output_dir):
    """
    Run inference on entire dataset and compute metrics.
    
    Args:
        config: Configuration object
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        output_dir: Directory to save results
    """
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print_rank0(f"Output directory: {os.path.abspath(output_dir)}")
    
    compute_metrics = config.inference.get("compute_metrics", True)
    render_video = config.inference.get("render_video", False)
    
    # Get AMP settings from training config
    amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, 'tf32': torch.float32}
    use_amp = config.training.get("use_amp", False)
    amp_dtype = amp_dtype_mapping[config.training.get("amp_dtype", "bf16")]
    
    total_samples = 0
    dataloader_iter = iter(dataloader)
    
    print_rank0(f"Starting inference on dataset...")
    print_rank0(f"Use AMP: {use_amp}, AMP dtype: {config.training.get('amp_dtype', 'bf16')}")
    print_rank0(f"Compute metrics: {compute_metrics}")
    print_rank0(f"Render video: {render_video}")
    
    with torch.no_grad():
        # Use tqdm only on rank 0
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(total=len(dataloader.dataset), disable=not is_main, desc="Inference")
        
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
            
            # Export results (images, metrics, videos)
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
    # Initialize configuration
    config = init_config()
    
    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    
    # Initialize distributed training (if applicable)
    ddp_info = init_distributed(seed=777)
    
    if dist.is_initialized():
        dist.barrier()
    
    # Set TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = config.training.get("use_tf32", True)
    torch.backends.cudnn.allow_tf32 = config.training.get("use_tf32", True)
    
    # Get output directory from config
    output_dir = config.inference.get("inference_out_dir", config.get("inference_out_dir", "./inference_output"))
    
    print_rank0(f"Initializing inference...")
    print_rank0(f"Dataset path: {config.training.dataset_path}")
    print_rank0(f"Output directory: {output_dir}")
    
    # Setup dataloader
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
        shuffle=False,  # Don't shuffle for inference
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
        drop_last=False,  # Don't drop last batch in inference
    )
    
    print_rank0(f"Dataloader ready. Dataset size: {len(dataset)}")
    
    # Initialize model
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)
    
    # Load checkpoint
    checkpoint_path = config.training.get("checkpoint_dir", "./checkpoints")
    print_rank0(f"Loading checkpoint from: {checkpoint_path}")
    
    load_result = model.load_ckpt(checkpoint_path)
    if load_result is None:
        print_rank0("ERROR: Failed to load checkpoint!")
        if dist.is_initialized():
            dist.barrier()
        exit(1)
    
    # Wrap with DDP if using distributed
    if dist.is_initialized() and ddp_info.world_size > 1:
        model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)
        print_rank0(f"Using DDP with world size: {ddp_info.world_size}")
    
    if dist.is_initialized():
        dist.barrier()
    
    # Run inference
    start_time = time.time()
    total_samples = run_inference(
        config=config,
        model=model,
        dataloader=dataloader,
        device=ddp_info.device,
        output_dir=output_dir
    )
    elapsed_time = time.time() - start_time
    
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Inference completed successfully!")
    print_rank0(f"Total samples processed: {total_samples}")
    print_rank0(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
    print_rank0(f"Average time per sample: {elapsed_time/total_samples:.2f}s")
    print_rank0(f"Results saved to: {os.path.abspath(output_dir)}")
    print_rank0(f"{'='*60}\n")
    
    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
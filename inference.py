import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()

# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(config.training.checkpoint_dir)


if ddp_info.is_main_process:  
    print(f"Running inference; save results to: {config.inference_out_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = {k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get model predictions
        result = model(batch)
        
        # For diffusion models, we need to render images using the diffusion process
        if config.model.get("use_diffusion", False):
            # Configure rendering parameters
            render_config = {
                "traj_type": "target",
                "num_frames": 1,
                "loop_video": False,
                "order_poses": False,
                "num_steps": config.inference.get("diffusion_steps", 50),  # Number of denoising steps
                "cfg_scale": config.inference.get("cfg_scale", 1.0),       # Classifier-free guidance scale
                "eta": config.inference.get("eta", 0.0)                   # Controls stochasticity
            }
            
            
            result = model.module.render_video(result, **render_config)
            
            # Extract the rendered image from video_rendering
            if hasattr(result, 'video_rendering') and result.video_rendering is not None:
                result.render = result.video_rendering.squeeze(1)  # Remove frame dimension
        
        
        if config.inference.get("compute_metrics", False):
            # Initialize metrics dictionary if not present
            if not hasattr(result, 'metrics') or result.metrics is None:
                result.metrics = {}
                
            # Compute metrics if we have ground truth
            if hasattr(result.target, 'image') and result.target.image is not None and result.render is not None:
                # Use the loss computer to calculate metrics
                with torch.no_grad():
                    metrics = model.module.loss_computer(
                        rendering=result.render, 
                        target=result.target.image,
                        predicted_noise=None,
                        noise=None
                    )
                    
                    # Copy metrics to result
                    for k, v in metrics.items():
                        if k != 'loss':  # Skip the loss value itself
                            result.metrics[k] = v
        
        # Export results including metrics
        export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
    
    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)
import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
import time
import litdata as ld
import random
import torchvision.transforms.functional as TF



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

class LitDataCollate:
    """
    A robust collate function that filters each batch for valid samples,
    performs view selection, resizes images to the target resolution,
    and then stacks them into a final batch tensor.
    """
    def __init__(self, config):
        self.config = config
        # Get the target image size from the model's config to avoid hardcoding.
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

                    # 2. Resize each selected image to the target size from the config.
                    resized_images = torch.stack(
                        [TF.resize(img, [self.target_size, self.target_size], antialias=True) for img in original_images]
                    )


                    processed_batch[key].append(resized_images)


                elif isinstance(value, torch.Tensor):
                    # For other tensors (like camera poses), just select the correct indices.
                    processed_batch[key].append(value[image_indices])
                elif key == "scene_name":
                    # For metadata like scene_name, just append it.
                    processed_batch[key].append(value)

        if not processed_batch.get("image"):
            return None

        # Stack all samples into a single batch tensor.
        final_batch = {}
        for key, value_list in processed_batch.items():
            if value_list and isinstance(value_list[0], torch.Tensor):
                final_batch[key] = torch.stack(value_list)
            else:
                final_batch[key] = value_list
        return final_batch

# --- Dataloader Setup ---
print("Initializing LitData StreamingDataset for inference...")
job_id = os.environ.get("SLURM_JOB_ID", f"local_inference_{int(time.time())}")

# --- RAM CACHING IMPLEMENTATION ---
cache_dir = f"/dev/shm/litdata_cache_{job_id}"
if ddp_info.is_main_process:
    os.makedirs(cache_dir, exist_ok=True)
dist.barrier()
print(f"Using RAM Disk cache directory for inference: {cache_dir}")

dataset = ld.StreamingDataset(
    input_dir=config.training.dataset_path,
    cache_dir=cache_dir,
    max_cache_size="50GB",
    shuffle=False, # Do not shuffle for inference
)

if len(dataset) == 0:
    print(f"\nFATAL ERROR: LitData dataset is empty. Check path: {os.path.abspath(config.training.dataset_path)}")
    dist.barrier()
    exit(1)

collate_fn = LitDataCollate(config)
dataloader = ld.StreamingDataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    num_workers=config.training.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False, # Do not drop last batch for inference
)
print(f"Dataloader is ready. Found {len(dataset)} total items.")
# --- End Dataloader Setup ---

dist.barrier()

# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(config.training.checkpoint_dir)


if ddp_info.is_main_process:
    #print(f"Running inference; save results to: {config.inference_out_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for i, batch in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}...")
        if batch is None:
            print(f"Skipping empty batch {i+1}.")
            continue
            
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        
        
        need_target_images = config.inference.get("compute_metrics", False)

        result = model(batch)
      
        
        #result = model.module.render_video_diffusion(result, **config.inference.render_video_config)
        export_results(result, config.inference_out_dir, compute_metrics=True)
    torch.cuda.empty_cache()
    summarize_evaluation(config.inference_out_dir)

dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)
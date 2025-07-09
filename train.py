import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0
import random
import litdata as ld
from fvcore.nn import FlopCountAnalysis


def analyze_flops(model, dataloader, config, ddp_info, amp_dtype_mapping):
    """
    A self-contained function to calculate GFLOPs and log them to wandb.
    """
    if not ddp_info.is_main_process:
        return

    print_rank0("--- Starting GFLOPs Analysis ---")
    
    sample_batch = None
    dataloader_iter = iter(dataloader)
    while sample_batch is None:
        try:
            # The collate_fn can return None, so we loop until we get a valid batch
            sample_batch = next(dataloader_iter)
        except StopIteration:
            print_rank0("Could not get a batch for FLOPs analysis. Dataloader exhausted.")
            return
    
    sample_batch = {k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

    model_for_flops = model.module if isinstance(model, DDP) else model
    model_for_flops.eval()

    original_pass_layers = model_for_flops.pass_layers
    model_for_flops.pass_layers = lambda tokens, **kwargs: original_pass_layers(tokens, gradient_checkpoint=False)

    try:
        with torch.no_grad(), torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            flops_analyzer = FlopCountAnalysis(model_for_flops, (sample_batch,))
            total_flops = flops_analyzer.total()
            
        gmacs = total_flops / 1e9
        gflops = gmacs * 2

        print_rank0(f"Model Configuration: {config.training.num_input_views} Input Views, {config.training.num_target_views} Target Views")
        print_rank0(f"Autocast dtype: {str(amp_dtype_mapping[config.training.amp_dtype])}")
        print_rank0(f"Total MACs for one forward pass: {gmacs:.4f} G")
        print_rank0(f"Estimated GFLOPs: {gflops:.4f} G")
        wandb.summary["GFLOPs"] = gflops

    finally:
        model_for_flops.pass_layers = original_pass_layers
        model_for_flops.train()
        print_rank0("--- GFLOPs Analysis Complete ---")


class LitDataCollate:
    """
    A robust collate function that filters each batch for valid samples
    before performing view selection and stacking.
    """
    def __init__(self, config):
        self.config = config

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
                if isinstance(value, torch.Tensor):
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


class ViewManager:
    def __init__(self, config):
        self.config = config
        self.original_values = {
            'num_views': config.training.num_views,
            'num_input_views': config.training.num_input_views,
            'num_target_views': config.training.num_target_views
        }
        self.variable_enabled = config.training.variable_amount_of_views
    
    def randomize_for_batch(self):
        if not self.variable_enabled: return
        num_views = 8  
        num_input_views = random.randint(1, 4)
        num_target_views = num_views - num_input_views
        self.config.training.num_views = num_views
        self.config.training.num_input_views = num_input_views
        self.config.training.num_target_views = num_target_views
        return {'num_views': num_views, 'num_input_views': num_input_views, 'num_target_views': num_target_views}
    
    def reset_to_original(self):
        if not self.variable_enabled: return
        for key, value in self.original_values.items():
            setattr(self.config.training, key, value)


# --- Main Execution ---
config = init_config()
view_manager = ViewManager(config)
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
ddp_info = init_distributed(seed=777)
dist.barrier()

if ddp_info.is_main_process:
    init_wandb_and_backup(config)
dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, 'tf32': torch.float32}

# --- Dataloader Setup ---
print_rank0("Initializing LitData StreamingDataset...")
job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")

# --- RAM CACHING IMPLEMENTATION ---
# Use the RAM Disk for caching to avoid HDD bottlenecks.
cache_dir = f"/dev/shm/litdata_cache_{job_id}"
if ddp_info.is_main_process:
    os.makedirs(cache_dir, exist_ok=True)
dist.barrier()
print_rank0(f"Using RAM Disk cache directory: {cache_dir}")

dataset = ld.StreamingDataset(
    input_dir=config.training.dataset_path,
    cache_dir=cache_dir,
    max_cache_size="50GB",  # Allocate 50GB of RAM for caching
    shuffle=True,
)
# --- END RAM CACHING IMPLEMENTATION ---


if len(dataset) == 0:
    print_rank0(f"\nFATAL ERROR: LitData dataset is empty. Check path: {os.path.abspath(config.training.dataset_path)}")
    dist.barrier()
    exit(1)

collate_fn = LitDataCollate(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

dataloader = ld.StreamingDataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    num_workers=config.training.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
)
dataloader_iter = iter(dataloader)
print_rank0(f"Dataloader is ready. Found {len(dataset)} total items.")
# --- End Dataloader Setup ---

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset)) if len(dataset) > 0 else 1

module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])

optimizer, optimized_param_dict, _ = create_optimizer(
    model, config.training.weight_decay, config.training.lr, (config.training.beta1, config.training.beta2)
)
optim_param_list = list(optimized_param_dict.values())

scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(optimizer, total_param_update_steps, config.training.warmup, scheduler_type=scheduler_type)

reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    config.training.checkpoint_dir, model, optimizer, lr_scheduler, reset_training_state
)

if not reset_training_state and cur_train_step > 0:
    dataloader_state_path = os.path.join(config.training.checkpoint_dir, f"dataloader_{cur_train_step:016}.pt")
    if os.path.exists(dataloader_state_path):
        print_rank0(f"Resuming dataloader state from {dataloader_state_path}")
        dataloader.load_state_dict(torch.load(dataloader_state_path))
    else:
        print_rank0(f"WARNING: Could not find dataloader state for step {cur_train_step}. Dataset will start from beginning.")

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

analyze_flops(model, dataloader, config, ddp_info, amp_dtype_mapping)

start_train_step = cur_train_step
model.train()

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset)) if len(dataset) > 0 else 0
    
    view_info = view_manager.randomize_for_batch()
    if view_info and ddp_info.is_main_process and cur_train_step % config.training.print_every == 0:
        print(f"Using random views: input={view_info['num_input_views']}, target={view_info['num_target_views']}")
    
    batch = None
    while batch is None:
        try:
            data = next(dataloader_iter)
            if data is not None:
                batch = {k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        except StopIteration:
            print_rank0(f"Dataloader iterator finished. Resetting for new epoch.")
            dataloader_iter = iter(dataloader)

    with torch.autocast(enabled=config.training.use_amp, device_type="cuda", dtype=amp_dtype_mapping[config.training.amp_dtype]):
        ret_dict = model(batch)

    loss_to_backward = ret_dict.loss_metrics.loss / grad_accum_steps
    scaler.scale(loss_to_backward).backward()
    cur_train_step += 1

    update_grads = (cur_train_step % grad_accum_steps == 0) or (cur_train_step == total_train_steps)
    export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)
    skip_optimizer_step = torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss)
    
    if skip_optimizer_step:
        print(f"NaN or Inf loss detected, skipping optimizer update.")
    
    total_grad_norm = None
    if update_grads and not skip_optimizer_step:
        scaler.unscale_(optimizer)

        if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.detach().norm().item()
            for layer_name, grad_norm in grad_norms.items():
                wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

        total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()
        
        display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0
        if display_grad_norm and ddp_info.is_main_process:
             wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)
        
        allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
        if total_grad_norm > allowed_gradnorm:
            print(f"WARNING: grad norm {total_grad_norm} too large, skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
        else:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        cur_param_update_step += 1

    if ddp_info.is_main_process:
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
            print(f"[Epoch {cur_epoch}] | Step: {cur_train_step} | Param Step: {cur_param_update_step} | LR: {optimizer.param_groups[0]['lr']:.6f} | loss: {loss_dict.get('loss', -1):.4f}")

        if (cur_train_step % config.training.wandb_log_every == 0) or (cur_train_step < 200 + start_train_step):
            log_dict = {
                "iter": cur_train_step, 
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm if total_grad_norm is not None else 0.0,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in ret_dict.loss_metrics.items()})
            wandb.log(log_dict, step=cur_train_step)

        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            model_weights = model.module.state_dict()
            checkpoint = {
                "model": model_weights, "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(), "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            dataloader_ckpt_path = os.path.join(config.training.checkpoint_dir, f"dataloader_{cur_train_step:016}.pt")
            
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, ckpt_path)
            torch.save(dataloader.state_dict(), dataloader_ckpt_path)
            print(f"Saved checkpoint and dataloader state at step {cur_train_step} to {os.path.abspath(ckpt_path)}")
        
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict)
            model.train()

    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()
        
dist.barrier()
dist.destroy_process_group()
import os
import time
import re
import logging
import subprocess
import glob
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("checkpoint_evaluator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# HARDCODED PARAMETERS
CHECKPOINT_DIR = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/experiments/checkpoints/LVSM_scene_decoder_only"  # Directory containing training checkpoints
RESULTS_BASE_DIR = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/evaluation_live_lvsm"              # Base directory for evaluation results
CONFIG_FILE = "configs/LVSM_scene_decoder_only.yaml"                                                 # Config file path (relative to project root)
DATASET_PATH = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/re10k/test/full_list.txt"       # Dataset file
METRICS_FILE = os.path.join(RESULTS_BASE_DIR, "checkpoint_metrics.csv")                                # File to save all metrics
SLEEP_TIME = 6                                                                                       # Time (in seconds) between scans
NUM_GPUS = 1                                                                                         # Number of GPUs to use for evaluation

def extract_checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    # Try matching "epoch_" pattern
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    # Try matching "ckpt_" pattern (skipping any leading zeros)
    match = re.search(r'ckpt_0*([\d]+)', filename)
    if match:
        return int(match.group(1))
    return None

def is_checkpoint_complete(checkpoint_path):
    try:
        size1 = os.path.getsize(checkpoint_path)
        time.sleep(3)
        size2 = os.path.getsize(checkpoint_path)
        return size1 == size2
    except Exception as e:
        logger.error(f"Error checking if checkpoint is complete: {e}")
        return False

def get_all_checkpoints(checkpoint_dir):
    # Collect checkpoint files (non-recursive) for given extensions.
    checkpoint_files = []
    for ext in ["*.pt", "*.pth", "*.ckpt"]:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoint_dir, ext)))
    
    checkpoints = []
    for path in checkpoint_files:
        epoch = extract_checkpoint_epoch(path)
        if epoch is not None:
            checkpoints.append((epoch, path))
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def run_inference(checkpoint_path, epoch, checkpoint_output_dir, config_file, dataset_path, num_gpus):
    # Create output directory for this checkpoint's evaluation
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    
    # Use fixed absolute path for inference.py
    inference_script = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/inference.py"
    if not os.path.exists(inference_script):
        logger.error(f"Inference script not found at: {inference_script}")
        return False

    # Convert config file to an absolute path using the project root
    project_root = "/home/stud/lavingal/storage/slurm/lavingal/LVSM"
    if not os.path.isabs(config_file):
        config_file_absolute = os.path.join(project_root, config_file)
    else:
        config_file_absolute = config_file
    if not os.path.exists(config_file_absolute):
        logger.error(f"Config file not found at: {config_file_absolute}")
        return False

    # Generate a unique port number to avoid conflicts
    port = 29500 + (epoch % 1000)
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--nnodes=1",
        f"--rdzv_id={epoch}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{port}",
        inference_script,
        f"--config={config_file_absolute}",
        f"training.dataset_path={dataset_path}",
        "training.batch_size_per_gpu=4",
        "training.target_has_input=false",
        "training.num_views=5",
        "training.square_crop=true",
        "training.num_input_views=2",
        "training.num_target_views=3",
        "inference.if_inference=true",
        "inference.compute_metrics=true",
        "inference.render_video=false",
        "inference.generate_website=false",
        f"inference_out_dir={checkpoint_output_dir}",
        f"training.checkpoint_dir={checkpoint_path}"
    ]
    
    logger.info(f"Running inference on checkpoint from epoch {epoch}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Inference failed for epoch {epoch}: {result.stderr}")
            return False
        logger.info(f"Inference completed for epoch {epoch}")
        return True
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        return False

def collect_metrics_from_subfolders(checkpoint_output_dir):
    """
    Collects metrics from all subfolders (expected to be numerically named) within the checkpoint output directory.
    """
    subfolders = [
        os.path.join(checkpoint_output_dir, d)
        for d in os.listdir(checkpoint_output_dir)
        if os.path.isdir(os.path.join(checkpoint_output_dir, d)) and d.isdigit()
    ]
    
    if not subfolders:
        logger.error(f"No sample subfolders found in {checkpoint_output_dir}")
        return None

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            logger.warning(f"Metrics file not found in {subfolder}, skipping...")
            continue
            
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                summary = data.get("summary", {})
                psnr_values.append(summary.get("psnr", 0))
                ssim_values.append(summary.get("ssim", 0))
                lpips_values.append(summary.get("lpips", 0))
            except json.JSONDecodeError as e:
                logger.error(f"Error reading metrics from {json_path}: {e}")

    if not psnr_values:
        logger.error("No valid metrics found in any subfolder")
        return None
    
    avg_metrics = {
        "PSNR": sum(psnr_values) / len(psnr_values),
        "SSIM": sum(ssim_values) / len(ssim_values),
        "LPIPS": sum(lpips_values) / len(lpips_values)
    }
    
    with open(os.path.join(checkpoint_output_dir, "evaluation_summary.txt"), "w") as f:
        f.write(f"PSNR: {avg_metrics['PSNR']:.6f}\n")
        f.write(f"SSIM: {avg_metrics['SSIM']:.6f}\n")
        f.write(f"LPIPS: {avg_metrics['LPIPS']:.6f}\n")
    
    return avg_metrics

def save_metrics_to_file(metrics_file, epoch, metrics):
    header = "Epoch,PSNR,SSIM,LPIPS\n"
    line = f"{epoch},{metrics.get('PSNR', 'N/A'):.6f},{metrics.get('SSIM', 'N/A'):.6f},{metrics.get('LPIPS', 'N/A'):.6f}\n"
    
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write(header)
    
    with open(metrics_file, 'a') as f:
        f.write(line)
    
    logger.info(f"Metrics saved for epoch {epoch}: PSNR={metrics.get('PSNR', 'N/A'):.6f}, SSIM={metrics.get('SSIM', 'N/A'):.6f}, LPIPS={metrics.get('LPIPS', 'N/A'):.6f}")

def process_pending_checkpoints(evaluated_checkpoints):
    """Process all available checkpoints that have not yet been evaluated."""
    checkpoints = get_all_checkpoints(CHECKPOINT_DIR)
    logger.info(f"Found {len(checkpoints)} checkpoint(s) in {CHECKPOINT_DIR}")
    
    for epoch, checkpoint_path in checkpoints:
        if checkpoint_path in evaluated_checkpoints:
            continue
        
        if not is_checkpoint_complete(checkpoint_path):
            logger.info(f"Checkpoint {checkpoint_path} is still being written. Skipping for now.")
            continue
        
        checkpoint_output_dir = os.path.join(RESULTS_BASE_DIR, f"epoch_{epoch}")
        success = run_inference(
            checkpoint_path, 
            epoch, 
            checkpoint_output_dir,
            CONFIG_FILE,
            DATASET_PATH,
            NUM_GPUS
        )
        
        if success:
            metrics = collect_metrics_from_subfolders(checkpoint_output_dir)
            if metrics:
                save_metrics_to_file(METRICS_FILE, epoch, metrics)
            evaluated_checkpoints.add(checkpoint_path)
    
    return evaluated_checkpoints

def main():
    # Create output directory if it doesn't exist
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    
    # Keep track of evaluated checkpoints
    evaluated_checkpoints = set()
    
    logger.info("Starting initial evaluation of all existing checkpoints.")
    evaluated_checkpoints = process_pending_checkpoints(evaluated_checkpoints)
    
    logger.info(f"Initial checkpoint evaluation complete. Monitoring for new checkpoints in {CHECKPOINT_DIR}")
    
    while True:
        try:
            evaluated_checkpoints = process_pending_checkpoints(evaluated_checkpoints)
            logger.info(f"Sleeping for {SLEEP_TIME} seconds before rechecking...")
            time.sleep(SLEEP_TIME)
        except KeyboardInterrupt:
            logger.info("Evaluation stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)  # Wait a bit before retrying

if __name__ == "__main__":
    main()
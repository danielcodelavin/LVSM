import os
import time
import re
import logging
import subprocess
import glob
import json
import csv
import pandas as pd

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
METRICS_FILE = os.path.join(RESULTS_BASE_DIR, "checkpoint_metrics.csv")                              # File to save all metrics
SLEEP_TIME = 60                                                                                      # Time (in seconds) between scans
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

def load_already_evaluated_epochs(metrics_file):
    """
    Load a list of already evaluated epochs from the metrics CSV file.
    
    Args:
        metrics_file: Path to the metrics CSV file
        
    Returns:
        A set of already evaluated epoch numbers
    """
    evaluated_epochs = set()
    
    if not os.path.exists(metrics_file):
        logger.info(f"No existing metrics file found at {metrics_file}")
        return evaluated_epochs
    
    try:
        # Try using pandas for robust CSV reading
        df = pd.read_csv(metrics_file)
        if 'Epoch' in df.columns:
            evaluated_epochs = set(df['Epoch'].astype(int).tolist())
            logger.info(f"Found {len(evaluated_epochs)} already evaluated epochs in metrics file")
        else:
            logger.warning(f"Metrics file exists but does not contain 'Epoch' column")
    except Exception as e:
        # Fallback to simple CSV reading if pandas fails or isn't available
        logger.warning(f"Error using pandas to read metrics file: {e}. Falling back to CSV reader.")
        try:
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Epoch' in row:
                        evaluated_epochs.add(int(row['Epoch']))
            logger.info(f"Found {len(evaluated_epochs)} already evaluated epochs in metrics file")
        except Exception as e2:
            logger.error(f"Error reading metrics file: {e2}")
    
    return evaluated_epochs

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
    
    # Prepare the command with diffusion model specific parameters
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
        "training.batch_size_per_gpu=16",
        "training.target_has_input=false",
        "training.num_views=5",
        "training.square_crop=true",
        "training.num_input_views=2",
        "training.num_target_views=3",
        "inference.if_inference=true",
        "inference.compute_metrics=true",
        "inference.render_video=true",
        "inference.generate_website=false",
        
        # Diffusion model parameters
        "model.diffusion.use_diffusion=true",
        "inference.use_sampling=true",
        "inference.diffusion_steps=50",
        "inference.eta=0.0",
        
        # Render video config
        "inference.render_video_config.traj_type=target",
        "inference.render_video_config.num_frames=1",
        "inference.render_video_config.loop_video=false",
        "inference.render_video_config.order_poses=false",
        "inference.render_video_config.num_inference_steps=50",  # Same as diffusion_steps
        
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

    metrics_collection = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'perceptual_loss': [],
        'diffusion_loss': []  # Added diffusion loss
    }

    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            logger.warning(f"Metrics file not found in {subfolder}, skipping...")
            continue
            
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                summary = data.get("summary", {})
                
                # Collect standard metrics
                for metric in ['psnr', 'ssim', 'lpips', 'perceptual_loss']:
                    if metric in summary:
                        metrics_collection[metric].append(summary[metric])
                
                # Collect diffusion-specific metrics
                if 'diffusion_loss' in summary:
                    metrics_collection['diffusion_loss'].append(summary['diffusion_loss'])
                
            except json.JSONDecodeError as e:
                logger.error(f"Error reading metrics from {json_path}: {e}")

    # Check if we found any metrics
    if not any(metrics_collection.values()):
        logger.error("No valid metrics found in any subfolder")
        return None
    
    # Calculate average for each available metric
    avg_metrics = {}
    for metric, values in metrics_collection.items():
        if values:  # Only calculate average if we have values
            avg_metrics[metric.upper()] = sum(values) / len(values)
    
    # Write metrics to summary file
    with open(os.path.join(checkpoint_output_dir, "evaluation_summary.txt"), "w") as f:
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")
    
    return avg_metrics

def save_metrics_to_file(metrics_file, epoch, metrics):
    """
    Save metrics for an epoch to the metrics CSV file.
    
    Args:
        metrics_file: Path to the metrics CSV file
        epoch: Epoch number
        metrics: Dictionary of metrics
    """
    # Define all possible metrics (standard + diffusion-specific)
    all_possible_metrics = ["Epoch", "PSNR", "SSIM", "LPIPS", "PERCEPTUAL_LOSS", "DIFFUSION_LOSS"]
    
    # Filter to include only metrics we actually have
    available_metrics = ["Epoch"] + [m for m in all_possible_metrics[1:] if m in metrics]
    
    # Create header
    header = ",".join(available_metrics) + "\n"
    
    # Create line with metric values
    line_items = [f"{epoch}"]
    for metric in available_metrics[1:]:  # Skip "Epoch"
        line_items.append(f"{metrics.get(metric, 'N/A'):.6f}")
    line = ",".join(line_items) + "\n"
    
    # Write to file
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write(header)
    else:
        # Check if we need to update the header (new metrics might be available)
        try:
            with open(metrics_file, 'r') as f:
                existing_header = f.readline().strip()
            
            existing_metrics = existing_header.split(',')
            new_metrics = [m for m in available_metrics if m not in existing_metrics]
            
            if new_metrics:
                # Need to update header and add N/A for previous epochs
                df = pd.read_csv(metrics_file)
                for metric in new_metrics:
                    df[metric] = "N/A"
                
                # Write updated CSV with new header
                df.to_csv(metrics_file, index=False)
                
                # Now append our new data
                with open(metrics_file, 'a') as f:
                    f.write(line)
                return
        except Exception as e:
            logger.warning(f"Error checking existing header: {e}. Will append normally.")
    
    # Normal append
    with open(metrics_file, 'a') as f:
        f.write(line)
    
    # Log metrics
    log_message = f"Metrics saved for epoch {epoch}: "
    log_entries = []
    for metric in available_metrics[1:]:  # Skip "Epoch"
        if metric in metrics:
            log_entries.append(f"{metric}={metrics[metric]:.6f}")
    log_message += ", ".join(log_entries)
    logger.info(log_message)

def process_pending_checkpoints(evaluated_checkpoints, evaluated_epochs):
    """
    Process all available checkpoints that have not yet been evaluated.
    
    Args:
        evaluated_checkpoints: Set of checkpoint paths that have already been evaluated
        evaluated_epochs: Set of epoch numbers that have already been evaluated
        
    Returns:
        Updated set of evaluated checkpoints
    """
    checkpoints = get_all_checkpoints(CHECKPOINT_DIR)
    logger.info(f"Found {len(checkpoints)} checkpoint(s) in {CHECKPOINT_DIR}")
    
    for epoch, checkpoint_path in checkpoints:
        # Skip if the checkpoint has already been evaluated
        if checkpoint_path in evaluated_checkpoints:
            continue
        
        # Skip if the epoch has already been evaluated (from CSV file)
        if epoch in evaluated_epochs:
            logger.info(f"Epoch {epoch} already evaluated according to metrics file. Skipping.")
            evaluated_checkpoints.add(checkpoint_path)
            continue
        
        # Skip if the checkpoint is still being written
        if not is_checkpoint_complete(checkpoint_path):
            logger.info(f"Checkpoint {checkpoint_path} is still being written. Skipping for now.")
            continue
        
        # Run inference on this checkpoint
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
            # Collect and save metrics
            metrics = collect_metrics_from_subfolders(checkpoint_output_dir)
            if metrics:
                save_metrics_to_file(METRICS_FILE, epoch, metrics)
                evaluated_epochs.add(epoch)  # Add to evaluated epochs
            evaluated_checkpoints.add(checkpoint_path)
    
    return evaluated_checkpoints

def main():
    # Create output directory if it doesn't exist
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    
    # Load already evaluated epochs from the metrics file
    evaluated_epochs = load_already_evaluated_epochs(METRICS_FILE)
    logger.info(f"Found {len(evaluated_epochs)} already evaluated epochs from metrics file")
    
    # Keep track of evaluated checkpoints
    evaluated_checkpoints = set()
    
    logger.info("Starting evaluation of pending checkpoints.")
    evaluated_checkpoints = process_pending_checkpoints(evaluated_checkpoints, evaluated_epochs)
    
    logger.info(f"Initial checkpoint evaluation complete. Monitoring for new checkpoints in {CHECKPOINT_DIR}")
    
    while True:
        try:
            # Refresh the list of evaluated epochs in case it was updated externally
            evaluated_epochs = load_already_evaluated_epochs(METRICS_FILE)
            
            # Process any new checkpoints
            evaluated_checkpoints = process_pending_checkpoints(evaluated_checkpoints, evaluated_epochs)
            
            logger.info(f"Sleeping for {SLEEP_TIME} seconds before rechecking...")
            time.sleep(SLEEP_TIME)
        except KeyboardInterrupt:
            logger.info("Evaluation stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.exception("Exception details:")
            time.sleep(60)  # Wait a bit before retrying

if __name__ == "__main__":
    main()
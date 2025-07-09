import sys
import os
import yaml
from easydict import EasyDict as edict
from data.dataset_scene import Dataset
from litdata import optimize
import time
import math
import shutil
import json
from tqdm import tqdm

project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root)

def load_config_for_prep(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def get_num_workers():
    """Detects the number of available CPU cores for the job."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()

if __name__ == '__main__':
    # --- Configuration ---
    CONFIG_FILE = "./configs/LVSM_object_decoder_only.yaml"
    MASTER_LIST_FILE = "./datasets/objaverseplus_processed/full_list_processed.txt"
    OUTPUT_CHUNK_DIR = "./datasets/objaverseplus_chunked"
    
    
    BATCH_SIZE = 100
    
    RAM_DISK_PATH = "/dev/shm"
   
    SOURCE_DATA_ROOT = os.path.dirname(os.path.dirname(MASTER_LIST_FILE))
    
   
    STATE_FILE = os.path.join(os.path.dirname(OUTPUT_CHUNK_DIR), "chunking_obja_progress.txt")

 
    config = load_config_for_prep(CONFIG_FILE)
    num_workers = get_num_workers()

    print("--- Resumable RAM-Disk Batch Chunking Script ---")
    
    if not os.path.exists(RAM_DISK_PATH):
        raise FileNotFoundError(f"RAM Disk not found at {RAM_DISK_PATH}. This script requires a Linux environment with /dev/shm.")

    with open(MASTER_LIST_FILE, 'r') as f:
        all_scene_paths = [line.strip() for line in f if line.strip()]
    
    total_scenes = len(all_scene_paths)
    total_batches = math.ceil(total_scenes / BATCH_SIZE)
    print(f"Found {total_scenes} total scenes. Splitting into {total_batches} batches of size {BATCH_SIZE}.")

    start_batch = 0
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try:
                start_batch = int(f.read().strip())
                print(f"Resuming from batch {start_batch + 1}/{total_batches}.")
            except (ValueError, IndexError):
                print("Could not read state file, starting from scratch.")
    
    for i in range(start_batch, total_batches):
        batch_num = i + 1
        print(f"\n--- Preparing Batch {batch_num}/{total_batches} ---")
        
        
        batch_ram_dir = os.path.join(RAM_DISK_PATH, f"lvsm_batch_{batch_num}")
        temp_list_file = os.path.join(batch_ram_dir, f"temp_batch_list.txt")
        
        try:
            os.makedirs(batch_ram_dir, exist_ok=True)
            
            
            start_index = i * BATCH_SIZE
            end_index = start_index + BATCH_SIZE
            batch_scene_paths_on_hdd = all_scene_paths[start_index:end_index]
            
            new_scene_paths_on_ram = []
            
            print(f"Copying {len(batch_scene_paths_on_hdd)} scenes to RAM Disk at {batch_ram_dir}...")
            for scene_path_hdd in tqdm(batch_scene_paths_on_hdd, desc="Copying to RAM"):
               
                scene_filename = os.path.basename(scene_path_hdd)
                scene_path_ram = os.path.join(batch_ram_dir, scene_filename)
                shutil.copy(scene_path_hdd, scene_path_ram)
                new_scene_paths_on_ram.append(scene_path_ram)

                
                with open(scene_path_hdd, 'r') as f:
                    scene_data = json.load(f)
                for frame in scene_data['frames']:
                    img_rel_path = frame['image_path']
                    img_src_path = os.path.join(SOURCE_DATA_ROOT, img_rel_path)
                    img_dest_path = os.path.join(batch_ram_dir, os.path.basename(img_rel_path))
                    
                    
                    os.makedirs(os.path.dirname(img_dest_path), exist_ok=True)
                    if not os.path.exists(img_dest_path): # Avoid re-copying if multiple scenes share images
                         shutil.copy(img_src_path, img_dest_path)

           
            with open(temp_list_file, 'w') as f:
                for path in new_scene_paths_on_ram:
                    f.write(f"{path}\n")

           
            print(f"Processing batch from RAM Disk with {num_workers} workers...")
            config.training.dataset_path = temp_list_file
            source_dataset = Dataset(config, load_all_views=True)
            
            mode = "append" if i > 0 or os.path.exists(os.path.join(OUTPUT_CHUNK_DIR, "index.json")) else "overwrite"
            print(f"LitData mode for this batch: '{mode}'")

            optimize(
                fn=source_dataset.__getitem__,
                inputs=list(range(len(source_dataset))),
                output_dir=OUTPUT_CHUNK_DIR,
                chunk_size=(256 * 1024 * 1024),
                num_workers=num_workers,
                mode=mode
            )
            
            
            with open(STATE_FILE, 'w') as f:
                f.write(str(batch_num))
            print(f"--- Successfully completed Batch {batch_num}/{total_batches}. Progress saved. ---")

        finally:
          
            if os.path.exists(batch_ram_dir):
                print(f"Cleaning up RAM Disk directory: {batch_ram_dir}")
                shutil.rmtree(batch_ram_dir)

    print("\n\n--- All batches processed. Chunking complete! ---")
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
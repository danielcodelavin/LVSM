import os
import json
import shutil
import time
import signal
import subprocess
from urllib.request import urlretrieve
from tqdm import tqdm

def download_with_timeout(url, filename, timeout=60):
    """Download a single file with timeout"""
    def handler(signum, frame):
        raise TimeoutError(f"Download timed out after {timeout} seconds")
    
    # Set timeout
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        urlretrieve(url, filename)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False
    finally:
        signal.alarm(0)  # Disable alarm

def setup_custom_path(target_path):
    """Set up a custom path for Objaverse downloads using symbolic links"""
    # Expand target path to absolute path
    target_path = os.path.abspath(os.path.expanduser(target_path))
    
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    print(f"Storage location: {target_path}")
    
    # Default objaverse path
    default_path = os.path.expanduser("~/.objaverse")
    
    # If the default directory already exists but is not a symlink to our target
    if os.path.exists(default_path):
        if os.path.islink(default_path) and os.readlink(default_path) == target_path:
            print(f"Symlink from {default_path} to {target_path} already exists")
            return
        
        # Backup the existing directory
        backup_path = default_path + "_backup_" + str(int(time.time()))
        print(f"Moving existing {default_path} to {backup_path}")
        shutil.move(default_path, backup_path)
    
    # Create a symbolic link from the default path to the custom path
    print(f"Creating symlink from {default_path} to {target_path}")
    os.symlink(target_path, default_path)

def robust_download_objaverse(target_path, batch_size=1000):
    """Download Objaverse one object at a time to avoid multiprocessing issues"""
    import objaverse
    
    # Setup paths
    setup_custom_path(target_path)
    
    # Create directory for progress tracking
    os.makedirs(os.path.join(target_path, "progress"), exist_ok=True)
    progress_file = os.path.join(target_path, "progress", "download_progress.json")
    failed_file = os.path.join(target_path, "progress", "failed_downloads.json")
    
    # Get all UIDs
    print("Loading all UIDs from Objaverse...")
    all_uids = objaverse.load_uids()
    total_uids = len(all_uids)
    print(f"Found {total_uids} 3D models in Objaverse")
    
    # Check for existing progress
    downloaded_uids = set()
    failed_uids = set()
    
    if os.path.exists(progress_file) and os.path.getsize(progress_file) > 0:
        try:
            with open(progress_file, 'r') as f:
                downloaded_uids = set(json.load(f))
            print(f"Resuming download: {len(downloaded_uids)}/{total_uids} already downloaded")
        except json.JSONDecodeError:
            print("Warning: Progress file was corrupted. Starting with empty progress.")
            
    if os.path.exists(failed_file) and os.path.getsize(failed_file) > 0:
        try:
            with open(failed_file, 'r') as f:
                failed_uids = set(json.load(f))
            print(f"Skipping {len(failed_uids)} previously failed downloads")
        except json.JSONDecodeError:
            print("Warning: Failed file was corrupted. Starting with empty failed list.")
    
    # Filter out already downloaded and failed UIDs
    remaining_uids = [uid for uid in all_uids if uid not in downloaded_uids and uid not in failed_uids]
    print(f"Remaining downloads: {len(remaining_uids)}")
    
    # Load annotations once to cache them
    print("Loading annotations (this will cache them for future use)...")
    objaverse.load_annotations()
    
    # Make sure the glbs directory structure exists
    glbs_dir = os.path.join(target_path, "hf-objaverse-v1", "glbs")
    os.makedirs(glbs_dir, exist_ok=True)
    
    # Download in batches for progress tracking
    total_batches = (len(remaining_uids) + batch_size - 1) // batch_size
    
    for i in range(0, len(remaining_uids), batch_size):
        batch_uids = remaining_uids[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"[Batch {batch_num}/{total_batches}] Downloading {len(batch_uids)} models...")
        
        # Process each UID individually
        for uid in tqdm(batch_uids, desc=f"Batch {batch_num}"):
            # Get the folder for this UID (using the same pattern as objaverse)
            uid_folder = f"{uid[0:3]}-{uid[3:6]}"
            uid_path = os.path.join(glbs_dir, uid_folder)
            os.makedirs(uid_path, exist_ok=True)
            
            glb_path = os.path.join(uid_path, f"{uid}.glb")
            
            # Skip if already downloaded
            if os.path.exists(glb_path) and os.path.getsize(glb_path) > 0:
                downloaded_uids.add(uid)
                continue
            
            # Try to download just this single object
            try:
                # Use a simple subprocess with timeout instead of the package's multiprocessing
                # This is a separate process, so if it hangs it won't affect the main script
                cmd = [
                    "python", "-c", 
                    f"import objaverse; objaverse.load_objects(['{uid}'])"
                ]
                
                # Run with timeout (3 minutes per object)
                result = subprocess.run(cmd, timeout=180, capture_output=True)
                
                if result.returncode == 0 and os.path.exists(glb_path) and os.path.getsize(glb_path) > 0:
                    downloaded_uids.add(uid)
                else:
                    print(f"Failed to download {uid}: {result.stderr.decode('utf-8')}")
                    failed_uids.add(uid)
            
            except Exception as e:
                print(f"Error downloading {uid}: {e}")
                failed_uids.add(uid)
            
            # Save progress every 10 objects
            if len(downloaded_uids) % 10 == 0:
                with open(progress_file, 'w') as f:
                    json.dump(list(downloaded_uids), f)
                with open(failed_file, 'w') as f:
                    json.dump(list(failed_uids), f)
        
        # Save progress after each batch
        with open(progress_file, 'w') as f:
            json.dump(list(downloaded_uids), f)
        with open(failed_file, 'w') as f:
            json.dump(list(failed_uids), f)
        
        print(f"[Batch {batch_num}/{total_batches}] Complete! "
              f"Total progress: {len(downloaded_uids)}/{total_uids} "
              f"({100*len(downloaded_uids)/total_uids:.2f}%)")
    
    print("\nDownload complete!")
    print(f"Successfully downloaded {len(downloaded_uids)}/{total_uids} models")
    print(f"Failed to download {len(failed_uids)} models")
    print(f"Data stored in: {target_path}")


if __name__ == "__main__":
    path = "/home/stud/lavingal/storage/group/dataset_mirrors/01_incoming/objaverse_1_0"
    batch_size = 1000
    
    robust_download_objaverse(path, batch_size)


    ####################### IMPORTANT NOTE #######################
    #   The annotations to each models are in the objaverse package
    #   they can simply be accessed using: annotation = objaverse.load_annotations([uid])
    #    and further: model_info = annotation[uid]
    #   eg. print(f"Model name: {model_info['name']}")
    #
    #   API :  https://objaverse.allenai.org/docs/objaverse-1.0/
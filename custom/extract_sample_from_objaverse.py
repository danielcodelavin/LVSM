import os
import subprocess
from pathlib import Path
import tempfile

# HARDCODED PATHS - MODIFY THESE
ZIP_PATH = "/home/stud/lavingal/storage/group/dataset_mirrors/01_incoming/ObjaverseXL-curated/xl_renderings.zip"
UUID_LIST_PATH = "/home/stud/lavingal/storage/group/dataset_mirrors/01_incoming/ObjaverseXL-curated/objaverseXL_curated_uuid_list.txt"
OUTPUT_DIR = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/custom/objaverse_sample"
NUM_SAMPLES = 5  # Number of sample objects to extract

def extract_objaverse_samples():
    """
    Extract sample objects from Objaverse XL multi-part zip files using 7z
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    
    # Read UUIDs from the list file (only the first NUM_SAMPLES)
    sample_uuids = []
    with open(UUID_LIST_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES:
                break
            uuid = line.strip()
            if uuid:
                sample_uuids.append(uuid)
    
    print(f"Selected {len(sample_uuids)} UUIDs to extract:")
    for uuid in sample_uuids:
        print(f"  - {uuid}")
    
    # Create a temporary include list file for 7z
    include_list_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    try:
        # Write patterns to include only the specific UUIDs
        for uuid in sample_uuids:
            include_list_file.write(f"{uuid}/*\n")
        include_list_file.close()
        
        # First, just list the files that would be extracted
        print("\nListing files that will be extracted (this will not extract anything)...")
        list_cmd = ["7z", "l", ZIP_PATH, f"@{include_list_file.name}"]
        try:
            result = subprocess.run(list_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Error listing files: {result.stderr}")
                return
            
            # Count how many files would be extracted
            file_lines = [line for line in result.stdout.splitlines() if ".png" in line or ".json" in line or ".npy" in line]
            print(f"Found approximately {len(file_lines)} files to extract")
            
            # Confirm before extracting
            user_confirm = input("Proceed with extraction? (y/n): ")
            if user_confirm.lower() != 'y':
                print("Extraction cancelled by user.")
                return
            
            # Perform the actual extraction
            print(f"\nExtracting files to {OUTPUT_DIR}...")
            extract_cmd = ["7z", "x", ZIP_PATH, f"-o{OUTPUT_DIR}", f"@{include_list_file.name}"]
            extract_result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if extract_result.returncode != 0:
                print(f"Error during extraction: {extract_result.stderr}")
                return
            
            print("Extraction completed successfully!")
            
        except Exception as e:
            print(f"Error executing 7z command: {str(e)}")
            return
            
    finally:
        # Clean up temporary file
        try:
            os.unlink(include_list_file.name)
        except:
            pass
    
    # Print summary of extracted content
    print("\nExtracted content structure:")
    total_size = 0
    for uuid in sample_uuids:
        uuid_dir = Path(OUTPUT_DIR) / uuid
        if uuid_dir.exists():
            files = list(uuid_dir.glob('*'))
            size = sum(f.stat().st_size for f in files)
            total_size += size
            print(f"{uuid}: {len(files)} files, {size/1024/1024:.2f} MB")
            
            # Group files by extension
            extensions = {}
            for file in files:
                ext = file.suffix
                if ext not in extensions:
                    extensions[ext] = []
                extensions[ext].append(file.name)
            
            # Print file types
            for ext, files_list in extensions.items():
                print(f"  - {ext}: {len(files_list)} files")
                # Show example of each type
                print(f"    Example: {files_list[0]}")
        else:
            print(f"{uuid}: No files found")
    
    print(f"\nTotal extracted: {total_size/1024/1024:.2f} MB")

if __name__ == "__main__":
    print(f"Starting extraction from {ZIP_PATH} to {OUTPUT_DIR}")
    extract_objaverse_samples()
    print("\nExtraction complete!")
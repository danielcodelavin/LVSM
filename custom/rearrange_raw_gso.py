# reorganize_files.py
#
# This script copies only the 'meshes' and 'materials' subdirectories
# for each object, discarding all other extraneous files and folders.
#
# Run this from your standard terminal: python reorganize_files.py
#

import os
import shutil

# --- CONFIGURE YOUR PATHS HERE ---

# The main input directory with the original file structure.
SOURCE_DIRECTORY = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/raw_gso"

# A new, empty directory where the clean object folders will be created.
DESTINATION_DIRECTORY = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/reorganized_gso"

# --- SCRIPT LOGIC ---

def main():
    """Main execution function."""
    print("--- Starting Selective File Reorganization ---")

    if not os.path.isdir(SOURCE_DIRECTORY):
        print(f"FATAL: Source directory not found: {SOURCE_DIRECTORY}")
        return

    # Use exist_ok=True to prevent errors if the script is run multiple times
    os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)
    print(f"Ensuring destination directory exists: {DESTINATION_DIRECTORY}")

    try:
        object_folders = [f for f in os.listdir(SOURCE_DIRECTORY) if os.path.isdir(os.path.join(SOURCE_DIRECTORY, f))]
    except OSError as e:
        print(f"FATAL: Could not read source directory: {e}")
        return

    print(f"Found {len(object_folders)} objects to process.")
    success_count = 0

    for object_name in sorted(object_folders):
        print(f"\n--- Processing: {object_name} ---")

        source_base_path = os.path.join(SOURCE_DIRECTORY, object_name)
        dest_object_dir = os.path.join(DESTINATION_DIRECTORY, object_name)

        try:
            # Create the top-level destination folder for the object
            os.makedirs(dest_object_dir, exist_ok=True)

            # --- KEY CHANGE: Copy entire folders instead of individual files ---

            # 1. Define paths for the 'meshes' folder
            source_meshes_folder = os.path.join(source_base_path, 'meshes')
            dest_meshes_folder = os.path.join(dest_object_dir, 'meshes')

            # 2. Define paths for the 'materials' folder
            source_materials_folder = os.path.join(source_base_path, 'materials')
            dest_materials_folder = os.path.join(dest_object_dir, 'materials')
            
            # Copy the 'meshes' folder if it exists
            if os.path.isdir(source_meshes_folder):
                print(f"  Copying 'meshes' folder...")
                shutil.copytree(source_meshes_folder, dest_meshes_folder, dirs_exist_ok=True)
            else:
                print(f"  Warning: 'meshes' folder not found for {object_name}.")

            # Copy the 'materials' folder if it exists
            if os.path.isdir(source_materials_folder):
                print(f"  Copying 'materials' folder...")
                shutil.copytree(source_materials_folder, dest_materials_folder, dirs_exist_ok=True)
            else:
                print(f"  Warning: 'materials' folder not found for {object_name}.")

            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to process {object_name}: {e}")

    print("\n--- Reorganization Finished ---")
    print(f"Successfully processed {success_count} / {len(object_folders)} objects.")
    print(f"Clean files are now located in: {DESTINATION_DIRECTORY}")

if __name__ == "__main__":
    main()
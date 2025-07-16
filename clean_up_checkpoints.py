import os
import re

def clean_checkpoints(master_path):
    """
    Deletes .pt files from subdirectories if their iteration number is not a multiple of 10,000.

    The script expects a directory structure like:
    master_path/
    ├── subfolder_1/
    │   ├── another_folder_to_ignore/
    │   ├── iter_00001000.pt
    │   ├── iter_00010000.pt
    │   └── some_other_file.txt
    └── subfolder_2/
        ├── iter_00020000.pt
        └── ...

    The script will recursively search through all directories under the master_path.

    Args:
        master_path (str): The absolute path to the master folder.
    """
    print(f"Starting cleanup in master folder: {master_path}\n")

    # Regular expression to find files named like 'iter_...pt' and extract the number
    # This is more robust than splitting by '_' or '.'
    pattern = re.compile(r'^iter_(\d+)\.pt$')

    # Check if the master path exists
    if not os.path.isdir(master_path):
        print(f"Error: Master path '{master_path}' does not exist or is not a directory.")
        return

    # os.walk() recursively visits every directory and file in the tree
    for dirpath, _, filenames in os.walk(master_path):
     #   print(f"Scanning directory: {dirpath}")
        for filename in filenames:
            if filename.endswith('.pt'):
                tensor = filename.split('.')
                #print (tensor)
                number = tensor[0].split('_')
                #print (number)
                number = number[1]
               # print (number)
                
                # Check if the filename matches our 'iter_...pt' pattern
                if tensor[1] == 'pt':
                    # The first group in the match is the number part
                    #iteration_number_str = match.group(1)
                    
                    try:
                        # Convert the extracted number string to an integer
                        iteration_number = int(number)
                        
                        # Check if the number is NOT a multiple of 10,000
                        if iteration_number % 10000 != 0:
                            file_to_delete_path = os.path.join(dirpath, filename)
                            try:
                                # Delete the file
                                os.remove(file_to_delete_path)
                                print(f"  DELETED: {file_to_delete_path} (iteration: {iteration_number})")
                            except OSError as e:
                                print(f"  ERROR: Could not delete {file_to_delete_path}. Reason: {e}")
                                
                    except ValueError:
                        # This will catch cases where the extracted part is not a valid number
                        print(f"  SKIPPED: Could not parse number from filename '{filename}' in '{dirpath}'")

    print("\nCleanup complete.")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this path to the absolute path of your master folder.
    # Example for Windows: "C:\\Users\\YourUser\\Desktop\\my_models"
    # Example for Linux/macOS: "/home/user/my_models"
    MASTER_FOLDER_PATH = "/storage/slurm/lavingal/lavingal/LVSM/experiments/checkpoints"

    # --- SAFETY WARNING ---
    # This script will permanently delete files. 
    # It's highly recommended to back up your data before running it.
    # You can perform a "dry run" by commenting out the `os.remove()` line
    # to see which files *would* be deleted without actually deleting them.
    
    clean_checkpoints(MASTER_FOLDER_PATH)

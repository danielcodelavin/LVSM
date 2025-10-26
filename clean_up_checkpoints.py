import os
import re

def clean_checkpoints(master_path):
    print(f"Starting cleanup in master folder: {master_path}\n")

    pattern = re.compile(r'^iter_(\d+)\.pt$')

    if not os.path.isdir(master_path):
        print(f"Error: Master path '{master_path}' does not exist or is not a directory.")
        return

    for dirpath, _, filenames in os.walk(master_path):
        for filename in filenames:
            if filename.endswith('.pt'):
                tensor = filename.split('.')
                number = tensor[0].split('_')
                number = number[1]
                
                if tensor[1] == 'pt':
                    try:
                        iteration_number = int(number)
                        
                        if iteration_number % 10000 != 0:
                            file_to_delete_path = os.path.join(dirpath, filename)
                            try:
                                os.remove(file_to_delete_path)
                                print(f"  DELETED: {file_to_delete_path} (iteration: {iteration_number})")
                            except OSError as e:
                                print(f"  ERROR: Could not delete {file_to_delete_path}. Reason: {e}")
                                
                    except ValueError:
                        print(f"  SKIPPED: Could not parse number from filename '{filename}' in '{dirpath}'")

    print("\nCleanup complete.")

if __name__ == "__main__":
    MASTER_FOLDER_PATH = "/storage/slurm/lavingal/lavingal/LVSM/experiments/checkpoints"
    
    clean_checkpoints(MASTER_FOLDER_PATH)

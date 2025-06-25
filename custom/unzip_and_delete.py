import os 
import zipfile

def unzip_to_named_folder_and_delete(target_dir):
    """
    Unzips all zip files in a directory. For each zip, it creates a new folder
    named after the zip file and extracts the contents there. The original zip
    file is deleted afterwards.
    """
    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found at {target_dir}")
        return

    for file in os.listdir(target_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(target_dir, file)
            
            # Create a directory name from the zip file name (e.g., "my_archive.zip" -> "my_archive")
            output_dir_name = os.path.splitext(file)[0]
            output_dir_path = os.path.join(target_dir, output_dir_name)
            
            try:
                # Create the new directory
                os.makedirs(output_dir_path, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract contents into the newly created directory
                    zip_ref.extractall(output_dir_path)
                
                # Delete the original zip file
                os.remove(zip_path)
                print(f"Unzipped '{zip_path}' to '{output_dir_path}' and deleted original.")

            except zipfile.BadZipFile:
                print(f"Error: Bad zip file, skipping: {zip_path}")
            except Exception as e:
                print(f"An error occurred with {zip_path}: {e}")

    print("All zip files have been processed.")


if __name__ == "__main__":
    # The path to the directory containing your .zip files
    unzip_to_named_folder_and_delete('/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/raw_gso')
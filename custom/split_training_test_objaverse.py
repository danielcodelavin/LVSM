import os

def create_file_list(input_dir, output_file):
    """
    Scans a directory for files, gets their absolute paths, 
    and writes them to an output file, one path per line.

    Args:
        input_dir (str): The absolute path to the directory to scan.
        output_file (str): The absolute path to the file to save the list.
    """
    print(f"Starting to scan directory: {input_dir}")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        # List all entries in the directory
        files_in_dir = os.listdir(input_dir)
        
        # Filter for files only, just in case there are subdirectories
        json_files = [f for f in files_in_dir if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.json')]
        
        print(f"Found {len(json_files)} JSON files to process.")

        with open(output_file, 'w') as f_out:
            # Iterate through each file and write its absolute path
            for filename in json_files:
                # Construct the full, absolute path
                absolute_path = os.path.join(input_dir, filename)
                # Write the path to the output file, followed by a newline
                f_out.write(absolute_path + '\n')
                
        print(f"Successfully wrote all paths to: {output_file}")

    except FileNotFoundError:
        print(f"Error: The input directory was not found at {input_dir}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    
    input_path = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/GSO_rendered/metadata"
    output_path = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/GSO_rendered/full_list.txt"
    
    # Run the function
    create_file_list(input_path, output_path)

import os

def create_file_list(input_dir, output_file):

    print(f"Starting to scan directory: {input_dir}")
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    
      
    files_in_dir = os.listdir(input_dir)
    
    json_files = [f for f in files_in_dir if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files to process.")

    with open(output_file, 'w') as f_out:
        
        for filename in json_files:
            
            absolute_path = os.path.join(input_dir, filename)
            
            f_out.write(absolute_path + '\n')

    print(f"Finished writing to: {output_file}")

if __name__ == '__main__':
    
    input_path = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/GSO_rendered/metadata"
    output_path = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/GSO_rendered/full_list.txt"
    
    # Run the function
    create_file_list(input_path, output_path)

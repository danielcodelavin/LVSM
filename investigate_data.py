import os
import torch
import litdata as ld
from rich import print
from rich.panel import Panel

def analyze_dataset_sample(dataset_path: str):
    """
    Loads the first sample from a LitData dataset and inspects its properties
    to verify the hypothesis of data type promotion causing size increase.
    """
    print(Panel(f"Analyzing dataset at: [bold cyan]{os.path.abspath(dataset_path)}[/bold cyan]", 
                title="[bold green]Dataset Inspector[/bold green]", expand=False))

    if not os.path.exists(dataset_path):
        print(f"[bold red]Error:[/bold red] Dataset path does not exist.")
        return

    # 1. Initialize the LitData StreamingDataset
    try:
        dataset = ld.StreamingDataset(input_dir=dataset_path, shuffle=False)
        if len(dataset) == 0:
            print("[bold red]Error:[/bold red] The dataset is empty or could not be loaded.")
            return
        
        # 2. Get the first sample from the dataset
        print("\nFetching the first data sample...")
        sample = next(iter(dataset))

    except Exception as e:
        print(f"[bold red]An error occurred while loading the dataset:[/bold red] {e}")
        return

    # 3. Inspect the sample, focusing on the 'image' tensor
    if 'image' not in sample or not isinstance(sample['image'], torch.Tensor):
        print("[bold yellow]Warning:[/bold yellow] Could not find an 'image' tensor in the sample.")
        print(f"Available keys: {list(sample.keys())}")
        return

    image_tensor = sample['image']
    
    # --- Print Analysis Results ---
    print("\n[--- [bold]Image Tensor Analysis[/bold] ---]")
    print(f"Shape: [yellow]{list(image_tensor.shape)}[/yellow]")
    print(f"Data Type (dtype): [bold magenta]{image_tensor.dtype}[/bold magenta]")

    # 4. Verify the hypothesis
    print("\n[--- [bold]Hypothesis Verification[/bold] ---]")
    
    if image_tensor.dtype == torch.float32:
        element_size_bytes = image_tensor.element_size()
        total_elements = image_tensor.nelement()
        
        # Calculate the size of the tensor in memory
        size_in_memory_mb = image_tensor.nbytes / (1024 * 1024)
        
        # Calculate what the size would be if it were uint8
        equivalent_uint8_size_mb = total_elements * 1 / (1024 * 1024)

        print(f"[bold green]Hypothesis CONFIRMED.[/bold green]")
        print(f"The data type is [bold magenta]torch.float32[/bold magenta], which uses [bold]{element_size_bytes} bytes[/bold] per value.")
        print(f"Total size of this tensor in memory: [bold cyan]{size_in_memory_mb:.2f} MB[/bold cyan]")
        print(f"Equivalent size if stored as uint8 (1 byte/value): [cyan]{equivalent_uint8_size_mb:.2f} MB[/cyan]")
        print("-" * 30)
        print(f"Size Ratio (float32 / uint8): [bold yellow]{size_in_memory_mb / equivalent_uint8_size_mb:.1f}x[/bold yellow]")

    else:
       
        print(f"The data type is [bold magenta]{image_tensor.dtype}[/bold magenta], not torch.float32.")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this path to point to your chunked dataset directory
    DATASET_PATH = "/storage/user/lavingal/objaverseplus_chunked"
    
    analyze_dataset_sample(DATASET_PATH)
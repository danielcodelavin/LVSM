import litdata as ld
import torch
import torch.distributed as dist
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_litdata.py /path/to/your/dataset")
        print("Or:    torchrun --nproc_per_node=2 debug_litdata.py /path/to/your/dataset")
        sys.exit(1)

    dataset_path = sys.argv[1]
    is_ddp = 'WORLD_SIZE' in os.environ

    rank = 0
    if is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()

    if rank == 0:
        print("--- LitData Diagnostic Test ---")
        print(f"LitData Version: {ld.__version__}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Running in Distributed Mode: {is_ddp}")
        print(f"Dataset Path: {os.path.abspath(dataset_path)}")
        print("---------------------------------")

    try:
        # Initialize the dataset
        dataset = ld.StreamingDataset(input_dir=dataset_path)
        
        # Check the length
        dataset_len = len(dataset)
        
        print(f"[Rank {rank}] ==> Dataset initialized successfully. Length: {dataset_len}")

        if dataset_len == 0:
            print(f"[Rank {rank}] ==> !!! FAILURE: Dataset length is 0. !!!")

    except Exception as e:
        print(f"[Rank {rank}] ==> !!! CRITICAL FAILURE: An exception occurred during dataset initialization. !!!")
        import traceback
        traceback.print_exc()

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
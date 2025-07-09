#!/bin/bash
#SBATCH --job-name="ramload chunked lvsm"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=3-12:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 train.py --config configs/LVSM_scene_decoder_only.yaml

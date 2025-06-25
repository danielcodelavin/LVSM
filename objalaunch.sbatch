#!/bin/bash
#SBATCH --job-name="4 Grad Objabase LVSM Extrap"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1,VRAM:96G
#SBATCH --time=2-12:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
torchrun --nproc_per_node 1 --nnodes 1 --rdzv-backend=c10d --rdzv-endpoint=localhost:12345 train.py --config configs/LVSM_object_decoder_only.yaml
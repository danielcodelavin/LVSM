#!/bin/bash
#SBATCH --job-name="INFER FINAL"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-4:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
torchrun --nproc_per_node 1 --nnodes 1  inf_dif.py --config configs/diffusiontest.yaml
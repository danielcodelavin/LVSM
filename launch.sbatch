#!/bin/bash
#SBATCH --job-name="model4_vggtperceptual"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out

source /home/stud/lavingal/miniforge3/etc/profile.d/conda.sh

# Activate your specific environment using mamba
mamba activate trainLVSM

echo "========================================="
echo "SLURM JOB: Running on host $(hostname)"
echo "SLURM JOB: Python path: $(which python)"
echo "========================================="


torchrun --nproc_per_node 1 --nnodes 1 --rdzv_id 109 --rdzv_backend c10d  train.py --config configs/model_4.yaml

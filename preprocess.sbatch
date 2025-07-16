#!/bin/bash
#SBATCH --job-name="reprocess re10k"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:0
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
python process_data.py
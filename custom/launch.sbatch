#!/bin/bash
#SBATCH --job-name="DownloadObjaverse"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:0
#SBATCH --time=12-00:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
python download_objaverse10.py
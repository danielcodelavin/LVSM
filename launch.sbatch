#!/bin/bash
#SBATCH --job-name="LVSMCheckEval"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1,VRAM:96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
python checkpoint_eval.py
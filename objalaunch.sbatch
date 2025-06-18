#!/bin/bash
#SBATCH --job-name="Objabase LVSM Extrap"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1,VRAM:96G
#SBATCH --time=7-12:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 train.py --config configs/LVSM_scene_decoder_only.yaml training.num_input_views=4 training.num_target_views=8 training.num_views=12 training.random_sample_views=True training.checkpoint_dir="./experiments/checkpoints/LVSM_objaverse_extrapol_norandom" training.dataset_path="./datasets/objaverseplus/full_list.txt" training.wandb_exp_name="OBJA_EXTRAP_NORANDOM_LVSM"
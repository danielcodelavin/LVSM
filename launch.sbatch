#!/bin/bash
#SBATCH --job-name="Rand_LVSM"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1,VRAM:96G
#SBATCH --time=4-12:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
torchrun --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:18635 inference.py --config configs/LVSM_scene_decoder_only.yaml training.dataset_path="/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/re10k/test/full_list.txt" training.batch_size_per_gpu=8 training.target_has_input=false training.num_views=5 training.square_crop=true training.num_input_views=2 training.num_target_views=3 training.uniform_crop_size=false training.random_sample_views=false training.variable_amount_of_views=false inference.if_inference=true inference.compute_metrics=true inference.render_video=true inference_out_dir=./experiments/evaluation/LVSM_RE10k_RANDOM training.checkpoint_dir="/home/stud/lavingal/storage/slurm/lavingal/LVSM/experiments/checkpoints/LVSM_random_re10k_baseline"
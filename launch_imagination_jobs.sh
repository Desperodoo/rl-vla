#!/bin/bash
# Launch 3 imagination evaluation jobs

source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3

cd /home/wjz/rl-vla

# Job 1: GPU 3, step-600
CUDA_VISIBLE_DEVICES=3 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-600.pt \
  --num_trajs 20 \
  --output_dir data/vlaw/synthetic/wm_eval_step600 \
  --gpu_id 0 --visualize --vis_count 5 \
  > logs/vlaw/imagination_eval_step600.log 2>&1 &
echo "JOB1_PID=$!"

# Job 2: GPU 8, step-1000
CUDA_VISIBLE_DEVICES=8 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1000.pt \
  --num_trajs 20 \
  --output_dir data/vlaw/synthetic/wm_eval_step1000 \
  --gpu_id 0 --visualize --vis_count 5 \
  > logs/vlaw/imagination_eval_step1000.log 2>&1 &
echo "JOB2_PID=$!"

# Job 3: GPU 9, step-1400
CUDA_VISIBLE_DEVICES=9 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1400.pt \
  --num_trajs 20 \
  --output_dir data/vlaw/synthetic/wm_eval_step1400 \
  --gpu_id 0 --visualize --vis_count 5 \
  > logs/vlaw/imagination_eval_step1400.log 2>&1 &
echo "JOB3_PID=$!"

echo "All 3 jobs launched."

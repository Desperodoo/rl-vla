#!/bin/bash
# Launch imagination evaluations for missing WM checkpoints (600, 1000, 1400)
# Uses GPUs 3, 8, 9 respectively
# Each run generates 20 trajectories + VAE-decoded viz

set -e

cd /home/wjz/rl-vla
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3

echo "=== Launching imagination evaluations ==="
echo "Time: $(date)"

# Job 1: GPU 3, checkpoint-600
CUDA_VISIBLE_DEVICES=3 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-600.pt \
  --num_trajs 20 --output_dir data/vlaw/synthetic/wm_eval_step600 \
  --gpu_id 0 --visualize --vis_count 5 \
  > logs/vlaw/imagination_eval_step600.log 2>&1 &
PID1=$!
echo "Job 1 (step 600, GPU 3): PID=$PID1"

# Job 2: GPU 8, checkpoint-1000
CUDA_VISIBLE_DEVICES=8 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1000.pt \
  --num_trajs 20 --output_dir data/vlaw/synthetic/wm_eval_step1000 \
  --gpu_id 0 --visualize --vis_count 5 \
  > logs/vlaw/imagination_eval_step1000.log 2>&1 &
PID2=$!
echo "Job 2 (step 1000, GPU 8): PID=$PID2"

# Job 3: GPU 9, checkpoint-1400
CUDA_VISIBLE_DEVICES=9 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1400.pt \
  --num_trajs 20 --output_dir data/vlaw/synthetic/wm_eval_step1400 \
  --gpu_id 0 --visualize --vis_count 5 \
  > logs/vlaw/imagination_eval_step1400.log 2>&1 &
PID3=$!
echo "Job 3 (step 1400, GPU 9): PID=$PID3"

echo ""
echo "=== All 3 jobs launched ==="
echo "PIDs: $PID1 $PID2 $PID3"
echo ""
echo "Monitor with:"
echo "  tail -f logs/vlaw/imagination_eval_step600.log"
echo "  tail -f logs/vlaw/imagination_eval_step1000.log"
echo "  tail -f logs/vlaw/imagination_eval_step1400.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"

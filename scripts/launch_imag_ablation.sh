#!/bin/bash
# Imagination parameter ablation experiments
# Baseline: ckpt-1200, num_interact=4 (short horizon), num_inference_steps=25, guidance_scale=2.0
# All experiments use 20 trajectories with visualization
#
# E1 (GPU 3): num_inference_steps=50  (2x denoising steps)
# E2 (GPU 8): guidance_scale=5.0     (higher CFG for action fidelity)
# E3 (GPU 9): guidance_scale=1.0     (no CFG, control group)

set -e

cd /home/wjz/rl-vla
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3

WM_CKPT="checkpoints/vlaw/world_model/iter1_v3_ext/checkpoint-1200.pt"
COMMON_ARGS="--num_trajs 20 --num_interact 4 --gpu_id 0 --visualize --vis_count 10"

echo "=== Imagination Ablation Experiments ==="
echo "Baseline: ckpt-1200, num_interact=4, steps=25, cfg=2.0"
echo "Time: $(date)"
echo ""

# E1: num_inference_steps=50
echo "[E1] num_inference_steps=50 (GPU 3)"
CUDA_VISIBLE_DEVICES=3 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt "$WM_CKPT" \
  --output_dir data/vlaw/synthetic/ablation_steps50 \
  --num_inference_steps 50 \
  $COMMON_ARGS \
  > logs/vlaw/ablation_steps50.log 2>&1 &
PID1=$!
echo "  PID=$PID1"

# E2: guidance_scale=5.0
echo "[E2] guidance_scale=5.0 (GPU 8)"
CUDA_VISIBLE_DEVICES=8 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt "$WM_CKPT" \
  --output_dir data/vlaw/synthetic/ablation_cfg5 \
  --guidance_scale 5.0 \
  $COMMON_ARGS \
  > logs/vlaw/ablation_cfg5.log 2>&1 &
PID2=$!
echo "  PID=$PID2"

# E3: guidance_scale=1.0 (no CFG)
echo "[E3] guidance_scale=1.0 (GPU 9)"
CUDA_VISIBLE_DEVICES=9 nohup python rlft/vlaw/scripts/run_imagination.py \
  --wm_ckpt "$WM_CKPT" \
  --output_dir data/vlaw/synthetic/ablation_cfg1 \
  --guidance_scale 1.0 \
  $COMMON_ARGS \
  > logs/vlaw/ablation_cfg1.log 2>&1 &
PID3=$!
echo "  PID=$PID3"

echo ""
echo "=== All 3 ablation jobs launched ==="
echo "PIDs: E1=$PID1 E2=$PID2 E3=$PID3"
echo ""
echo "Monitor:"
echo "  tail -f logs/vlaw/ablation_steps50.log"
echo "  tail -f logs/vlaw/ablation_cfg5.log"
echo "  tail -f logs/vlaw/ablation_cfg1.log"
echo ""
echo "Check GPU:"
echo "  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"

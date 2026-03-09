#!/bin/bash
# Wrapper to launch imagination evaluations
# Created by Claude Code session on 2026-03-08
# Run with: bash scripts/launch_imagination_eval_wrapper.sh

CONDA_BASE="$HOME/miniconda3"
PYTHON="$CONDA_BASE/envs/rlft_ms3/bin/python"
WD="/home/wjz/rl-vla"
CKPT_BASE="$WD/checkpoints/vlaw/world_model/iter1_v3_ext"
OUT_BASE="$WD/data/vlaw/synthetic"
LOG_BASE="$WD/logs/vlaw"
SCRIPT="$WD/rlft/vlaw/scripts/run_imagination.py"

export PYTHONPATH="$WD:$WD/ctrl_world"

cd "$WD"

# Job 1: GPU 3, step 600
CUDA_VISIBLE_DEVICES=3 nohup "$PYTHON" "$SCRIPT" \
  --wm_ckpt "$CKPT_BASE/checkpoint-600.pt" \
  --num_trajs 20 --output_dir "$OUT_BASE/wm_eval_step600" \
  --gpu_id 0 --visualize --vis_count 5 \
  > "$LOG_BASE/imagination_eval_step600.log" 2>&1 &
echo "Job 1 (step 600, GPU 3): PID=$!"

# Job 2: GPU 8, step 1000
CUDA_VISIBLE_DEVICES=8 nohup "$PYTHON" "$SCRIPT" \
  --wm_ckpt "$CKPT_BASE/checkpoint-1000.pt" \
  --num_trajs 20 --output_dir "$OUT_BASE/wm_eval_step1000" \
  --gpu_id 0 --visualize --vis_count 5 \
  > "$LOG_BASE/imagination_eval_step1000.log" 2>&1 &
echo "Job 2 (step 1000, GPU 8): PID=$!"

# Job 3: GPU 9, step 1400
CUDA_VISIBLE_DEVICES=9 nohup "$PYTHON" "$SCRIPT" \
  --wm_ckpt "$CKPT_BASE/checkpoint-1400.pt" \
  --num_trajs 20 --output_dir "$OUT_BASE/wm_eval_step1400" \
  --gpu_id 0 --visualize --vis_count 5 \
  > "$LOG_BASE/imagination_eval_step1400.log" 2>&1 &
echo "Job 3 (step 1400, GPU 9): PID=$!"

echo ""
echo "All jobs launched. Check logs:"
echo "  tail -f $LOG_BASE/imagination_eval_step600.log"
echo "  tail -f $LOG_BASE/imagination_eval_step1000.log"
echo "  tail -f $LOG_BASE/imagination_eval_step1400.log"

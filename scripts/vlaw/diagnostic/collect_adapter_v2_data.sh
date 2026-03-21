#!/bin/bash
# Phase 1.1: Collect adapter training data with frame_skip=4 (aligned with WM)
# 4 types × 400 trajectories, staggered GPU launch (30s intervals)
#
# Usage:
#   bash scripts/vlaw/diagnostic/collect_adapter_v2_data.sh
#
set -euo pipefail

# Base conda activation
eval "$(/home/wjz/miniconda3/bin/conda shell.bash hook)"
conda activate rlft_ms3

# FIX: correct checkpoint path (double-nested fair_comparison/)
CKPT="runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt"
COMMON_ARGS="--frame_skip 4 --max_episode_steps 200 --num_episodes 400 --num_envs 32 --checkpoint_path $CKPT"

echo "[Phase 1.1] Collecting adapter v2 data (frame_skip=4, max_steps=200)"
echo "  Config: $COMMON_ARGS"
echo ""

# --- Type 1: Random (GPU 2) ---
echo "[$(date +%H:%M:%S)] Starting adapter_v2_random (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python scripts/collect_acp_data.py $COMMON_ARGS \
    --noise_mode random \
    --output_dir data/vlaw/rollouts/adapter_v2_random \
    --gpu_id 0 \
    > logs/vlaw/adapter_v2_random.log 2>&1 &
PID_RANDOM=$!
echo "  PID=$PID_RANDOM"

sleep 30  # Stagger to avoid NVIDIA RM mutex contention

# --- Type 2: Clean pretrained (GPU 3) ---
echo "[$(date +%H:%M:%S)] Starting adapter_v2_clean (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 python scripts/collect_acp_data.py $COMMON_ARGS \
    --noise_mode none \
    --output_dir data/vlaw/rollouts/adapter_v2_clean \
    --gpu_id 0 \
    > logs/vlaw/adapter_v2_clean.log 2>&1 &
PID_CLEAN=$!
echo "  PID=$PID_CLEAN"

sleep 30

# --- Type 3: Teleop OU noise (GPU 4) ---
echo "[$(date +%H:%M:%S)] Starting adapter_v2_teleop (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 python scripts/collect_acp_data.py $COMMON_ARGS \
    --noise_mode teleop \
    --ou_sigma 0.07 --pause_prob 0.04 \
    --output_dir data/vlaw/rollouts/adapter_v2_teleop \
    --gpu_id 0 \
    > logs/vlaw/adapter_v2_teleop.log 2>&1 &
PID_TELEOP=$!
echo "  PID=$PID_TELEOP"

sleep 30

# --- Type 4: Gaussian noise (GPU 5) ---
echo "[$(date +%H:%M:%S)] Starting adapter_v2_gaussian (GPU 5)..."
CUDA_VISIBLE_DEVICES=5 python scripts/collect_acp_data.py $COMMON_ARGS \
    --noise_mode rl_explore \
    --explore_sigma 0.25 \
    --output_dir data/vlaw/rollouts/adapter_v2_gaussian \
    --gpu_id 0 \
    > logs/vlaw/adapter_v2_gaussian.log 2>&1 &
PID_GAUSSIAN=$!
echo "  PID=$PID_GAUSSIAN"

echo ""
echo "[Phase 1.1] All 4 collection jobs launched."
echo "  PIDs: random=$PID_RANDOM, clean=$PID_CLEAN, teleop=$PID_TELEOP, gaussian=$PID_GAUSSIAN"
echo "  Logs: logs/vlaw/adapter_v2_*.log"
echo ""
echo "Waiting for all jobs to complete..."

# Wait for all processes
FAILED=0
for PID in $PID_RANDOM $PID_CLEAN $PID_TELEOP $PID_GAUSSIAN; do
    if ! wait $PID; then
        echo "[ERROR] PID=$PID failed!"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "[Phase 1.1] ✅ All 4 data collections completed successfully!"
    echo ""
    # Quick summary
    for DIR in adapter_v2_random adapter_v2_clean adapter_v2_teleop adapter_v2_gaussian; do
        FILES=$(ls -1 data/vlaw/rollouts/$DIR/*.h5 2>/dev/null | wc -l)
        SIZE=$(du -sh data/vlaw/rollouts/$DIR/ 2>/dev/null | cut -f1)
        echo "  $DIR: $FILES files, $SIZE"
    done
else
    echo ""
    echo "[Phase 1.1] ❌ $FAILED collection(s) failed. Check logs."
fi

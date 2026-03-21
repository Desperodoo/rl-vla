#!/bin/bash
# Phase 1.3: PSNR comparison V1 vs V1.1 vs Tiled vs GT
# Uses test_adapter_psnr.py with different adapter checkpoints
#
# Usage:
#   bash scripts/vlaw/diagnostic/compare_v1_v1.1.sh
#
set -euo pipefail

eval "$(/home/wjz/miniconda3/bin/conda shell.bash hook)"
conda activate ctrl_world  # Need WM for PSNR evaluation

V1_CKPT="checkpoints/vlaw/dynamics_adapter/best.pt"
V11_CKPT="checkpoints/vlaw/dynamics_adapter_v1.1/best.pt"
WM_CKPT="checkpoints/vlaw/world_model/iter1_v5/checkpoint-800.pt"
DATA_H5="data/vlaw/encoded/train_v5/LiftPegUpright-v1/*.h5"
RESULTS_DIR="results/vlaw/adapter_comparison"

mkdir -p "$RESULTS_DIR"

echo "[Phase 1.3] Comparing Adapter V1 vs V1.1 PSNR..."
echo ""

# Test 1: V1 baseline
echo "=== Testing V1 (baseline) ==="
CUDA_VISIBLE_DEVICES=8 python scripts/vlaw/diagnostic/test_adapter_psnr.py \
    --data_h5 $DATA_H5 \
    --adapter_ckpt "$V1_CKPT" \
    --wm_checkpoint "$WM_CKPT" \
    --num_samples 30 \
    --gpu_id 0 2>&1 | tee "$RESULTS_DIR/psnr_v1.log"

echo ""

# Test 2: V1.1 (frame_skip aligned)
echo "=== Testing V1.1 (fs=4 aligned) ==="
CUDA_VISIBLE_DEVICES=8 python scripts/vlaw/diagnostic/test_adapter_psnr.py \
    --data_h5 $DATA_H5 \
    --adapter_ckpt "$V11_CKPT" \
    --wm_checkpoint "$WM_CKPT" \
    --num_samples 30 \
    --gpu_id 0 2>&1 | tee "$RESULTS_DIR/psnr_v1.1.log"

echo ""
echo "[Phase 1.3] Comparison complete!"
echo "  Results: $RESULTS_DIR/psnr_v1.log, psnr_v1.1.log"
echo ""
echo "Check logs for detailed per-step breakdown."

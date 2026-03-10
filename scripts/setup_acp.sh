#!/bin/bash
# Download Gemma + run demo_prep in sequence
cd /home/wjz/rl-vla

echo "=== Step 1: Download Gemma ==="
conda run -n rlft_ms3 python scripts/download_gemma.py
echo "=== Gemma download exit code: $? ==="

echo "=== Step 2: Demo prep ==="
conda run -n rlft_ms3 python rlft/vlaw/data/demo_prep.py \
    --env_id LiftPegUpright-v1 \
    --num_trajs 25 \
    --target_hw 128 \
    --output_dir data/vlaw/rollouts/mixed
echo "=== Demo prep exit code: $? ==="

echo "=== ALL STEPS DONE ==="

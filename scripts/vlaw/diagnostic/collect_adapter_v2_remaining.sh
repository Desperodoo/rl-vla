#!/bin/bash
# Re-run 3 failed adapter v2 data collections (clean, teleop, gaussian)
# Sequential to avoid NVIDIA RM mutex lock issues
set -euo pipefail

eval "$(/home/wjz/miniconda3/bin/conda shell.bash hook)"
conda activate rlft_ms3

CKPT="runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt"
COMMON="--frame_skip 4 --max_episode_steps 200 --num_episodes 400 --num_envs 32 --checkpoint_path $CKPT"

mkdir -p logs/vlaw

echo "[$(date)] Starting clean (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python scripts/collect_acp_data.py $COMMON \
    --noise_mode none \
    --output_dir data/vlaw/rollouts/adapter_v2_clean \
    --gpu_id 0 2>&1 | tee logs/vlaw/adapter_v2_clean.log

echo "[$(date)] Starting teleop (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 python scripts/collect_acp_data.py $COMMON \
    --noise_mode teleop --ou_sigma 0.07 --pause_prob 0.04 \
    --output_dir data/vlaw/rollouts/adapter_v2_teleop \
    --gpu_id 0 2>&1 | tee logs/vlaw/adapter_v2_teleop.log

echo "[$(date)] Starting gaussian (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 python scripts/collect_acp_data.py $COMMON \
    --noise_mode rl_explore --explore_sigma 0.25 \
    --output_dir data/vlaw/rollouts/adapter_v2_gaussian \
    --gpu_id 0 2>&1 | tee logs/vlaw/adapter_v2_gaussian.log

echo ""
echo "[$(date)] All 3 collections done. Summary:"
for DIR in adapter_v2_random adapter_v2_clean adapter_v2_teleop adapter_v2_gaussian; do
    if [ -d "data/vlaw/rollouts/$DIR" ]; then
        SIZE=$(du -sh data/vlaw/rollouts/$DIR/ 2>/dev/null | cut -f1)
        echo "  $DIR: $SIZE"
    fi
done

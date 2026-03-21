#!/bin/bash
# Phase 1.2: Retrain Adapter V1.1 with all frame_skip=4 data
# Data sources:
#   - mixed (1200 traj, fs=4) - original training data
#   - adapter_v2_random (400 traj, fs=4)
#   - adapter_v2_clean (400 traj, fs=4)
#   - adapter_v2_teleop (400 traj, fs=4)
#   - adapter_v2_gaussian (400 traj, fs=4)
# Total: ~2800 trajectories
#
# Usage:
#   bash scripts/vlaw/diagnostic/train_adapter_v1.1.sh
#
set -euo pipefail

eval "$(/home/wjz/miniconda3/bin/conda shell.bash hook)"
conda activate rlft_ms3

# Verify all data directories exist
for DIR in \
    "data/vlaw/rollouts/mixed/LiftPegUpright-v1" \
    "data/vlaw/rollouts/adapter_v2_random" \
    "data/vlaw/rollouts/adapter_v2_clean" \
    "data/vlaw/rollouts/adapter_v2_teleop" \
    "data/vlaw/rollouts/adapter_v2_gaussian"; do
    if [ ! -d "$DIR" ] || [ -z "$(ls -A $DIR/*.h5 2>/dev/null)" ]; then
        echo "[ERROR] Missing or empty data dir: $DIR"
        exit 1
    fi
done

echo "[Phase 1.2] Training Adapter V1.1 with all fs=4 data..."
echo ""

CUDA_VISIBLE_DEVICES=8 python -c "
from rlft.vlaw.world_model.dynamics_adapter import DynamicsAdapterConfig, DynamicsAdapterTrainer

cfg = DynamicsAdapterConfig(
    mode='train',
    hdf5_dirs=(
        'data/vlaw/rollouts/adapter_v2_random',
        'data/vlaw/rollouts/adapter_v2_clean',
        'data/vlaw/rollouts/adapter_v2_teleop',
        'data/vlaw/rollouts/adapter_v2_gaussian',
    ),
    checkpoint_dir='checkpoints/vlaw/dynamics_adapter_v1.1',
    epochs=100,
    batch_size=512,
    lr=3e-4,
    gpu_id=0,
    model_version='v1',  # V1 architecture (3-layer MLP)
    early_stop_patience=10,
)

trainer = DynamicsAdapterTrainer(cfg)
# Main dir as first arg, extra dirs from config
trainer.train('data/vlaw/rollouts/mixed/LiftPegUpright-v1')
print()
print('[Phase 1.2] Training complete!')
print(f'  Checkpoint: {cfg.checkpoint_dir}/best.pt')
"

echo ""
echo "[Phase 1.2] V1.1 training finished. Next: run Phase 1.3 PSNR comparison."

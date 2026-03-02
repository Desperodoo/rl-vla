#!/bin/bash
# Wrapper to run BC scaling in tmux with proper conda init
set -e
cd /home/wjz/rl-vla
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3
echo "Conda env: $(conda info --envs | grep '*')"
echo "Python: $(which python)"
echo "Starting BC scaling..."
mkdir -p logs/vlaw/bc_scaling results/vlaw/bc_scaling
exec bash scripts/vlaw/run_bc_scaling.sh "$@"

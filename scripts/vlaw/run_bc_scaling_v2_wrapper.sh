#!/bin/bash
# Wrapper to run BC scaling V2 with conda initialization in tmux
set -e
cd /home/wjz/rl-vla
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3
echo "Conda env: $(conda info --envs | grep '*')"
echo "Python: $(which python)"
echo "Starting BC Scaling V2 (20K steps, 6 groups, parallel)..."
mkdir -p logs/vlaw/bc_scaling_v2 results/vlaw/bc_scaling_v2
exec bash scripts/vlaw/run_bc_scaling_v2_parallel.sh "$@"

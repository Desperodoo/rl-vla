#!/bin/bash
cd /home/wjz/rl-vla
source /home/wjz/miniconda3/etc/profile.d/conda.sh
conda activate rlft_ms3

CUDA_VISIBLE_DEVICES=3 python scripts/viz_ablation_comparison.py \
  --dirs data/vlaw/synthetic/wm_eval_step1200_short data/vlaw/synthetic/ablation_steps50 data/vlaw/synthetic/ablation_cfg5 data/vlaw/synthetic/ablation_cfg1 \
  --labels baseline_steps25_cfg2 steps50 cfg5 cfg1 \
  --output_dir data/vlaw/synthetic/ablation_comparison \
  --num_trajs 5 --num_frames 6 --gpu_id 0

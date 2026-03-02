#!/bin/bash
# T-EXP-WM-05: WM 最优步数搜索训练
set -e

# Conda 初始化
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ctrl_world

cd /home/wjz/rl-vla/ctrl_world

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

echo "[$(date)] Starting WM optimal steps training..."
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

accelerate launch --num_processes 4 --use_deepspeed --deepspeed_config_file ds_zero2.json \
  scripts/train_wm.py \
  --ckpt_path ../checkpoints/vlaw/world_model/pretrained/Ctrl-World/checkpoint-10000.pt \
  --dataset_root_path ../data/vlaw/encoded \
  --dataset_meta_info_path ../data/vlaw/meta_info/maniskill \
  --output_dir ../checkpoints/vlaw/world_model/ablation_optimal_steps \
  --max_train_steps 2000 --validation_steps 100 --checkpointing_steps 500 \
  --gradient_accumulation_steps 8 \
  --task_type maniskill --height 384 --width 192 --action_dim 7 \
  --num_frames 15 --num_history 4

echo "[$(date)] Training completed."

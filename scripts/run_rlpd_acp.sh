#!/bin/bash
# Full RLPD + ACP reward training run
# GPU 0: SAC RL training + ManiSkill envs
# GPU 1: ACP value model inference
cd /home/wjz/rl-vla

export PYTHONPATH=/home/wjz/rl-vla
export CUDA_VISIBLE_DEVICES=0,1

conda run -n rlft_ms3 python -m rlft.online.train_rlpd \
  --reward_mode acp \
  --acp_checkpoint checkpoints/vlaw/acp/iter1/best.safetensors \
  --acp_device cuda:1 \
  --total_timesteps 500000 \
  --num_envs 16 \
  --track \
  --wandb_project_name rlpd-acp \
  --exp_name acp_sac

#!/bin/bash
# ============================================================================
# run_rlpd_sac_acp_v2.sh — SAC from scratch + ACP v2_combined reward (retrain)
#
# Replaces the original run (wandb bxl448kv) which used demo-only overfitted ACP.
# Run AFTER train_acp_multi.sh completes.
#
# GPU layout:
#   cuda:0 → SAC RL training + ManiSkill 16 envs
#   cuda:1 → ACP value model inference
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_rlpd_sac_acp_v2.sh
# ============================================================================

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

ACP_CKPT="${ACP_CKPT:-checkpoints/vlaw/acp/v2_combined/best.safetensors}"

if [[ ! -f "$ACP_CKPT" ]]; then
    echo "[run_rlpd_sac_acp_v2] ERROR: ACP checkpoint not found: ${ACP_CKPT}"
    echo "  Run: bash scripts/train_acp_multi.sh --version v2_combined"
    exit 1
fi

echo "[run_rlpd_sac_acp_v2] ACP checkpoint : ${ACP_CKPT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

conda run -n rlft_ms3 python -m rlft.online.train_rlpd \
    --algorithm sac \
    \
    --reward_mode acp \
    --acp_checkpoint "${ACP_CKPT}" \
    --acp_device cuda:1 \
    --acp_reward_scale 100.0 \
    \
    --env_id LiftPegUpright-v1 \
    --num_envs 16 \
    --total_timesteps 500000 \
    --max_episode_steps 100 \
    \
    --online_ratio 0.15 \
    --utd_ratio 20 \
    --lr_actor 1e-4 \
    --lr_critic 1e-4 \
    --num_qs 10 \
    --num_min_qs 2 \
    \
    --seed 42 \
    --track \
    --wandb_project_name rlpd-acp \
    --exp_name sac_acp_v2

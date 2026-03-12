#!/bin/bash
# ============================================================================
# run_rlpd_awsc_acp.sh — RLPD with AWSC (pretrained policy init) + ACP reward
#
# GPU layout (parallel with SAC run):
#   cuda:0 → AWSC RL training + ManiSkill 16 envs
#   cuda:1 → ACP value model inference
#
# Pretrained policy (original IL-trained, same as compare_data_efficiency input):
#   runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt
#
# ACP checkpoint (update path after running train_acp_multi.sh):
#   checkpoints/vlaw/acp/v2_combined/best.safetensors  ← use this after retrain
#   checkpoints/vlaw/acp/iter1/best.safetensors         ← fallback (demo-only)
#
# Usage:
#   # After ACP v2 training:
#   ACP_CKPT=checkpoints/vlaw/acp/v2_combined/best.safetensors \
#     CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_rlpd_awsc_acp.sh
#
#   # With demo-only ACP (baseline):
#   ACP_CKPT=checkpoints/vlaw/acp/iter1/best.safetensors \
#     CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_rlpd_awsc_acp.sh
# ============================================================================

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

# ── ACP checkpoint (override via env var) ────────────────────────────────────
ACP_CKPT="${ACP_CKPT:-checkpoints/vlaw/acp/v2_combined/best.safetensors}"

# Fallback to demo-only checkpoint if v2_combined not yet trained
if [[ ! -f "$ACP_CKPT" ]]; then
    echo "[run_rlpd_awsc_acp] WARNING: ${ACP_CKPT} not found, falling back to iter1"
    ACP_CKPT="checkpoints/vlaw/acp/iter1/best.safetensors"
fi

# Use the ORIGINAL IL-trained ShortCut Flow policy (same as compare_data_efficiency input).
# NOT the RLPD-finetuned output (best_s42__1772570560) — that already has high SR
# and cannot demonstrate improvement from ACP reward.
PRETRAIN_PATH="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"

echo "[run_rlpd_awsc_acp] ACP checkpoint : ${ACP_CKPT}"
echo "[run_rlpd_awsc_acp] Pretrain path  : ${PRETRAIN_PATH}"
echo "[run_rlpd_awsc_acp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}"

# ── Use environment's CUDA_VISIBLE_DEVICES (default 2,3) ─────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

conda run -n rlft_ms3 python -m rlft.online.train_rlpd \
    --algorithm awsc \
    --pretrain_path "${PRETRAIN_PATH}" \
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
    --awsc_beta 50.0 \
    --awsc_bc_weight 2.0 \
    --awsc_advantage_mode per_state_v \
    --awsc_num_inference_steps 8 \
    \
    --seed 42 \
    --track \
    --wandb_project_name rlpd-acp \
    --exp_name awsc_acp_v2

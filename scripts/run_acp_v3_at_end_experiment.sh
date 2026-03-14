#!/bin/bash
# ============================================================================
# run_acp_v3_at_end_experiment.sh
#
# Two-phase experiment:
#   Phase 1: Train ACP v3_at_end value model with success_at_end semantics
#            (GPU 0, ~30 min for 12K steps)
#   Phase 2: Run AWSC + ACP v3_at_end mirror experiment
#            (GPU 0+1, ~6h for 500K steps)
#
# This tests whether changing the ACP value target from success_once to
# success_at_end improves success_at_end performance in RLPD.
#
# Environment: rlft_ms3
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

SEED="${SEED:-42}"

# ── Phase 1: Train ACP v3_at_end ──────────────────────────────────────────
ACP_OUT="checkpoints/vlaw/acp/v3_at_end"
ACP_BEST="${ACP_OUT}/best.safetensors"

DIR_DEMO="data/vlaw/rollouts/mixed"
DIR_POL="data/vlaw/rollouts/pretrained_policy"
DIR_TELE="data/vlaw/rollouts/teleop_sim"
DIR_RL="data/vlaw/rollouts/rl_prior"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 1] Training ACP v3_at_end (success_at_end semantics)"
echo "  Output: ${ACP_OUT}"
echo "  GPU: 0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=0 conda run -n rlft_ms3 \
    python rlft/vlaw/scripts/run_acp_train.py \
    --data_dirs "${DIR_DEMO}" "${DIR_POL}" "${DIR_TELE}" "${DIR_RL}" \
    --output_dir "${ACP_OUT}" \
    --num_steps 12000 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --eval_interval 200 \
    --save_interval 1000 \
    --value-target.success-mode success_at_end \
    --wandb_run_name "acp_v3_at_end" \
    2>&1 | tee logs/vlaw/acp_v3_at_end_train.log

if [[ ! -f "$ACP_BEST" ]]; then
    echo "[Phase 1] ERROR: ACP checkpoint not found at ${ACP_BEST}"
    exit 1
fi
echo "[Phase 1] ✅ ACP v3_at_end training complete: ${ACP_BEST}"

# ── Phase 2: AWSC + ACP v3_at_end ────────────────────────────────────────
CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Phase 2] AWSC + ACP v3_at_end (GPU 0+1, 500K steps)"
echo "  ACP: ${ACP_BEST}"
echo "  Pretrained: ${CHECKPOINT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=0,1 conda run -n rlft_ms3 \
    python -m rlft.online.train_rlpd \
    --algorithm awsc \
    --pretrain_path "${CHECKPOINT}" \
    \
    --reward_mode acp \
    --acp_checkpoint "${ACP_BEST}" \
    --acp_device cuda:1 \
    --acp_reward_scale 100.0 \
    \
    --env_id LiftPegUpright-v1 \
    --num_envs 50 \
    --num_eval_envs 50 \
    --total_timesteps 500000 \
    --max_episode_steps 100 \
    \
    --online_ratio 0.15 \
    --utd_ratio 20 \
    --lr_actor 1e-4 \
    --lr_critic 1e-4 \
    --num_qs 10 \
    --num_min_qs 2 \
    --awsc_beta 50.0 \
    --awsc_bc_weight 2.0 \
    --awsc_advantage_mode per_state_v \
    --awsc_num_inference_steps 8 \
    \
    --seed "${SEED}" \
    --track \
    --wandb_project_name rlpd-acp-mirror \
    --exp_name "awsc_acp_v3_at_end_s${SEED}" \
    2>&1 | tee logs/vlaw/acp_v3_at_end_awsc_s${SEED}.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[DONE] Full experiment complete."
echo "  ACP v3_at_end: ${ACP_BEST}"
echo "  AWSC run: runs/awsc_acp_v3_at_end_s${SEED}*/"
echo "  Logs: logs/vlaw/acp_v3_at_end_*.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

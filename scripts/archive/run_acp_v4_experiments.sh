#!/bin/bash
# ============================================================================
# run_acp_v4_experiments.sh — RLPD experiments with ACP v4 prescriptions
#
# Based on v3 internals diagnosis (docs/vlaw/figures/rlpd_acp_v3_internals/):
#   Prescription 1: Increase ACP reward scale 100 → 500
#   Prescription 2: Increase BC weight 2 → 4/8  (AWSC only)
#   Prescription 3: Lower gamma (PLD 0.99→0.7, DSRL 0.95→0.7)
#   Prescription 4: Early stopping on SO degradation (AWSC only)
#   SAE-aware checkpoint saving (built into all three scripts)
#
# V3 diagnoses:
#   AWSC: Critic OK (Q_range=3.9), but reward gap 90-350x, actor overfitting
#   PLD:  Critic F (Q_range=114-140) from gamma=0.99 → fix: gamma=0.7
#   DSRL: Critic F (Q_range=76-86)  from gamma=0.95 → fix: gamma=0.7
#
# GPU layout (each experiment uses 2 GPUs):
#   GPU 0+1: AWSC + v3_so + bc=4 + scale=500 + early_stop  (500K max)
#   GPU 2+3: AWSC + v3_so + bc=8 + scale=500 + early_stop  (500K max)
#   GPU 4+5: PLD  + v3_so + gamma=0.7 + scale=500           (71K)
#   GPU 6+7: DSRL + v3_so + gamma=0.7 + scale=500           (71K)
#
# Usage:
#   bash scripts/run_acp_v4_experiments.sh           # Launch all 4
#   bash scripts/run_acp_v4_experiments.sh --awsc     # Launch AWSC only
#   bash scripts/run_acp_v4_experiments.sh --pld-dsrl # Launch PLD+DSRL only
#   bash scripts/run_acp_v4_experiments.sh --status   # Check status
#
# Environment: rlft_ms3
# WandB project: rlpd-acp-v4
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

# ── Common settings ──────────────────────────────────────────────────────
CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
ACP_CKPT="checkpoints/vlaw/acp/v3_so/best.safetensors"
SEED="${SEED:-42}"
ENV_ID="LiftPegUpright-v1"
NUM_ENVS=50
NUM_EVAL_ENVS=50
MAX_EPISODE_STEPS=100
WANDB_PROJECT="rlpd-acp-v4"

# ── v4 prescription values ──────────────────────────────────────────────
ACP_REWARD_SCALE=500
TOTAL_STEPS_AWSC=500000
TOTAL_STEPS_PLD=71000
TOTAL_STEPS_DSRL=71000

# ── Parse arguments ──────────────────────────────────────────────────────
MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --status) MODE="status"; shift ;;
        --awsc) MODE="awsc"; shift ;;
        --pld-dsrl) MODE="pld-dsrl"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Verify prerequisites ─────────────────────────────────────────────────
for f in "$CHECKPOINT" "$ACP_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "[v4_exp] ERROR: Not found: $f"
        exit 1
    fi
done

mkdir -p logs/vlaw

# ── Status check ─────────────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v4_exp] Status check"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for log in logs/vlaw/acp_v4_*.log; do
        if [[ -f "$log" ]]; then
            name=$(basename "$log" .log)
            last_line=$(tail -1 "$log" 2>/dev/null || echo "empty")
            echo "  $name: $last_line"
        fi
    done
    echo ""
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 0
fi

PIDS=()

# ══════════════════════════════════════════════════════════════════════════
# AWSC experiments (GPU 0-3)
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "all" || "$MODE" == "awsc" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v4_exp] Launching AWSC experiments (prescription-tuned)"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  ACP:        ${ACP_CKPT}"
    echo "  Scale:      ${ACP_REWARD_SCALE}"
    echo "  Seed:       ${SEED}"
    echo "  WandB:      ${WANDB_PROJECT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── AWSC + bc_weight=4 (GPU 0+1) ─────────────────────────────────────
    echo "[v4_exp] Launching AWSC + bc=4 + scale=500 (GPU 0+1) ..."
    CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        --algorithm awsc \
        --pretrain_path "${CHECKPOINT}" \
        --reward_mode acp \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_AWSC}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --online_ratio 0.15 \
        --utd_ratio 20 \
        --lr_actor 1e-4 \
        --lr_critic 1e-4 \
        --num_qs 10 \
        --num_min_qs 2 \
        --awsc_beta 50.0 \
        --awsc_bc_weight 4.0 \
        --awsc_advantage_mode per_state_v \
        --awsc_num_inference_steps 8 \
        --early_stop \
        --early_stop_patience 5 \
        --early_stop_so_threshold 0.8 \
        --early_stop_min_steps 100000 \
        --seed "${SEED}" \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --exp_name "awsc_v4_bc4_scale500_s${SEED}" \
        > logs/vlaw/acp_v4_awsc_bc4_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> logs/vlaw/acp_v4_awsc_bc4_s${SEED}.log"

    # ── AWSC + bc_weight=8 (GPU 2+3) ─────────────────────────────────────
    echo "[v4_exp] Launching AWSC + bc=8 + scale=500 (GPU 2+3) ..."
    CUDA_VISIBLE_DEVICES=2,3 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_rlpd \
        --algorithm awsc \
        --pretrain_path "${CHECKPOINT}" \
        --reward_mode acp \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_AWSC}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --online_ratio 0.15 \
        --utd_ratio 20 \
        --lr_actor 1e-4 \
        --lr_critic 1e-4 \
        --num_qs 10 \
        --num_min_qs 2 \
        --awsc_beta 50.0 \
        --awsc_bc_weight 8.0 \
        --awsc_advantage_mode per_state_v \
        --awsc_num_inference_steps 8 \
        --early_stop \
        --early_stop_patience 5 \
        --early_stop_so_threshold 0.8 \
        --early_stop_min_steps 100000 \
        --seed "${SEED}" \
        --track \
        --wandb_project_name "${WANDB_PROJECT}" \
        --exp_name "awsc_v4_bc8_scale500_s${SEED}" \
        > logs/vlaw/acp_v4_awsc_bc8_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> logs/vlaw/acp_v4_awsc_bc8_s${SEED}.log"
fi

# ══════════════════════════════════════════════════════════════════════════
# PLD + DSRL experiments (GPU 4-7) — v4 prescription: lower gamma
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "all" || "$MODE" == "pld-dsrl" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v4_exp] Launching PLD + DSRL experiments (gamma-lowered)"
    echo "  PLD:  gamma 0.99→0.7, scale=${ACP_REWARD_SCALE}"
    echo "  DSRL: gamma 0.95→0.7, scale=${ACP_REWARD_SCALE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── PLD-SAC + gamma=0.7 + scale=500 (GPU 4+5) ────────────────────────
    echo "[v4_exp] Launching PLD + gamma=0.7 + scale=500 (GPU 4+5) ..."
    CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_pld \
        --checkpoint "${CHECKPOINT}" \
        --acp_reward \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_PLD}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --action_scale 0.3 \
        --utd_ratio 60 \
        --gamma 0.7 \
        --target_entropy -3.5 \
        --init_temperature 0.1 \
        --learning_rate 1e-4 \
        --num_layers 3 \
        --layer_size 1024 \
        --num_qs 5 \
        --calql_pretrain_steps 1000 \
        --calql_alpha 0.0 \
        --online_ratio 1.0 \
        --offline_demo_episodes 50 \
        --seed "${SEED}" \
        --track \
        --wandb_project "${WANDB_PROJECT}" \
        --exp_name "pld_v4_gamma07_scale500_s${SEED}" \
        > logs/vlaw/acp_v4_pld_gamma07_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> logs/vlaw/acp_v4_pld_gamma07_s${SEED}.log"

    # ── DSRL-SAC + gamma=0.7 + scale=500 (GPU 6+7) ───────────────────────
    echo "[v4_exp] Launching DSRL + gamma=0.7 + scale=500 (GPU 6+7) ..."
    CUDA_VISIBLE_DEVICES=6,7 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        --checkpoint "${CHECKPOINT}" \
        --acp_reward \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_DSRL}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --action_magnitude 2.5 \
        --utd_ratio 60 \
        --gamma 0.7 \
        --target_entropy -3.5 \
        --log_std_init -5.0 \
        --learning_rate 3e-4 \
        --num_layers 3 \
        --layer_size 2048 \
        --num_qs 10 \
        --num_seed_steps 0 \
        --seed "${SEED}" \
        --track \
        --wandb_project "${WANDB_PROJECT}" \
        --exp_name "dsrl_v4_gamma07_scale500_s${SEED}" \
        > logs/vlaw/acp_v4_dsrl_gamma07_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> logs/vlaw/acp_v4_dsrl_gamma07_s${SEED}.log"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[v4_exp] Launched. PIDs: ${PIDS[*]:-none}"
echo ""
echo "Monitor:"
echo "  bash scripts/run_acp_v4_experiments.sh --status"
echo "  nvidia-smi"
echo ""
echo "When done, run diagnosis:"
echo "  python scripts/analyze_training_internals.py --project ${WANDB_PROJECT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

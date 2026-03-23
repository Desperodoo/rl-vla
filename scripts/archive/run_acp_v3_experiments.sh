#!/bin/bash
# ============================================================================
# run_acp_v3_experiments.sh — RLPD experiments with ACP v3 reward models
#
# Runs 3 algorithms (AWSC, PLD-SAC, DSRL-SAC) × 2 ACP versions (v3_so, v3_sae)
# = 6 experiments total.
#
# GPU layout (each experiment uses 2 GPUs):
#   Wave 1 (5 parallel):
#     GPU 0+1: AWSC  + v3_so    (500K steps)
#     GPU 2+3: AWSC  + v3_sae   (500K steps)
#     GPU 4+5: PLD   + v3_so    (71K steps)
#     GPU 6+7: PLD   + v3_sae   (71K steps)
#     GPU 8+9: DSRL  + v3_so    (71K steps)
#   Wave 2 (after PLD/DSRL finish):
#     GPU 4+5: DSRL  + v3_sae   (71K steps)
#
# Usage:
#   bash scripts/run_acp_v3_experiments.sh                   # Wave 1 (5 experiments)
#   bash scripts/run_acp_v3_experiments.sh --wave2            # Wave 2 (DSRL+v3_sae)
#   bash scripts/run_acp_v3_experiments.sh --algo awsc        # AWSC only (both versions)
#   bash scripts/run_acp_v3_experiments.sh --status           # Check status
#
# Environment: rlft_ms3
# WandB project: rlpd-acp-v3
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

# ── Common settings ──────────────────────────────────────────────────────
CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
ACP_V3_SO="checkpoints/vlaw/acp/v3_so/best.safetensors"
ACP_V3_SAE="checkpoints/vlaw/acp/v3_sae/best.safetensors"
SEED="${SEED:-42}"
ENV_ID="LiftPegUpright-v1"
NUM_ENVS=50
NUM_EVAL_ENVS=50
MAX_EPISODE_STEPS=100
ACP_REWARD_SCALE="${ACP_REWARD_SCALE:-100.0}"
WANDB_PROJECT="rlpd-acp-v3"

TOTAL_STEPS_AWSC=500000
TOTAL_STEPS_PLD=71000
TOTAL_STEPS_DSRL=71000

# ── Parse arguments ──────────────────────────────────────────────────────
MODE="wave1"
ALGO="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --wave2) MODE="wave2"; shift ;;
        --algo) ALGO="$2"; shift 2 ;;
        --status) MODE="status"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Verify prerequisites ─────────────────────────────────────────────────
for f in "$CHECKPOINT" "$ACP_V3_SO" "$ACP_V3_SAE"; do
    if [[ ! -f "$f" ]]; then
        echo "[v3_exp] ERROR: Not found: $f"
        exit 1
    fi
done

mkdir -p logs/vlaw

# ── Status check ─────────────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v3_exp] Status check"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for log in logs/vlaw/acp_v3_*.log; do
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
# Wave 1: 5 experiments parallel
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "wave1" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v3_exp] Wave 1 — 5 experiments"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  ACP v3_so:  ${ACP_V3_SO}"
    echo "  ACP v3_sae: ${ACP_V3_SAE}"
    echo "  Seed: ${SEED}"
    echo "  WandB: ${WANDB_PROJECT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── AWSC + v3_so (GPU 0+1) ───────────────────────────────────────────
    if [[ "$ALGO" == "all" || "$ALGO" == "awsc" ]]; then
        echo "[v3_exp] Launching AWSC + v3_so (GPU 0+1) ..."
        CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n rlft_ms3 --no-capture-output \
            env PYTHONPATH=/home/wjz/rl-vla \
            python -m rlft.online.train_rlpd \
            --algorithm awsc \
            --pretrain_path "${CHECKPOINT}" \
            --reward_mode acp \
            --acp_checkpoint "${ACP_V3_SO}" \
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
            --awsc_bc_weight 2.0 \
            --awsc_advantage_mode per_state_v \
            --awsc_num_inference_steps 8 \
            --seed "${SEED}" \
            --track \
            --wandb_project_name "${WANDB_PROJECT}" \
            --exp_name "awsc_acp_v3_so_s${SEED}" \
            > logs/vlaw/acp_v3_awsc_so_s${SEED}.log 2>&1 &
        PIDS+=($!)
        echo "  PID=$! → logs/vlaw/acp_v3_awsc_so_s${SEED}.log"
    fi

    # ── AWSC + v3_sae (GPU 2+3) ──────────────────────────────────────────
    if [[ "$ALGO" == "all" || "$ALGO" == "awsc" ]]; then
        echo "[v3_exp] Launching AWSC + v3_sae (GPU 2+3) ..."
        CUDA_VISIBLE_DEVICES=2,3 nohup conda run -n rlft_ms3 --no-capture-output \
            env PYTHONPATH=/home/wjz/rl-vla \
            python -m rlft.online.train_rlpd \
            --algorithm awsc \
            --pretrain_path "${CHECKPOINT}" \
            --reward_mode acp \
            --acp_checkpoint "${ACP_V3_SAE}" \
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
            --awsc_bc_weight 2.0 \
            --awsc_advantage_mode per_state_v \
            --awsc_num_inference_steps 8 \
            --seed "${SEED}" \
            --track \
            --wandb_project_name "${WANDB_PROJECT}" \
            --exp_name "awsc_acp_v3_sae_s${SEED}" \
            > logs/vlaw/acp_v3_awsc_sae_s${SEED}.log 2>&1 &
        PIDS+=($!)
        echo "  PID=$! → logs/vlaw/acp_v3_awsc_sae_s${SEED}.log"
    fi

    # ── PLD-SAC + v3_so (GPU 4+5) ────────────────────────────────────────
    if [[ "$ALGO" == "all" || "$ALGO" == "pld" ]]; then
        echo "[v3_exp] Launching PLD + v3_so (GPU 4+5) ..."
        CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 --no-capture-output \
            env PYTHONPATH=/home/wjz/rl-vla \
            python -m rlft.online.train_pld \
            --checkpoint "${CHECKPOINT}" \
            --acp_reward \
            --acp_checkpoint "${ACP_V3_SO}" \
            --acp_device cuda:1 \
            --acp_reward_scale "${ACP_REWARD_SCALE}" \
            --env_id "${ENV_ID}" \
            --num_envs "${NUM_ENVS}" \
            --num_eval_envs "${NUM_EVAL_ENVS}" \
            --total_timesteps "${TOTAL_STEPS_PLD}" \
            --max_episode_steps "${MAX_EPISODE_STEPS}" \
            --action_scale 0.3 \
            --utd_ratio 60 \
            --gamma 0.99 \
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
            --exp_name "pld_acp_v3_so_s${SEED}" \
            > logs/vlaw/acp_v3_pld_so_s${SEED}.log 2>&1 &
        PIDS+=($!)
        echo "  PID=$! → logs/vlaw/acp_v3_pld_so_s${SEED}.log"
    fi

    # ── PLD-SAC + v3_sae (GPU 6+7) ───────────────────────────────────────
    if [[ "$ALGO" == "all" || "$ALGO" == "pld" ]]; then
        echo "[v3_exp] Launching PLD + v3_sae (GPU 6+7) ..."
        CUDA_VISIBLE_DEVICES=6,7 nohup conda run -n rlft_ms3 --no-capture-output \
            env PYTHONPATH=/home/wjz/rl-vla \
            python -m rlft.online.train_pld \
            --checkpoint "${CHECKPOINT}" \
            --acp_reward \
            --acp_checkpoint "${ACP_V3_SAE}" \
            --acp_device cuda:1 \
            --acp_reward_scale "${ACP_REWARD_SCALE}" \
            --env_id "${ENV_ID}" \
            --num_envs "${NUM_ENVS}" \
            --num_eval_envs "${NUM_EVAL_ENVS}" \
            --total_timesteps "${TOTAL_STEPS_PLD}" \
            --max_episode_steps "${MAX_EPISODE_STEPS}" \
            --action_scale 0.3 \
            --utd_ratio 60 \
            --gamma 0.99 \
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
            --exp_name "pld_acp_v3_sae_s${SEED}" \
            > logs/vlaw/acp_v3_pld_sae_s${SEED}.log 2>&1 &
        PIDS+=($!)
        echo "  PID=$! → logs/vlaw/acp_v3_pld_sae_s${SEED}.log"
    fi

    # ── DSRL-SAC + v3_so (GPU 8+9) ───────────────────────────────────────
    if [[ "$ALGO" == "all" || "$ALGO" == "dsrl" ]]; then
        echo "[v3_exp] Launching DSRL + v3_so (GPU 8+9) ..."
        CUDA_VISIBLE_DEVICES=8,9 nohup conda run -n rlft_ms3 --no-capture-output \
            env PYTHONPATH=/home/wjz/rl-vla \
            python -m rlft.online.train_dsrl \
            --checkpoint "${CHECKPOINT}" \
            --acp_reward \
            --acp_checkpoint "${ACP_V3_SO}" \
            --acp_device cuda:1 \
            --acp_reward_scale "${ACP_REWARD_SCALE}" \
            --env_id "${ENV_ID}" \
            --num_envs "${NUM_ENVS}" \
            --num_eval_envs "${NUM_EVAL_ENVS}" \
            --total_timesteps "${TOTAL_STEPS_DSRL}" \
            --max_episode_steps "${MAX_EPISODE_STEPS}" \
            --action_magnitude 2.5 \
            --utd_ratio 60 \
            --gamma 0.95 \
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
            --exp_name "dsrl_acp_v3_so_s${SEED}" \
            > logs/vlaw/acp_v3_dsrl_so_s${SEED}.log 2>&1 &
        PIDS+=($!)
        echo "  PID=$! → logs/vlaw/acp_v3_dsrl_so_s${SEED}.log"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v3_exp] Wave 1 launched. PIDs: ${PIDS[*]:-none}"
    echo ""
    echo "Monitor:"
    echo "  bash scripts/run_acp_v3_experiments.sh --status"
    echo "  nvidia-smi"
    echo ""
    echo "When PLD/DSRL finish (~71K steps), launch Wave 2:"
    echo "  bash scripts/run_acp_v3_experiments.sh --wave2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ══════════════════════════════════════════════════════════════════════════
# Wave 2: DSRL + v3_sae (after PLD/DSRL v3_so finish)
# ══════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "wave2" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[v3_exp] Wave 2 — DSRL + v3_sae (GPU 4+5)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 --no-capture-output \
        env PYTHONPATH=/home/wjz/rl-vla \
        python -m rlft.online.train_dsrl \
        --checkpoint "${CHECKPOINT}" \
        --acp_reward \
        --acp_checkpoint "${ACP_V3_SAE}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_DSRL}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --action_magnitude 2.5 \
        --utd_ratio 60 \
        --gamma 0.95 \
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
        --exp_name "dsrl_acp_v3_sae_s${SEED}" \
        > logs/vlaw/acp_v3_dsrl_sae_s${SEED}.log 2>&1 &
    echo "  PID=$! → logs/vlaw/acp_v3_dsrl_sae_s${SEED}.log"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

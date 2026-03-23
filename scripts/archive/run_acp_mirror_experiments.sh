#!/bin/bash
# ============================================================================
# run_acp_mirror_experiments.sh — Mirror of compare_data_efficiency with ACP reward
#
# Runs the same 3 algorithms (AWSC, PLD-SAC, DSRL-SAC) with identical
# hyperparameters as scripts/compare_data_efficiency/configs/fair_comparison.sh,
# but replaces sim dense reward with ACP TD-shaped reward.
#
# This enables direct comparison:
#   - Sim reward results:  runs/fair_comparison/{awsc,pld,dsrl}/
#   - ACP reward results:  runs/acp_mirror/{awsc,pld,dsrl}/
#
# Pretrained checkpoint (shared, same as compare_data_efficiency):
#   runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt
#
# GPU layout (each experiment uses 2 GPUs: training + ACP inference):
#   Experiment 1 (AWSC):  cuda:0 (train) + cuda:1 (ACP)
#   Experiment 2 (PLD):   cuda:2 (train) + cuda:3 (ACP)
#   Experiment 3 (DSRL):  cuda:4 (train) + cuda:5 (ACP)
#
# Usage:
#   bash scripts/run_acp_mirror_experiments.sh                    # all 3 parallel
#   bash scripts/run_acp_mirror_experiments.sh --algo awsc        # only AWSC
#   bash scripts/run_acp_mirror_experiments.sh --algo pld         # only PLD
#   bash scripts/run_acp_mirror_experiments.sh --algo dsrl        # only DSRL
#   SEED=100 bash scripts/run_acp_mirror_experiments.sh           # custom seed
#
# Environment: rlft_ms3
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

# ── Common settings (matching compare_data_efficiency/config.sh) ────────────
CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
ACP_CKPT="${ACP_CKPT:-checkpoints/vlaw/acp/v2_combined/best.safetensors}"
SEED="${SEED:-42}"
ENV_ID="LiftPegUpright-v1"
NUM_ENVS=50
NUM_EVAL_ENVS=50
MAX_EPISODE_STEPS=100
ACP_REWARD_SCALE="${ACP_REWARD_SCALE:-100.0}"

# Comparison Scheme A: unified real robot steps
TOTAL_STEPS_AWSC=500000
TOTAL_STEPS_PLD=71000
TOTAL_STEPS_DSRL=71000

# ── Parse arguments ─────────────────────────────────────────────────────────
ALGO="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --algo) ALGO="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Verify prerequisites ────────────────────────────────────────────────────
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[acp_mirror] ERROR: Pretrained checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

if [[ ! -f "$ACP_CKPT" ]]; then
    echo "[acp_mirror] ERROR: ACP checkpoint not found: ${ACP_CKPT}"
    echo "  Run: bash scripts/train_acp_multi.sh --version v2_combined"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[acp_mirror] ACP Mirror Experiments — compare_data_efficiency"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  ACP model:   ${ACP_CKPT}"
echo "  Seed:        ${SEED}"
echo "  Algorithm:   ${ALGO}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p logs/vlaw

PIDS=()

# ── Experiment 1: AWSC + ACP ────────────────────────────────────────────────
# Same hyperparameters as fair_comparison.sh AWSC_CONFIGS "best"
if [[ "$ALGO" == "all" || "$ALGO" == "awsc" ]]; then
    echo "[acp_mirror] Launching AWSC + ACP (GPU 0+1) ..."
    CUDA_VISIBLE_DEVICES=0,1 nohup conda run -n rlft_ms3 \
        python -m rlft.online.train_rlpd \
        --algorithm awsc \
        --pretrain_path "${CHECKPOINT}" \
        \
        --reward_mode acp \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_AWSC}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
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
        --exp_name "awsc_acp_mirror_s${SEED}" \
        > logs/vlaw/acp_mirror_awsc_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! → logs/vlaw/acp_mirror_awsc_s${SEED}.log"
fi

# ── Experiment 2: PLD-SAC + ACP ────────────────────────────────────────────
# Same hyperparameters as fair_comparison.sh PLD_CONFIGS "best"
if [[ "$ALGO" == "all" || "$ALGO" == "pld" ]]; then
    echo "[acp_mirror] Launching PLD-SAC + ACP (GPU 2+3) ..."
    CUDA_VISIBLE_DEVICES=2,3 nohup conda run -n rlft_ms3 \
        python -m rlft.online.train_pld \
        --checkpoint "${CHECKPOINT}" \
        \
        --acp_reward \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_PLD}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        \
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
        \
        --seed "${SEED}" \
        --track \
        --wandb_project "rlpd-acp-mirror" \
        --exp_name "pld_acp_mirror_s${SEED}" \
        > logs/vlaw/acp_mirror_pld_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! → logs/vlaw/acp_mirror_pld_s${SEED}.log"
fi

# ── Experiment 3: DSRL-SAC + ACP ───────────────────────────────────────────
# Same hyperparameters as fair_comparison.sh DSRL_CONFIGS "best"
if [[ "$ALGO" == "all" || "$ALGO" == "dsrl" ]]; then
    echo "[acp_mirror] Launching DSRL-SAC + ACP (GPU 4+5) ..."
    CUDA_VISIBLE_DEVICES=4,5 nohup conda run -n rlft_ms3 \
        python -m rlft.online.train_dsrl \
        --checkpoint "${CHECKPOINT}" \
        \
        --acp_reward \
        --acp_checkpoint "${ACP_CKPT}" \
        --acp_device cuda:1 \
        --acp_reward_scale "${ACP_REWARD_SCALE}" \
        \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_eval_envs "${NUM_EVAL_ENVS}" \
        --total_timesteps "${TOTAL_STEPS_DSRL}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        \
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
        \
        --seed "${SEED}" \
        --track \
        --wandb_project "rlpd-acp-mirror" \
        --exp_name "dsrl_acp_mirror_s${SEED}" \
        > logs/vlaw/acp_mirror_dsrl_s${SEED}.log 2>&1 &
    PIDS+=($!)
    echo "  PID=$! → logs/vlaw/acp_mirror_dsrl_s${SEED}.log"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[acp_mirror] All experiments launched. PIDs: ${PIDS[*]:-none}"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/vlaw/acp_mirror_awsc_s${SEED}.log"
echo "  tail -f logs/vlaw/acp_mirror_pld_s${SEED}.log"
echo "  tail -f logs/vlaw/acp_mirror_dsrl_s${SEED}.log"
echo ""
echo "GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Compare with sim-reward baselines in runs/fair_comparison/"

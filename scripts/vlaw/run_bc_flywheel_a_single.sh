#!/bin/bash
# =============================================================================
# T-BC-FLYWHEEL-A: Single experiment runner
# 纯 demo 从头训练 ShortCut Flow (100K 步)，作为数据飞轮 A 组对照
#
# Usage:
#   bash scripts/vlaw/run_bc_flywheel_a_single.sh <num_demos> <gpu_id>
#
# Example:
#   bash scripts/vlaw/run_bc_flywheel_a_single.sh 100 0
# =============================================================================
set -e

NUM_DEMOS=$1
GPU_ID=$2

if [ -z "$NUM_DEMOS" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <num_demos> <gpu_id>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- Configuration ---
DEMO_PATH="$HOME/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
ENV_ID="LiftPegUpright-v1"
ALGORITHM="shortcut_flow"
TOTAL_ITERS=100000
EVAL_FREQ=10000
SAVE_FREQ=50000
BATCH_SIZE=256
LR=3e-4
NUM_EVAL_EPISODES=50
NUM_EVAL_ENVS=25
WANDB_PROJECT="BC_Flywheel_A_LiftPeg"
SEED=1

EXP_NAME="bc_flywheel_a_${NUM_DEMOS}demos_${TOTAL_ITERS}steps"
LOG_DIR="/home/wjz/rl-vla/logs/vlaw/bc_flywheel_a"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_gpu${GPU_ID}.log"
RESULTS_DIR="/home/wjz/rl-vla/results/vlaw/bc_flywheel_a"
RESULTS_JSONL="${RESULTS_DIR}/flywheel_a_results.jsonl"
CKPT_DIR="/home/wjz/rl-vla/checkpoints/vlaw/policy/flywheel_a_${NUM_DEMOS}demos"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

echo "=============================================="
echo "T-BC-FLYWHEEL-A: ${NUM_DEMOS} demos on GPU ${GPU_ID}"
echo "=============================================="
echo "Start time: $(date)"
echo "Total iters: $TOTAL_ITERS"
echo "Eval freq: $EVAL_FREQ"
echo "Save freq: $SAVE_FREQ"
echo "Eval episodes: $NUM_EVAL_EPISODES"
echo "Checkpoint dir: $CKPT_DIR"
echo ""

cd /home/wjz/rl-vla

python -m rlft.offline.train_maniskill \
    --env_id "$ENV_ID" \
    --demo_path "$DEMO_PATH" \
    --algorithm "$ALGORITHM" \
    --num_demos "$NUM_DEMOS" \
    --total_iters "$TOTAL_ITERS" \
    --eval_freq "$EVAL_FREQ" \
    --save_freq "$SAVE_FREQ" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_eval_episodes "$NUM_EVAL_EPISODES" \
    --num_eval_envs "$NUM_EVAL_ENVS" \
    --seed "$SEED" \
    --exp_name "$EXP_NAME" \
    --track \
    --wandb_project_name "$WANDB_PROJECT" \
    --obs_mode rgb \
    --pred_horizon 8 \
    --obs_horizon 2 \
    --sc_fixed_step_size 0.15 \
    --sc_num_inference_steps 8 \
    --sc_self_consistency_k 0.25 \
    --sc_step_size_mode fixed \
    --ema_decay 0.9995 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Extract eval metrics from log
BEST_SUCCESS_ONCE=$(grep -oP "success_once.*?([0-9]+\.[0-9]+)" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+\.[0-9]+$' || echo "N/A")
BEST_SUCCESS_AT_END=$(grep -oP "success_at_end.*?([0-9]+\.[0-9]+)" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+\.[0-9]+$' || echo "N/A")

# Copy best checkpoint to designated directory
LATEST_RUN=$(ls -td /home/wjz/rl-vla/runs/${EXP_NAME}__* 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/best_eval_success_once.pt" ]; then
    cp "$LATEST_RUN/best_eval_success_once.pt" "$CKPT_DIR/"
    echo "Copied best checkpoint to $CKPT_DIR/best_eval_success_once.pt"
fi
if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/latest.pt" ]; then
    cp "$LATEST_RUN/latest.pt" "$CKPT_DIR/"
    echo "Copied latest checkpoint to $CKPT_DIR/latest.pt"
fi

# Append to JSONL
echo "{\"n_demos\": $NUM_DEMOS, \"total_iters\": $TOTAL_ITERS, \"best_success_once\": \"$BEST_SUCCESS_ONCE\", \"best_success_at_end\": \"$BEST_SUCCESS_AT_END\", \"exit_code\": $EXIT_CODE, \"gpu\": $GPU_ID, \"timestamp\": \"$(date +%Y%m%d_%H%M%S)\"}" >> "$RESULTS_JSONL"

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "[DONE] ${NUM_DEMOS} demos -> success_once=$BEST_SUCCESS_ONCE, success_at_end=$BEST_SUCCESS_AT_END"
else
    echo "[FAIL] ${NUM_DEMOS} demos (exit=$EXIT_CODE)"
fi
echo "Timestamp: $(date)"
echo "=============================================="

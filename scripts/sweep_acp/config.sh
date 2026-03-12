#!/bin/bash
# =============================================================================
# ACP Sweep Configuration — Global settings for ACP reward hyperparameter sweep
# =============================================================================
# Sweeps ACP reward-specific parameters (reward_scale, ACP version, etc.)
# for AWSC (via train_rlpd) algorithm.
#
# Each experiment uses 2 GPUs: 1 for RL training + 1 for ACP value model.
#
# Usage:
#   source config.sh
#   GPU_PAIRS="0,1 2,3 4,5 6,7 8,9" source config.sh
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration — pairs of (train_gpu, acp_gpu)
# Each experiment needs 2 GPUs. With 10 GPUs we run 5 experiments in parallel.
# -----------------------------------------------------------------------------
# GPU pairs: "train,acp" format
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7" "8,9")
NUM_GPU_PAIRS=${#GPU_PAIRS[@]}

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
ENV_ID="${ENV_ID:-LiftPegUpright-v1}"
OBS_MODE="${OBS_MODE:-rgb}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"
REWARD_MODE="${REWARD_MODE:-dense}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"

# -----------------------------------------------------------------------------
# Training Configuration
# AWSC=500K steps (matching fair_comparison)
# -----------------------------------------------------------------------------
TOTAL_STEPS_AWSC="${TOTAL_STEPS_AWSC:-500000}"

NUM_ENVS="${NUM_ENVS:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-50000000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"

# Seed
SEED="${SEED:-42}"

# -----------------------------------------------------------------------------
# Pretrained Checkpoint (shared across all algorithms)
# -----------------------------------------------------------------------------
DEFAULT_CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
PRED_HORIZON="${PRED_HORIZON:-8}"

# Demo path for RLPD/AWSC
DEMO_PATH="${DEMO_PATH:-~/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5}"

# -----------------------------------------------------------------------------
# ACP Configuration
# -----------------------------------------------------------------------------
DEFAULT_ACP_CKPT="checkpoints/vlaw/acp/v2_combined/best.safetensors"
ACP_CKPT="${ACP_CKPT:-${DEFAULT_ACP_CKPT}}"
ACP_REWARD_SCALE="${ACP_REWARD_SCALE:-100.0}"
ACP_TASK_INSTRUCTION="${ACP_TASK_INSTRUCTION:-Pick up the peg and lift it upright.}"

# Available ACP checkpoints (for version sweep)
declare -A ACP_VERSIONS=(
    ["v2_combined"]="checkpoints/vlaw/acp/v2_combined/best.safetensors"
    ["v2_teleop_sim"]="checkpoints/vlaw/acp/v2_teleop_sim/best.safetensors"
    ["v2_rl_prior"]="checkpoints/vlaw/acp/v2_rl_prior/best.safetensors"
    ["v2_pretrained_pol"]="checkpoints/vlaw/acp/v2_pretrained_pol/best.safetensors"
    ["v2_demo_only"]="checkpoints/vlaw/acp/v2_demo_only/best.safetensors"
)

# -----------------------------------------------------------------------------
# Sweep Configuration
# -----------------------------------------------------------------------------
CONFIG_VERSION="${CONFIG_VERSION:-v1}"

SWEEP_BASE_DIR="${SWEEP_BASE_DIR:-runs/acp_sweep}"
EXP_NAME="${EXP_NAME:-acp_sweep}"

# Algorithm definitions (AWSC only)
ALL_ALGORITHMS=("awsc_acp")

# Retry settings
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# -----------------------------------------------------------------------------
# WandB Configuration
# -----------------------------------------------------------------------------
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-ACP-Sweep}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# -----------------------------------------------------------------------------
# Algorithm-specific defaults (matching fair_comparison / acp_mirror best configs)
# -----------------------------------------------------------------------------

# AWSC defaults (from fair_comparison best + acp_mirror)
AWSC_ONLINE_RATIO="0.15"
AWSC_UTD_RATIO="20"
AWSC_LR_ACTOR="1e-4"
AWSC_LR_CRITIC="1e-4"
AWSC_NUM_QS="10"
AWSC_NUM_MIN_QS="2"
AWSC_BETA="50.0"
AWSC_BC_WEIGHT="2.0"
AWSC_ADVANTAGE_MODE="per_state_v"
AWSC_NUM_INFERENCE_STEPS="8"

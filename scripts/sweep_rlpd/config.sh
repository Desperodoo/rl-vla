#!/bin/bash
# =============================================================================
# Sweep Configuration - Global settings for RLPD hyperparameter sweep
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
AVAILABLE_GPUS=(1 2 3 4 5 6 7 8 9)
NUM_GPUS=${#AVAILABLE_GPUS[@]}

# -----------------------------------------------------------------------------
# Environment Configuration (对齐 sweep 文件夹)
# -----------------------------------------------------------------------------
ENV_ID="${ENV_ID:-LiftPegUpright-v1}"
OBS_MODE="${OBS_MODE:-rgb}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"

# Training timesteps (Online RL 使用 timesteps 而非 iterations)
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"

# Online RL 特有配置
NUM_ENVS="${NUM_ENVS:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-25}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"

# Demo path (对齐 sweep 文件夹，使用 ~/.maniskill/demos/)
DEMO_PATH="${DEMO_PATH:-~/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5}"

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
# Config version: "v1" (wave 1), "v2" (wave 2), or "v3" (wave 3)
# Can be overridden by --config-version flag
CONFIG_VERSION="${CONFIG_VERSION:-v1}"

if [[ "${CONFIG_VERSION}" == "v4" ]]; then
    EXP_NAME="${EXP_NAME:-rlpd_sweep_v4}"
    # v4 建议 500K 训练 (v3 250K 数据显示多数配置仍在上升)
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
elif [[ "${CONFIG_VERSION}" == "v3" ]]; then
    EXP_NAME="${EXP_NAME:-rlpd_sweep_v3}"
elif [[ "${CONFIG_VERSION}" == "v2" ]]; then
    EXP_NAME="${EXP_NAME:-rlpd_sweep_v2}"
else
    EXP_NAME="${EXP_NAME:-rlpd_sweep}"
fi
SWEEP_BASE_DIR="${SWEEP_BASE_DIR:-runs/${EXP_NAME}}"

# WandB 配置
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-${EXP_NAME}}"

# Maximum retry attempts for failed experiments
MAX_RETRIES="${MAX_RETRIES:-3}"

# Retry delay (seconds) after CUDA failure (Online RL 训练时间长，增加等待时间)
RETRY_DELAY="${RETRY_DELAY:-30}"

# -----------------------------------------------------------------------------
# Algorithm Definitions
# -----------------------------------------------------------------------------
# Stage 1: From scratch algorithms (无依赖)
STAGE1_ALGORITHMS=(sac)

# Stage 2: Algorithms that can use pretrained models
STAGE2_ALGORITHMS=(awsc)

# Default pretrained checkpoint paths (Wave 3 sweep 最优模型)
# AWSC: cw0.3_step0.15 — success_once=0.85 (Wave 3 全局最优)
DEFAULT_AWSC_PRETRAIN_PATH="${DEFAULT_AWSC_PRETRAIN_PATH:-runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt}"

ALL_ALGORITHMS=(
    "${STAGE1_ALGORITHMS[@]}"
    "${STAGE2_ALGORITHMS[@]}"
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-logs/sweep_rlpd}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

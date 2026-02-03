#!/bin/bash
# =============================================================================
# Sweep Configuration - Global settings for RLPD hyperparameter sweep
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
# 支持外部指定 GPU 列表
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    IFS=',' read -ra AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    # 自动检测可用 GPU
    _gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    AVAILABLE_GPUS=()
    for ((i=0; i<_gpu_count; i++)); do
        AVAILABLE_GPUS+=($i)
    done
    unset _gpu_count
fi
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
EXP_NAME="${EXP_NAME:-rlpd_sweep}"
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

ALL_ALGORITHMS=(
    "${STAGE1_ALGORITHMS[@]}"
    "${STAGE2_ALGORITHMS[@]}"
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-logs/sweep_rlpd}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

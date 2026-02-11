#!/bin/bash
# =============================================================================
# Sweep Configuration - Global settings for hyperparameter sweep
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
AVAILABLE_GPUS=(1 2 3 4 5 6 7 8 9)
NUM_GPUS=${#AVAILABLE_GPUS[@]}

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
ENV_ID="LiftPegUpright-v1"
OBS_MODE="rgb"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"
TOTAL_ITERS=50000

# Demo path (relative to workspace)
DEMO_PATH="~/.maniskill/demos/${ENV_ID}/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"

# -----------------------------------------------------------------------------
# Experiment Configuration
# -----------------------------------------------------------------------------
if [[ "${CONFIG_VERSION}" == "v3" ]]; then
    EXP_NAME="maniskill_sweep_v3"
elif [[ "${CONFIG_VERSION}" == "v2" ]]; then
    EXP_NAME="maniskill_sweep"
else
    EXP_NAME="maniskill_sweep"
fi
SWEEP_BASE_DIR="runs/${EXP_NAME}"

# Config version: "v1" (wave 1) or "v2" (wave 2)
# Can be overridden by --config-version flag
CONFIG_VERSION="${CONFIG_VERSION:-v1}"

# Maximum retry attempts for failed experiments
MAX_RETRIES=3

# Retry delay (seconds) after CUDA failure
RETRY_DELAY=10

# -----------------------------------------------------------------------------
# Algorithm Stage Definitions (for cascade sweep)
# Stage 1: Pure IL algorithms
# Stage 2: RL algorithms that depend on IL results
# Stage 3: RL algorithms that depend on Stage 2
# -----------------------------------------------------------------------------
if [[ "${CONFIG_VERSION}" == "v3" ]]; then
    # Wave 3: drop diffusion_policy / flow_matching / reflected_flow
    STAGE1_ALGORITHMS=(consistency_flow shortcut_flow)
    STAGE2_ALGORITHMS=(cpql awcp sac)
    STAGE3_ALGORITHMS=(aw_shortcut_flow)
else
    STAGE1_ALGORITHMS=(flow_matching diffusion_policy consistency_flow shortcut_flow reflected_flow)
    STAGE2_ALGORITHMS=(cpql awcp sac)
    STAGE3_ALGORITHMS=(aw_shortcut_flow)
fi

ALL_ALGORITHMS=(
    "${STAGE1_ALGORITHMS[@]}"
    "${STAGE2_ALGORITHMS[@]}"
    "${STAGE3_ALGORITHMS[@]}"
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_DIR="logs/sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

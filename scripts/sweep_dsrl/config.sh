#!/bin/bash
# =============================================================================
# DSRL Sweep Configuration
# =============================================================================
# Global settings for DSRL-SAC hyperparameter sweep.
# DSRL = Diffusion-based SAC in Residual/Noise space of a pretrained ShortCut Flow.
#
# Usage:
#   source config.sh          # Loaded automatically by utils.sh / sweep.sh
#   GPU_IDS="0,1" source config.sh   # Override GPU list
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
AVAILABLE_GPUS=(1 2 3 4 5 6 7 8 9)
NUM_GPUS=${#AVAILABLE_GPUS[@]}

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
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
NUM_ENVS="${NUM_ENVS:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"

# -----------------------------------------------------------------------------
# Pretrained Checkpoint (REQUIRED for DSRL)
# -----------------------------------------------------------------------------
# DSRL always requires a pretrained ShortCut Flow policy checkpoint.
DEFAULT_CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

# pred_horizon must match the offline training checkpoint (default 8)
PRED_HORIZON="${PRED_HORIZON:-8}"

# -----------------------------------------------------------------------------
# Sweep Configuration
# -----------------------------------------------------------------------------
CONFIG_VERSION="${CONFIG_VERSION:-v1}"

# Sweep base directory for experiment outputs
SWEEP_BASE_DIR="${SWEEP_BASE_DIR:-runs/dsrl_sweep}"
if [[ "$CONFIG_VERSION" == "v2" ]]; then
    SWEEP_BASE_DIR="runs/dsrl_sweep_v2"
fi

# Experiment name prefix
EXP_NAME="${EXP_NAME:-dsrl_sweep}"

# Algorithm definition (DSRL only has one algorithm)
ALL_ALGORITHMS=("dsrl_sac")

# Retry settings
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# -----------------------------------------------------------------------------
# WandB Configuration
# -----------------------------------------------------------------------------
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-DSRL-SAC-Sweep}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

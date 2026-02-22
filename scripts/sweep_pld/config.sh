#!/bin/bash
# =============================================================================
# PLD Sweep Configuration
# =============================================================================
# Global settings for PLD-SAC hyperparameter sweep.
# PLD = Policy-guided Learned Diffusion — SAC in residual action space of a
# pretrained ShortCut Flow, with Cal-QL pretraining and base policy probing.
#
# Usage:
#   source config.sh          # Loaded automatically by utils.sh / sweep.sh
#   GPU_IDS="0,1" source config.sh   # Override GPU list
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9)
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
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
NUM_ENVS="${NUM_ENVS:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-50000000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"

# -----------------------------------------------------------------------------
# Pretrained Checkpoint (REQUIRED for PLD)
# -----------------------------------------------------------------------------
# PLD always requires a pretrained ShortCut Flow policy checkpoint.
DEFAULT_CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

# pred_horizon must match the offline training checkpoint (default 8)
PRED_HORIZON="${PRED_HORIZON:-8}"

# -----------------------------------------------------------------------------
# PLD-specific Defaults
# -----------------------------------------------------------------------------
# Residual action scale ξ ∈ [-action_scale, +action_scale] (PLD sweep best)
ACTION_SCALE="${ACTION_SCALE:-0.3}"

# Cal-QL offline pretraining steps (0 = skip pretraining)
CALQL_PRETRAIN_STEPS="${CALQL_PRETRAIN_STEPS:-1000}"

# Cal-QL conservative loss coefficient
CALQL_ALPHA="${CALQL_ALPHA:-0.0}"

# Offline demo episodes collected by base policy
OFFLINE_DEMO_EPISODES="${OFFLINE_DEMO_EPISODES:-50}"

# Base policy probing (see PLD paper Section 3.2)
PROBE_STEPS="${PROBE_STEPS:-5}"
PROBING_ALPHA="${PROBING_ALPHA:-0.6}"

# Online / offline buffer mixing ratio
ONLINE_RATIO="${ONLINE_RATIO:-1.0}"

# Buffer sizes
ONLINE_BUFFER_SIZE="${ONLINE_BUFFER_SIZE:-500000}"
OFFLINE_BUFFER_SIZE="${OFFLINE_BUFFER_SIZE:-200000}"

# -----------------------------------------------------------------------------
# Sweep Configuration
# -----------------------------------------------------------------------------
CONFIG_VERSION="${CONFIG_VERSION:-v1}"

# Experiment name prefix and sweep base dir are derived unconditionally from
# CONFIG_VERSION so that re-sourcing (e.g. with `source sweep.sh`) always
# picks up the correct directory for the requested version.
if [[ "$CONFIG_VERSION" == "v2" ]]; then
    EXP_NAME="pld_sweep_v2"
elif [[ "$CONFIG_VERSION" == "v3" ]]; then
    EXP_NAME="pld_sweep_v3"
else
    EXP_NAME="pld_sweep"
fi
# Allow explicit override via environment variable
EXP_NAME="${EXP_NAME_OVERRIDE:-$EXP_NAME}"
SWEEP_BASE_DIR="${SWEEP_BASE_DIR_OVERRIDE:-runs/${EXP_NAME}}"

# Algorithm definition (PLD only has one algorithm)
ALL_ALGORITHMS=("pld_sac")

# Retry settings
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# -----------------------------------------------------------------------------
# WandB Configuration
# -----------------------------------------------------------------------------
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-PLD-SAC-Sweep}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

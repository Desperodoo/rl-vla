#!/bin/bash
# =============================================================================
# Data-Efficiency Comparison — Configuration
# =============================================================================
# Global settings for the fair comparison of AWSC (RLPD) vs PLD-SAC vs DSRL-SAC.
#
# 【X 轴定义 —— 三算法计步单位不同！】
#
#   AWSC (train_rlpd.py):
#     total_steps += num_envs 在 for step_idx in range(act_horizon) 内部
#     → 每个真实 robot step 都计数一次
#
#   PLD / DSRL (train_pld.py / train_dsrl.py):
#     train_adapter.step() 内部执行完整 act_chunk（~7 real steps），
#     之后才 total_steps += num_envs（计一次）
#
# 【对比方案】
#   方案 A：统一以「真实机器人步」为单位（推荐，物理含义清晰）
#     AWSC 500K robot steps  ≈  PLD/DSRL 71K chunk decisions × 7
#   方案 B：统一以「chunk 决策次数」为单位（SMDP 语义）
#     PLD/DSRL 500K chunks  ≈  AWSC 3.5M robot steps
#   方案 C：快速验证（x 轴不对齐）
#     三者均 500K，绘图说明差异即可
#
# Usage:
#   source config.sh                              # 默认方案 A
#   COMPARISON_SCHEME=B source config.sh          # 方案 B
#   GPU_IDS="5,6,7" SEED=100 source config.sh    # 自定义 GPU / seed
# =============================================================================

# -----------------------------------------------------------------------------
# GPU Configuration  (默认 GPU 0-1,3-9，避开 GPU 2 在用)
# -----------------------------------------------------------------------------
if [[ -n "${GPU_IDS:-}" ]]; then
    IFS=',' read -ra AVAILABLE_GPUS <<< "$GPU_IDS"
else
    AVAILABLE_GPUS=(0 1 3 4 5 6 7 8 9)
fi
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
# Pretrained Checkpoint (shared across all three algorithms)
# -----------------------------------------------------------------------------
DEFAULT_CHECKPOINT="runs/maniskill_sweep_v3/aw_shortcut_flow/cw0.3_step0.15__1770390417/checkpoints/best_eval_success_once.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

# pred_horizon must match the offline training checkpoint (default 8)
PRED_HORIZON="${PRED_HORIZON:-8}"
ACT_HORIZON="${ACT_HORIZON:-7}"     # pred_horizon - (obs_horizon - 1) = 8 - 1 = 7

# Demo path (AWSC/RLPD 需要)
DEMO_PATH="${DEMO_PATH:-~/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5}"

# -----------------------------------------------------------------------------
# Shared Training Configuration (控制变量)
# -----------------------------------------------------------------------------
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_FREQ="${SAVE_FREQ:-50000000}"      # 不保存中间 checkpoint，节省磁盘
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"

# Per-algorithm evaluation frequency:
#   AWSC 500K steps → eval_freq=5000 ≈ 100 eval points
#   PLD/DSRL 71K steps → eval_freq=2000 ≈ 35 eval points (更密 → 更平滑的曲线)
EVAL_FREQ_AWSC="${EVAL_FREQ_AWSC:-5000}"
EVAL_FREQ_PLD="${EVAL_FREQ_PLD:-2000}"
EVAL_FREQ_DSRL="${EVAL_FREQ_DSRL:-2000}"
# 兼容旧脚本的全局回退值
EVAL_FREQ="${EVAL_FREQ:-5000}"

# -----------------------------------------------------------------------------
# Multi-Seed Configuration
# -----------------------------------------------------------------------------
# 10 组种子用于重复实验 (可通过环境变量 SEEDS 覆盖，逗号分隔)
if [[ -n "${SEEDS:-}" ]]; then
    IFS=',' read -ra SEED_LIST <<< "$SEEDS"
else
    SEED_LIST=(42 100 200 300 400 500 600 700 800 900)
fi
NUM_SEEDS=${#SEED_LIST[@]}

# -----------------------------------------------------------------------------
# Comparison Scheme — total_timesteps 分配
# -----------------------------------------------------------------------------
COMPARISON_SCHEME="${COMPARISON_SCHEME:-A}"

case "${COMPARISON_SCHEME}" in
    A|a)
        # 方案 A：统一真实机器人步 (推荐正式对比)
        # AWSC 500K robot steps ; PLD/DSRL ~71K chunk decisions ≈ 500K robot steps
        TOTAL_STEPS_AWSC="${TOTAL_STEPS_AWSC:-500000}"
        TOTAL_STEPS_PLD="${TOTAL_STEPS_PLD:-71000}"
        TOTAL_STEPS_DSRL="${TOTAL_STEPS_DSRL:-71000}"
        ;;
    B|b)
        # 方案 B：统一 chunk 决策次数
        TOTAL_STEPS_AWSC="${TOTAL_STEPS_AWSC:-3500000}"
        TOTAL_STEPS_PLD="${TOTAL_STEPS_PLD:-500000}"
        TOTAL_STEPS_DSRL="${TOTAL_STEPS_DSRL:-500000}"
        ;;
    C|c)
        # 方案 C：快速验证（x 轴不对齐）
        TOTAL_STEPS_AWSC="${TOTAL_STEPS_AWSC:-500000}"
        TOTAL_STEPS_PLD="${TOTAL_STEPS_PLD:-500000}"
        TOTAL_STEPS_DSRL="${TOTAL_STEPS_DSRL:-500000}"
        ;;
    *)
        echo "Unknown COMPARISON_SCHEME: ${COMPARISON_SCHEME}. Use A/B/C."
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Experiment Output
# -----------------------------------------------------------------------------
EXP_NAME="${EXP_NAME:-fair_comparison}"
SWEEP_BASE_DIR="${SWEEP_BASE_DIR:-runs/${EXP_NAME}}"

# All "algorithms" in this comparison (maps to subdirectories)
ALL_ALGORITHMS=("awsc" "pld" "dsrl")

# Retry settings (matching sweep_dsrl/pld)
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# -----------------------------------------------------------------------------
# WandB Configuration
# -----------------------------------------------------------------------------
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-RL-Fair-Comparison}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

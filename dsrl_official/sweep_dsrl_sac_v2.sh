#!/bin/bash
#
# DSRL-SAC Sweep v2: Deep dive into UTD, Architecture, and Buffer Size
# Based on v1 findings: UTD=40 significantly outperforms other configs
#
# Usage: ./sweep_dsrl_sac_v2.sh [--gpus "0 1"] [--dry-run]
#
# =============================================================================
# ABLATION STUDY v2
# =============================================================================
#
# 1. UTD RATIO (extended range to find optimal):
#    - utd_60: higher UTD
#    - utd_80: even higher
#    - utd_100: extreme
#
# 2. NETWORK ARCHITECTURE:
#    - arch_small: 2 layers x 512 units
#    - arch_medium: 3 layers x 512 units (default)
#    - arch_large: 3 layers x 1024 units
#    - arch_xlarge: 4 layers x 1024 units
#
# 3. BUFFER SIZE:
#    - buffer_100k: 100,000
#    - buffer_500k: 500,000
#    - buffer_1m: 1,000,000 (default)
#    - buffer_2m: 2,000,000
#
# 4. ACTION MAGNITUDE:
#    - mag_1.0: noise range [-1.0, 1.0]
#    - mag_1.5: noise range [-1.5, 1.5] (default)
#    - mag_2.0: noise range [-2.0, 2.0]
#    - mag_2.5: noise range [-2.5, 2.5]
#
# =============================================================================

set -e

# Default configurations
GPUS=(0 1)
DRY_RUN=false
TOTAL_TIMESTEPS=500000
N_ENVS=100
N_EVAL_ENVS=50
LEARNING_STARTS=1000
WANDB_PROJECT="maniskill_dsrl_sweep_v2"
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"
MAX_EPISODE_STEPS=100
LOG_INTERVAL=1000
SAVE_FREQ=200000
EVAL_FREQ=20000

# Config variants: "category:variant"
# All experiments use DSRL-SAC only
CONFIGS=(
    # === UTD RATIO ABLATION (extended range, skipping utd=40 from v1) ===
    "utd:60"
    "utd:80"
    "utd:100"
    
    # === NETWORK ARCHITECTURE ABLATION ===
    "arch:small"
    "arch:medium"
    "arch:large"
    "arch:xlarge"
    
    # === BUFFER SIZE ABLATION ===
    "buffer:100k"
    "buffer:500k"
    
    # === ACTION MAGNITUDE ABLATION ===
    "mag:1.0"
    "mag:2.0"
    "mag:2.5"
)

SEEDS=(0)
LOG_DIR="/tmp/dsrl_sweep_v2"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpus)
            IFS=' ' read -ra GPUS <<< "$2"
            shift 2
            ;;
        --env)
            ENV_ID="$2"
            shift 2
            ;;
        --timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --n-envs)
            N_ENVS="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --configs)
            IFS=' ' read -ra CONFIGS <<< "$2"
            shift 2
            ;;
        --seeds)
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus \"0 1\"        GPUs to use (default: \"0 1\")"
            echo "  --dry-run            Print commands without executing"
            echo "  --env ENV_ID         ManiSkill3 environment (default: LiftPegUpright-v1)"
            echo "  --timesteps N        Total training timesteps (default: 1000000)"
            echo "  --n-envs N           Number of parallel envs (default: 100)"
            echo "  --wandb-project P    Wandb project name"
            echo "  --seeds \"0 1 2\"     Random seeds (default: \"0\")"
            echo "  --log-dir DIR        Log directory (default: /tmp/dsrl_sweep_v2)"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"

TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "=========================================="
echo "DSRL-SAC Sweep v2"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "N envs (train): $N_ENVS"
echo "N envs (eval): $N_EVAL_ENVS"
echo "Total experiments: $TOTAL_TASKS"
echo "Seeds: ${SEEDS[*]}"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo ""
echo "Ablation Categories:"
echo "  - utd: Extended UTD range (60, 80, 100) - 40 already in v1"
echo "  - arch: Network architecture"
echo "  - buffer: Buffer size"
echo "  - mag: Action magnitude"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build task list
declare -a TASKS
idx=0
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASKS[$idx]="$cfg|$seed"
        ((idx+=1))
    done
done

COMPLETED=0
FAILED=0

run_task() {
    local gpu=$1
    local task=$2
    local task_num=$3
    IFS='|' read -r cfg seed <<< "$task"
    IFS=':' read -r category variant <<< "$cfg"

    # =========================================================================
    # DEFAULT VALUES (fixed from v1 best practices)
    # =========================================================================
    utd=40                    # Best from v1
    num_layers=3
    layer_size=512
    learning_rate=0.0003
    action_magnitude=1.5
    batch_size=256
    buffer_size=1000000
    n_critics=2
    
    # Profile name for logging
    profile="${category}-${variant}"

    # =========================================================================
    # APPLY CONFIGURATION BASED ON CATEGORY AND VARIANT
    # =========================================================================
    case "$category" in
        utd)
            case "$variant" in
                60) utd=60 ;;
                80) utd=80 ;;
                100) utd=100 ;;
                *)
                    echo "Unknown utd variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        arch)
            case "$variant" in
                small)
                    num_layers=2
                    layer_size=512
                    ;;
                medium)
                    num_layers=3
                    layer_size=512
                    ;;
                large)
                    num_layers=3
                    layer_size=1024
                    ;;
                xlarge)
                    num_layers=4
                    layer_size=1024
                    ;;
                *)
                    echo "Unknown arch variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        buffer)
            case "$variant" in
                100k) buffer_size=100000 ;;
                500k) buffer_size=500000 ;;
                1m) buffer_size=1000000 ;;
                *)
                    echo "Unknown buffer variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        mag)
            case "$variant" in
                1.0) action_magnitude=1.0 ;;
                1.5) action_magnitude=1.5 ;;
                2.0) action_magnitude=2.0 ;;
                2.5) action_magnitude=2.5 ;;
                *)
                    echo "Unknown mag variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        *)
            echo "Unknown category: $category"
            return 1
            ;;
    esac

    exp_name="dsrl-sac-${profile}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"

    # Build command
    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_dsrl_sac.py \
        --env-id $ENV_ID \
        --control-mode $CONTROL_MODE \
        --sim-backend $SIM_BACKEND \
        --max-episode-steps $MAX_EPISODE_STEPS \
        --n-envs $N_ENVS \
        --n-eval-envs $N_EVAL_ENVS \
        --seed $seed \
        --total-timesteps $TOTAL_TIMESTEPS \
        --learning-rate $learning_rate \
        --buffer-size $buffer_size \
        --batch-size $batch_size \
        --utd $utd \
        --num-layers $num_layers \
        --layer-size $layer_size \
        --n-critics $n_critics \
        --action-magnitude $action_magnitude \
        --learning-starts $LEARNING_STARTS \
        --log-interval $LOG_INTERVAL \
        --save-freq $SAVE_FREQ \
        --eval-freq $EVAL_FREQ \
        --exp-name $exp_name \
        --track \
        --wandb-project $WANDB_PROJECT \
        --wandb-group dsrl_sac_v2"

    if [ "$DRY_RUN" = true ]; then
        echo "[GPU $gpu] [DRY RUN] $exp_name"
        echo "$CMD" > "$log_file"
        return 0
    fi

    if eval "$CMD" >> "$log_file" 2>&1; then
        echo "[GPU $gpu] ✓ Completed: $exp_name"
        echo "completed" >> "$log_file"
        return 0
    else
        echo "[GPU $gpu] ✗ Failed: $exp_name (see $log_file)"
        echo "failed" >> "$log_file"
        return 1
    fi
}

# Parallel dispatcher
declare -A gpu_queue
for gpu in "${GPUS[@]}"; do
    gpu_queue[$gpu]=""
done

next_task=0
active_pids=()

assign_tasks() {
    for gpu in "${GPUS[@]}"; do
        if [ -n "${gpu_queue[$gpu]}" ]; then
            local pid=${gpu_queue[$gpu]}
            if ! kill -0 "$pid" 2>/dev/null; then
                if wait "$pid" 2>/dev/null; then
                    ((COMPLETED+=1))
                else
                    ((FAILED+=1))
                fi
                gpu_queue[$gpu]=""
            fi
        fi

        if [ -z "${gpu_queue[$gpu]}" ] && [ $next_task -lt $TOTAL_TASKS ]; then
            run_task "$gpu" "${TASKS[$next_task]}" $((next_task + 1)) &
            gpu_queue[$gpu]=$!
            ((next_task+=1))
        fi
    done
}

# Main loop
while [ $next_task -lt $TOTAL_TASKS ] || [ $COMPLETED -lt $TOTAL_TASKS ]; do
    assign_tasks
    sleep 2
done

# Wait for remaining tasks
for gpu in "${GPUS[@]}"; do
    if [ -n "${gpu_queue[$gpu]}" ]; then
        if wait "${gpu_queue[$gpu]}"; then
            ((COMPLETED+=1))
        else
            ((FAILED+=1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "DSRL-SAC Sweep v2 completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

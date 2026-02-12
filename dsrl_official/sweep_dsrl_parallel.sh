#!/bin/bash
#
# Parallel grid search for DSRL design variants
# Compares DSRL-SAC vs DSRL-NA with various hyperparameter configurations
#
# Usage: ./sweep_dsrl_parallel.sh [--gpus "0 1"] [--dry-run] [--env ENV_ID]
#
# =============================================================================
# ABLATION STUDY ORGANIZATION
# =============================================================================
#
# 1. ALGORITHM COMPARISON: DSRL-SAC vs DSRL-NA
#    - sac: Environment wrapper mode (noise as action space)
#    - na: Policy internal sampling with Q^W distillation
#
# 2. UTD RATIO ABLATION: Update-To-Data ratio
#    - utd_10: 10 gradient steps per env step
#    - utd_20: 20 gradient steps per env step (default)
#    - utd_40: 40 gradient steps per env step
#
# 3. NETWORK ARCHITECTURE ABLATION:
#    - arch_small: 2 layers x 1024 units
#    - arch_default: 3 layers x 2048 units
#    - arch_large: 4 layers x 2048 units
#
# 4. LEARNING RATE ABLATION:
#    - lr_low: 1e-4
#    - lr_default: 3e-4
#    - lr_high: 1e-3
#
# 5. ACTION MAGNITUDE ABLATION (SAC only):
#    - mag_1.0: noise range [-1.0, 1.0]
#    - mag_1.5: noise range [-1.5, 1.5] (default)
#    - mag_2.0: noise range [-2.0, 2.0]
#
# 6. BATCH SIZE ABLATION:
#    - batch_128: batch size 128
#    - batch_256: batch size 256 (default)
#    - batch_512: batch size 512
#
# 7. BUFFER SIZE ABLATION:
#    - buffer_100k: 100,000
#    - buffer_500k: 500,000
#    - buffer_1m: 1,000,000 (default)
#
# 8. LAYER NORM ABLATION (NA only):
#    - ln_on: with layer norm (default)
#    - ln_off: without layer norm
#
# =============================================================================

set -e

# Default configurations
GPUS=(0 1)
DRY_RUN=false
TOTAL_TIMESTEPS=100000
N_ENVS=50
N_EVAL_ENVS=50
LEARNING_STARTS=1000
WANDB_PROJECT="maniskill_dsrl_sweep"
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"
MAX_EPISODE_STEPS=100
LOG_INTERVAL=1000
SAVE_FREQ=100000

# Config variants organized by ablation category
# Format: "algorithm:category:variant"
CONFIGS=(
    # === ALGORITHM COMPARISON (baseline) ===
    "sac:baseline:default"
    "na:baseline:default"
    
    # === UTD RATIO ABLATION ===
    "sac:utd:10"
    "sac:utd:20"
    "sac:utd:40"
    "na:utd:10"
    "na:utd:20"
    "na:utd:40"
    
    # === ACTION MAGNITUDE ABLATION (SAC only) ===
    "sac:mag:1.0"
    "sac:mag:2.0"
    
    # === LAYER NORM ABLATION (NA only) ===
    "na:ln:off"
)

SEEDS=(0)
LOG_DIR="/tmp/dsrl_sweep"

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
        --control-mode)
            CONTROL_MODE="$2"
            shift 2
            ;;
        --sim-backend)
            SIM_BACKEND="$2"
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
            echo "  --gpus \"0 1 2\"      GPUs to use (default: \"0 1\")"
            echo "  --dry-run            Print commands without executing"
            echo "  --env ENV_ID         ManiSkill3 environment (default: LiftPegUpright-v1)"
            echo "  --timesteps N        Total training timesteps (default: 500000)"
            echo "  --n-envs N           Number of parallel envs (default: 50)"
            echo "  --wandb-project P    Wandb project name"
            echo "  --seeds \"0 1 2\"     Random seeds (default: \"0 1 2\")"
            echo "  --log-dir DIR        Log directory (default: /tmp/dsrl_sweep)"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"

TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "=========================================="
echo "DSRL Design Sweep"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "N envs: $N_ENVS"
echo "Total experiments: $TOTAL_TASKS"
echo "Seeds: ${SEEDS[*]}"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo ""
echo "Ablation Categories:"
echo "  - baseline: Algorithm comparison (SAC vs NA)"
echo "  - utd: Update-To-Data ratio"
echo "  - arch: Network architecture"
echo "  - lr: Learning rate"
echo "  - mag: Action magnitude (SAC only)"
echo "  - batch: Batch size"
echo "  - buffer: Buffer size"
echo "  - ln: Layer norm (NA only)"
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
    IFS=':' read -r algorithm category variant <<< "$cfg"

    # =========================================================================
    # DEFAULT VALUES
    # =========================================================================
    utd=20
    num_layers=3
    layer_size=512
    learning_rate=0.0003
    action_magnitude=1.5
    batch_size=256
    buffer_size=1000000
    use_layer_norm=true
    n_critics=2
    
    # Profile name for logging
    profile="${algorithm}-${category}-${variant}"

    # =========================================================================
    # APPLY CONFIGURATION BASED ON CATEGORY AND VARIANT
    # =========================================================================
    case "$category" in
        baseline)
            # Use all defaults
            profile="${algorithm}-baseline"
            ;;
        
        utd)
            case "$variant" in
                10) utd=10 ;;
                20) utd=20 ;;
                40) utd=40 ;;
                *)
                    echo "Unknown utd variant: $variant"
                    return 1
                    ;;
            esac
            ;;

        mag)
            # Only applicable to SAC
            if [ "$algorithm" != "sac" ]; then
                echo "Warning: mag ablation only for SAC, skipping"
                return 0
            fi
            case "$variant" in
                1.0) action_magnitude=1.0 ;;
                1.5) action_magnitude=1.5 ;;
                2.0) action_magnitude=2.0 ;;
                *)
                    echo "Unknown mag variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        ln)
            # Only applicable to NA
            if [ "$algorithm" != "na" ]; then
                echo "Warning: ln ablation only for NA, skipping"
                return 0
            fi
            case "$variant" in
                on) use_layer_norm=true ;;
                off) use_layer_norm=false ;;
                *)
                    echo "Unknown ln variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        *)
            echo "Unknown category: $category"
            return 1
            ;;
    esac

    exp_name="dsrl-${profile}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"

    # Build command based on algorithm
    if [ "$algorithm" = "sac" ]; then
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
            --exp-name $exp_name \
            --track \
            --wandb-project $WANDB_PROJECT \
            --wandb-group dsrl_sac"
    else
        # DSRL-NA
        CMD="CUDA_VISIBLE_DEVICES=$gpu python train_dsrl_na.py \
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
            --learning-starts $LEARNING_STARTS \
            --log-interval $LOG_INTERVAL \
            --save-freq $SAVE_FREQ \
            --exp-name $exp_name \
            --track \
            --wandb-project $WANDB_PROJECT \
            --wandb-group dsrl_na"
        
        # Append layer norm flag
        if [ "$use_layer_norm" = true ]; then
            CMD+=" --use-layer-norm"
        else
            CMD+=" --no-use-layer-norm"
        fi
    fi

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
echo "DSRL sweep completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

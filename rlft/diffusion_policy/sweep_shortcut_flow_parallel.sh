#!/bin/bash
#
# Parallel grid search for ShortCut Flow design variants
# Organized by ablation categories for systematic comparison
#
# Usage: ./sweep_shortcut_flow_parallel.sh [--gpus "2 3 4"] [--dry-run] [--env ENV_ID]
#
# =============================================================================
# ABLATION STUDY ORGANIZATION
# =============================================================================
#
# 1. BASELINE: Default configuration
#    - baseline: power2 step size, uniform t, velocity target, adaptive inference
#
# 2. STEP SIZE ABLATION: How to sample step sizes during training
#    - step_uniform: Uniform step size sampling instead of power-of-2
#    - step_fixed_small: Fixed small step size (1/16)
#    - step_fixed_large: Fixed large step size (1/4)
#
# 3. TIME SAMPLING ABLATION: How to sample time t
#    - t_truncated: Directly sample in valid range [t_min, 1-2d]
#    - t_restricted: Restricted t range [0.05, 0.95]
#
# 4. TARGET MODE ABLATION: What to match in shortcut loss
#    - target_endpoint: Match endpoint x1 instead of velocity
#
# 5. TEACHER ABLATION: How to compute shortcut target
#    - teacher_1step: Single-step teacher (faster but less accurate)
#    - teacher_3step: Three-step teacher (more accurate but slower)
#    - teacher_online: Use online network instead of EMA
#
# 6. INFERENCE MODE ABLATION: How to sample at inference
#    - infer_uniform: Uniform steps instead of adaptive
#    - infer_4steps: 4 uniform steps
#    - infer_16steps: 16 uniform steps
#
# 7. LOSS WEIGHT ABLATION: Balance between flow and shortcut loss
#    - weight_flow_only: Only flow loss (no shortcut)
#    - weight_shortcut_heavy: Higher shortcut weight
#    - weight_flow_heavy: Higher flow weight
#
# 8. CONSISTENCY FRACTION ABLATION: How much batch for shortcut loss
#    - cons_k_10: 10% of batch for consistency
#    - cons_k_50: 50% of batch for consistency
#    - cons_k_100: 100% of batch for consistency
#
# =============================================================================

set -e

# Default configurations
GPUS=(2 3 4 5 6 7 8 9)
DRY_RUN=false
TOTAL_ITERS=30000
EVAL_FREQ=2000
LOG_FREQ=100
NUM_EVAL_EPISODES=100
NUM_EVAL_ENVS=50
NUM_DEMOS=1000
SIM_BACKEND="physx_cuda"
WANDB_PROJECT="maniskill_shortcut_grid"
MAX_EPISODE_STEPS=100
ENV_ID="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
OBS_MODE="rgb"
BC_WEIGHT=1.0
SHORTCUT_WEIGHT=1.0

# Config variants organized by ablation category
# Format: "category:config_name"
CONFIGS=(
    # === BASELINE ===
    "baseline:default"
    
    # === STEP SIZE ABLATION ===
    "step:uniform"
    "step:fixed_small"
    "step:fixed_large"
    
    # === TIME SAMPLING ABLATION ===
    "time:truncated"
    "time:restricted"
    
    # === TARGET MODE ABLATION ===
    "target:endpoint"
    
    # === TEACHER ABLATION ===
    "teacher:1step"
    "teacher:3step"
    "teacher:online"
    
    # === INFERENCE MODE ABLATION ===
    "infer:uniform"
    "infer:4steps"
    "infer:16steps"
    
    # === LOSS WEIGHT ABLATION ===
    "weight:flow_only"
    "weight:shortcut_heavy"
    "weight:flow_heavy"
    
    # === CONSISTENCY FRACTION ABLATION ===
    "cons:k10"
    "cons:k50"
    "cons:k100"
)

SEEDS=(0)
LOG_DIR="/tmp/shortcut_flow_grid"

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
        --iters)
            TOTAL_ITERS="$2"
            shift 2
            ;;
        --num-demos)
            NUM_DEMOS="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

DEMO_PATH="$HOME/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5"
mkdir -p "$LOG_DIR"

TOTAL_TASKS=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

echo "=========================================="
echo "ShortCut Flow Design Sweep"
echo "=========================================="
echo "Available GPUs: ${GPUS[*]}"
echo "Environment: $ENV_ID"
echo "Demo: $DEMO_PATH"
echo "Total experiments: $TOTAL_TASKS"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo ""
echo "Ablation Categories:"
echo "  - baseline: Default configuration"
echo "  - step: Step size sampling mode"
echo "  - time: Time sampling strategy"
echo "  - target: Target computation mode"
echo "  - teacher: Teacher network configuration"
echo "  - infer: Inference mode"
echo "  - weight: Loss weight balance"
echo "  - cons: Consistency fraction"
echo "=========================================="
echo ""

cd /home/amax/rlft/diffusion_policy

# Check demo file
if [ ! -f "$DEMO_PATH" ]; then
    echo "Error: Demo file not found: $DEMO_PATH"
    exit 1
fi

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
    # DEFAULT VALUES (baseline configuration)
    # =========================================================================
    # Time sampling
    sc_t_min=0.0
    sc_t_max=1.0
    sc_t_sampling_mode="uniform"
    
    # Step size
    sc_step_size_mode="power2"
    sc_min_step_size=0.0625    # 1/16
    sc_max_step_size=0.5       # 1/2
    sc_fixed_step_size=0.125   # 1/8
    
    # Target computation
    sc_target_mode="velocity"
    sc_teacher_steps=2
    sc_use_ema_teacher=true
    
    # Inference
    sc_inference_mode="adaptive"
    sc_num_inference_steps=8
    
    # Loss weights
    bc_weight=$BC_WEIGHT
    shortcut_weight=$SHORTCUT_WEIGHT
    self_consistency_k=0.25
    
    # Profile name for logging
    profile="${category}-${variant}"

    # =========================================================================
    # APPLY CONFIGURATION BASED ON CATEGORY AND VARIANT
    # =========================================================================
    case "$category" in
        baseline)
            # Use all defaults
            profile="baseline"
            ;;
        
        step)
            case "$variant" in
                uniform)
                    sc_step_size_mode="uniform"
                    ;;
                fixed_small)
                    sc_step_size_mode="fixed"
                    sc_fixed_step_size=0.0625  # 1/16
                    ;;
                fixed_large)
                    sc_step_size_mode="fixed"
                    sc_fixed_step_size=0.25    # 1/4
                    ;;
                *)
                    echo "Unknown step variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        time)
            case "$variant" in
                truncated)
                    sc_t_sampling_mode="truncated"
                    ;;
                restricted)
                    sc_t_min=0.05
                    sc_t_max=0.95
                    ;;
                *)
                    echo "Unknown time variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        target)
            case "$variant" in
                endpoint)
                    sc_target_mode="endpoint"
                    ;;
                *)
                    echo "Unknown target variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        teacher)
            case "$variant" in
                1step)
                    sc_teacher_steps=1
                    ;;
                3step)
                    sc_teacher_steps=3
                    ;;
                online)
                    sc_use_ema_teacher=false
                    ;;
                *)
                    echo "Unknown teacher variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        infer)
            case "$variant" in
                uniform)
                    sc_inference_mode="uniform"
                    ;;
                4steps)
                    sc_inference_mode="uniform"
                    sc_num_inference_steps=4
                    ;;
                16steps)
                    sc_inference_mode="uniform"
                    sc_num_inference_steps=16
                    ;;
                *)
                    echo "Unknown infer variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        weight)
            case "$variant" in
                flow_only)
                    shortcut_weight=0.0
                    ;;
                shortcut_heavy)
                    bc_weight=0.5
                    shortcut_weight=1.0
                    ;;
                flow_heavy)
                    bc_weight=1.0
                    shortcut_weight=0.5
                    ;;
                *)
                    echo "Unknown weight variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        cons)
            case "$variant" in
                k10)
                    self_consistency_k=0.1
                    ;;
                k50)
                    self_consistency_k=0.5
                    ;;
                k100)
                    self_consistency_k=1.0
                    ;;
                *)
                    echo "Unknown cons variant: $variant"
                    return 1
                    ;;
            esac
            ;;
        
        *)
            echo "Unknown category: $category"
            return 1
            ;;
    esac

    exp_name="sc-${ENV_ID}-${profile}-seed${seed}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[GPU $gpu] [$task_num/$TOTAL_TASKS] Starting: $exp_name"

    CMD="CUDA_VISIBLE_DEVICES=$gpu python train_offline_rl.py \
        --algorithm shortcut_flow \
        --env_id $ENV_ID \
        --obs_mode $OBS_MODE \
        --demo_path $DEMO_PATH \
        --control_mode $CONTROL_MODE \
        --sim_backend $SIM_BACKEND \
        --num_demos $NUM_DEMOS \
        --seed $seed \
        --total_iters $TOTAL_ITERS \
        --eval_freq $EVAL_FREQ \
        --log_freq $LOG_FREQ \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --num_eval_envs $NUM_EVAL_ENVS \
        --exp_name $exp_name \
        --track \
        --wandb_project_name $WANDB_PROJECT \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --bc_weight $bc_weight \
        --consistency_weight $shortcut_weight \
        --self_consistency_k $self_consistency_k \
        --sc_t_min $sc_t_min \
        --sc_t_max $sc_t_max \
        --sc_t_sampling_mode $sc_t_sampling_mode \
        --sc_step_size_mode $sc_step_size_mode \
        --sc_min_step_size $sc_min_step_size \
        --sc_max_step_size $sc_max_step_size \
        --sc_fixed_step_size $sc_fixed_step_size \
        --sc_target_mode $sc_target_mode \
        --sc_teacher_steps $sc_teacher_steps \
        --sc_inference_mode $sc_inference_mode \
        --sc_num_inference_steps $sc_num_inference_steps"
    
    # Append boolean flag only when true
    if [ "$sc_use_ema_teacher" = true ]; then CMD+=" --sc_use_ema_teacher"; fi
    if [ "$sc_use_ema_teacher" = false ]; then CMD+=" --no-sc_use_ema_teacher"; fi

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

while [ $next_task -lt $TOTAL_TASKS ] || [ $COMPLETED -lt $TOTAL_TASKS ]; do
    assign_tasks
    sleep 2
done

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
echo "ShortCut Flow sweep completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Logs: $LOG_DIR"
echo "=========================================="

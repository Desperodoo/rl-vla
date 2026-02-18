#!/bin/bash
# =============================================================================
# Sweep Utilities — Helper functions for DSRL hyperparameter sweep
# =============================================================================

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# -----------------------------------------------------------------------------
# Color Output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Experiment Path Helpers
# -----------------------------------------------------------------------------
get_exp_dir() {
    local algorithm=$1
    local config_name=$2
    echo "${SWEEP_BASE_DIR}/${algorithm}/${config_name}"
}

# Find the actual experiment directory (with timestamp suffix)
# train_dsrl.py creates dirs like: runs/{exp_name}__{timestamp}
find_actual_exp_dir() {
    local algorithm=$1
    local config_name=$2
    local base_dir="${SWEEP_BASE_DIR}/${algorithm}"

    local matches
    matches=$(find "$base_dir" -maxdepth 1 -type d -name "${config_name}__*" 2>/dev/null | sort -r | head -1)

    if [[ -n "$matches" ]]; then
        echo "$matches"
    else
        echo "${base_dir}/${config_name}"
    fi
}

get_checkpoint_path() {
    local algorithm=$1
    local config_name=$2
    local exp_dir
    exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")

    if [[ -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]]; then
        echo "${exp_dir}/checkpoints/best_eval_success_once.pt"
    else
        echo "${exp_dir}/checkpoints/final.pt"
    fi
}

# -----------------------------------------------------------------------------
# Experiment Status Detection
# -----------------------------------------------------------------------------
is_experiment_successful() {
    local exp_dir=$1

    # Best checkpoint (train_dsrl.py saves best.pt)
    if [[ -f "${exp_dir}/checkpoints/best.pt" ]]; then
        return 0
    fi

    # Legacy checkpoint name
    if [[ -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]]; then
        return 0
    fi

    # Final checkpoint
    if [[ -f "${exp_dir}/checkpoints/final.pt" ]]; then
        return 0
    fi

    # Check log for completion markers
    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        if grep -qE "Training completed|Saving final checkpoint|100%.*${TOTAL_TIMESTEPS}/${TOTAL_TIMESTEPS}" "$log_file" 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

is_experiment_failed() {
    local exp_dir=$1

    if is_experiment_successful "$exp_dir"; then
        return 1
    fi

    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        # Fatal CUDA / OOM errors
        if grep -qE "(CUDA error|RuntimeError.*CUDA|illegal memory access|Segmentation fault|OutOfMemory|OOM|PhysX Internal CUDA error|out of memory)" "$log_file" 2>/dev/null; then
            return 0
        fi
        # Log exists with checkpoint dir but no final checkpoint → incomplete
        if [[ -d "${exp_dir}/checkpoints" ]]; then
            return 0
        fi
    fi

    if [[ -d "$exp_dir" ]] && [[ -d "${exp_dir}/checkpoints" ]]; then
        return 0
    fi

    return 1
}

# -----------------------------------------------------------------------------
# Parse Metrics from Log File
# -----------------------------------------------------------------------------
parse_metrics_from_log() {
    local log_file=$1
    local metric_name=$2

    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    local value
    # Try wandb summary first
    value=$(grep "wandb:.*${metric_name}" "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}')

    if [[ "$value" == "▁" ]] || [[ -z "$value" ]]; then
        value=""
    fi

    echo "$value"
}

# Parse best success rate from train_dsrl.py log output
# Handles: "Done. Best success rate: 98.00%" → 0.98
# Also handles wandb summary truncation (final/best_success_rate hidden behind "+10 ...")
parse_best_success_rate() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    local value

    # 1. Try wandb summary: final/best_success_rate (may be truncated)
    value=$(grep 'wandb:.*final/best_success_rate' "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [[ -n "$value" ]] && [[ "$value" != "▁" ]]; then
        echo "$value"
        return
    fi

    # 2. Parse "Done. Best success rate: XX.XX%" (always present at end of training)
    value=$(grep -oP 'Done\. Best success rate: \K[\d.]+' "$log_file" 2>/dev/null | tail -1)
    if [[ -n "$value" ]]; then
        # Convert percentage to decimal (98.00 → 0.98)
        value=$(echo "scale=4; $value / 100" | bc 2>/dev/null)
        echo "$value"
        return
    fi

    echo ""
}

# Parse success_at_end from train_dsrl.py log
parse_success_at_end() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    local value

    # 1. Try wandb summary: eval/success_at_end
    value=$(grep 'wandb:.*eval/success_at_end' "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [[ -n "$value" ]] && [[ "$value" != "▁" ]]; then
        echo "$value"
        return
    fi

    # 2. Parse plain text "  success_at_end: X.XXXX"
    value=$(grep -oP '^\s+success_at_end: \K[\d.]+' "$log_file" 2>/dev/null | tail -1)
    if [[ -n "$value" ]]; then
        echo "$value"
        return
    fi

    echo ""
}

# -----------------------------------------------------------------------------
# Checkpoint Validation
# -----------------------------------------------------------------------------
check_checkpoint() {
    local ckpt_path="${CHECKPOINT/#\~/$HOME}"
    if [[ ! -f "$ckpt_path" ]]; then
        log_error "Pretrained checkpoint not found: ${CHECKPOINT}"
        log_info "DSRL requires a pretrained ShortCut Flow checkpoint."
        log_info "Set CHECKPOINT env var or edit config.sh"
        return 1
    fi
    log_info "Pretrained checkpoint: ${CHECKPOINT}"
    return 0
}

# -----------------------------------------------------------------------------
# Run Single Experiment with Retry
# -----------------------------------------------------------------------------
run_experiment() {
    local gpu_id=$1
    local algorithm=$2       # always "dsrl_sac" but kept for directory structure
    local config_name=$3
    local extra_args=$4

    local base_exp_dir
    base_exp_dir=$(get_exp_dir "$algorithm" "$config_name")

    # Skip if already completed
    local actual_exp_dir
    actual_exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
    if [[ -d "$actual_exp_dir" ]] && is_experiment_successful "$actual_exp_dir"; then
        log_info "Skipping ${algorithm}/${config_name} (already completed)"
        return 0
    fi

    mkdir -p "$base_exp_dir"
    local log_file="${base_exp_dir}/train.log"

    # WandB args
    local wandb_args=""
    if [[ "${USE_WANDB}" == "true" ]]; then
        wandb_args="--track --wandb_project ${WANDB_PROJECT}"
        if [[ -n "${WANDB_ENTITY}" ]]; then
            wandb_args+=" --wandb_entity ${WANDB_ENTITY}"
        fi
    fi

    # Build command — python -m rlft.online.train_dsrl
    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.online.train_dsrl"
    cmd+=" --env_id ${ENV_ID}"
    cmd+=" --obs_mode ${OBS_MODE}"
    cmd+=" --control_mode ${CONTROL_MODE}"
    cmd+=" --sim_backend ${SIM_BACKEND}"
    cmd+=" --reward_mode ${REWARD_MODE}"
    cmd+=" --max_episode_steps ${MAX_EPISODE_STEPS}"
    cmd+=" --total_timesteps ${TOTAL_TIMESTEPS}"
    cmd+=" --num_envs ${NUM_ENVS}"
    cmd+=" --num_eval_envs ${NUM_EVAL_ENVS}"
    cmd+=" --eval_freq ${EVAL_FREQ}"
    cmd+=" --save_freq ${SAVE_FREQ}"
    cmd+=" --num_eval_episodes ${NUM_EVAL_EPISODES}"
    cmd+=" --checkpoint ${CHECKPOINT}"
    cmd+=" --pred_horizon ${PRED_HORIZON}"
    cmd+=" --exp_name ${EXP_NAME}/${algorithm}/${config_name}"
    cmd+=" ${wandb_args}"
    cmd+=" ${extra_args}"

    local attempt=0
    local success=false

    while [[ $attempt -lt $MAX_RETRIES ]] && [[ "$success" == "false" ]]; do
        attempt=$((attempt + 1))

        if [[ $attempt -gt 1 ]]; then
            log_warning "Retry ${attempt}/${MAX_RETRIES} for ${algorithm}/${config_name}"
            sleep "$RETRY_DELAY"
            if [[ -f "$log_file" ]]; then
                mv "$log_file" "${log_file}.failed.$((attempt-1))"
            fi
        fi

        log_info "[GPU ${gpu_id}] Running ${algorithm}/${config_name} (attempt ${attempt}/${MAX_RETRIES})"

        eval "$cmd" > "${log_file}" 2>&1
        local exit_code=$?

        actual_exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")

        if [[ $exit_code -eq 0 ]] && is_experiment_successful "$actual_exp_dir"; then
            success=true
            log_success "${algorithm}/${config_name} completed"
        else
            if grep -qE "(CUDA error|RuntimeError.*CUDA|illegal memory access|Segmentation fault|OutOfMemory|OOM|PhysX Internal CUDA error|out of memory)" "${log_file}" 2>/dev/null; then
                log_warning "CUDA error detected, will retry..."
            else
                log_error "${algorithm}/${config_name} failed with non-retryable error"
                break
            fi
        fi
    done

    if [[ "$success" == "false" ]]; then
        log_error "${algorithm}/${config_name} failed after ${attempt} attempts"
        return 1
    fi

    return 0
}

# -----------------------------------------------------------------------------
# Batch Scheduling: Run configs in parallel batches (GPU-exclusive mode)
# -----------------------------------------------------------------------------
run_batch() {
    local algorithm=$1
    shift
    local configs=("$@")

    local total=${#configs[@]}
    local batch_size=${NUM_GPUS}
    local batch_num=0

    log_info "Running ${total} configs for ${algorithm} (batch size: ${batch_size})"
    log_info "Available GPUs: ${AVAILABLE_GPUS[*]}"

    for ((i=0; i<total; i+=batch_size)); do
        batch_num=$((batch_num + 1))
        local batch_end=$((i + batch_size))
        if [[ $batch_end -gt $total ]]; then
            batch_end=$total
        fi

        log_info "=== Batch ${batch_num}: configs $((i+1))-${batch_end} of ${total} ==="

        local pids=()
        local gpu_idx=0

        for ((j=i; j<batch_end; j++)); do
            local config="${configs[$j]}"
            local config_name
            config_name=$(echo "$config" | cut -d':' -f1)
            local extra_args
            extra_args=$(echo "$config" | cut -d':' -f2-)

            if [[ "$config_name" == "$extra_args" ]]; then
                extra_args=""
            fi

            local gpu_id=${AVAILABLE_GPUS[$gpu_idx]}

            run_experiment "$gpu_id" "$algorithm" "$config_name" "$extra_args" &
            pids+=($!)

            gpu_idx=$((gpu_idx + 1))
        done

        log_info "Waiting for batch ${batch_num} to complete..."
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
        log_info "Batch ${batch_num} completed"
    done
}

# -----------------------------------------------------------------------------
# Global Queue Scheduling: Fill GPUs across algorithms
# -----------------------------------------------------------------------------
run_sweep_queue() {
    local items=("$@")
    local total=${#items[@]}
    local idx=0
    local completed=0

    local -a free_gpus=("${AVAILABLE_GPUS[@]}")
    declare -A pid_to_gpu=()

    log_info "Running ${total} configs across algorithms (GPUs: ${#AVAILABLE_GPUS[@]})"

    while [[ $idx -lt $total || ${#pid_to_gpu[@]} -gt 0 ]]; do
        # Launch as many as possible
        while [[ $idx -lt $total && ${#free_gpus[@]} -gt 0 ]]; do
            local item="${items[$idx]}"
            local algorithm="${item%%|*}"
            local rest="${item#*|}"
            local config_name="${rest%%|*}"
            local extra_args="${rest#*|}"

            local gpu_id="${free_gpus[0]}"
            free_gpus=("${free_gpus[@]:1}")

            log_info "[GPU ${gpu_id}] Launching ${algorithm}/${config_name} ($((idx+1))/${total})"
            run_experiment "$gpu_id" "$algorithm" "$config_name" "$extra_args" &
            local pid=$!
            pid_to_gpu[$pid]="$gpu_id"

            idx=$((idx + 1))
        done

        # Collect finished processes and free GPUs
        for pid in "${!pid_to_gpu[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || true
                free_gpus+=("${pid_to_gpu[$pid]}")
                unset 'pid_to_gpu[$pid]'
                completed=$((completed + 1))
            fi
        done

        if [[ $idx -lt $total && ${#free_gpus[@]} -eq 0 ]]; then
            sleep 1
        fi
    done

    log_success "Queue completed! (${completed}/${total})"
}

# -----------------------------------------------------------------------------
# Config Directory Resolution
# -----------------------------------------------------------------------------
get_config_dir() {
    local config_dir="configs"
    if [[ "$CONFIG_VERSION" == "v2" ]]; then
        config_dir="configs_v2"
    elif [[ "$CONFIG_VERSION" == "v3" ]]; then
        config_dir="configs_v3"
    fi
    echo "$config_dir"
}

# -----------------------------------------------------------------------------
# Load Algorithm Configs
# -----------------------------------------------------------------------------
load_algorithm_configs() {
    local algorithm=$1
    local config_dir
    config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: ${config_file}"
        return 1
    fi

    source "$config_file"

    if [[ ${#SWEEP_CONFIGS[@]} -eq 0 ]]; then
        log_error "No configs found in ${config_file}"
        return 1
    fi

    echo "${SWEEP_CONFIGS[@]}"
}

# -----------------------------------------------------------------------------
# Find Failed Experiments
# -----------------------------------------------------------------------------
find_failed_experiments() {
    local algorithm=$1
    local failed=()

    local config_dir
    config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"
    if [[ ! -f "$config_file" ]]; then
        return
    fi

    source "$config_file"

    for config in "${SWEEP_CONFIGS[@]}"; do
        local config_name
        config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir
        exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")

        if is_experiment_failed "$exp_dir"; then
            failed+=("$config")
        fi
    done

    echo "${failed[@]}"
}

# -----------------------------------------------------------------------------
# Analyze Results (basic)
# -----------------------------------------------------------------------------
analyze_algorithm() {
    local algorithm=$1

    local config_dir
    config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"
    if [[ ! -f "$config_file" ]]; then
        return
    fi

    source "$config_file"

    echo "========================================"
    echo "Algorithm: ${algorithm}"
    echo "========================================"

    local total=0
    local success=0
    local failed=0
    local not_started=0

    for config in "${SWEEP_CONFIGS[@]}"; do
        local config_name
        config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir
        exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")

        total=$((total + 1))

        if is_experiment_successful "$exp_dir"; then
            echo -e "  ${GREEN}✓${NC} ${config_name}"
            success=$((success + 1))
        elif is_experiment_failed "$exp_dir"; then
            echo -e "  ${RED}✗${NC} ${config_name} (failed)"
            failed=$((failed + 1))
        else
            echo -e "  ${YELLOW}○${NC} ${config_name} (not started)"
            not_started=$((not_started + 1))
        fi
    done

    echo "----------------------------------------"
    echo "Total: ${total} | Success: ${success} | Failed: ${failed} | Not Started: ${not_started}"
    echo ""
}

# -----------------------------------------------------------------------------
# Analyze Results with Metrics (detailed analysis with sorted results)
# -----------------------------------------------------------------------------
analyze_algorithm_with_metrics() {
    local algorithm=$1
    local config_dir
    config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"

    if [[ ! -f "$config_file" ]]; then
        return
    fi

    source "$config_file"

    echo "========================================"
    echo -e "${CYAN}Algorithm: ${algorithm}${NC}"
    echo "========================================"

    local results=()
    local total=0
    local completed=0
    local failed=0
    local not_started=0

    for config in "${SWEEP_CONFIGS[@]}"; do
        local config_name
        config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir
        exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
        local log_file="${exp_dir}/train.log"

        local base_dir
        base_dir=$(get_exp_dir "$algorithm" "$config_name")
        if [[ ! -f "$log_file" ]] && [[ -f "${base_dir}/train.log" ]]; then
            log_file="${base_dir}/train.log"
        fi

        total=$((total + 1))

        local success_once
        success_once=$(parse_best_success_rate "$log_file")
        local success_end
        success_end=$(parse_success_at_end "$log_file")
        local status="not_started"

        if is_experiment_successful "$exp_dir"; then
            status="success"
            completed=$((completed + 1))
        elif is_experiment_failed "$exp_dir"; then
            status="failed"
            failed=$((failed + 1))
        else
            not_started=$((not_started + 1))
        fi

        if [[ -n "$success_once" ]]; then
            results+=("${success_once}|${success_end}|${config_name}|${status}")
        else
            results+=("0|0|${config_name}|${status}")
        fi
    done

    echo ""
    echo -e "${BLUE}Results sorted by success_once:${NC}"
    echo "----------------------------------------"
    printf "%-40s %12s %12s %10s\n" "Config" "success_once" "success_end" "Status"
    echo "----------------------------------------"

    local sorted_results
    sorted_results=($(printf '%s\n' "${results[@]}" | sort -t'|' -k1 -rn))

    local best_config=""
    local best_score="0"

    for result in "${sorted_results[@]}"; do
        local s_once s_end cfg stat
        s_once=$(echo "$result" | cut -d'|' -f1)
        s_end=$(echo "$result" | cut -d'|' -f2)
        cfg=$(echo "$result" | cut -d'|' -f3)
        stat=$(echo "$result" | cut -d'|' -f4)

        if [[ -z "$best_config" ]] && [[ "$stat" == "success" ]]; then
            best_config="$cfg"
            best_score="$s_once"
        fi

        local color="" status_icon=""
        case $stat in
            success) color="${GREEN}"; status_icon="✓" ;;
            failed)  color="${RED}";   status_icon="✗" ;;
            *)       color="${YELLOW}"; status_icon="○" ;;
        esac

        if [[ "$s_once" != "0" ]]; then
            printf "${color}%-40s %12s %12s %10s${NC}\n" "$cfg" "$s_once" "$s_end" "$status_icon"
        else
            printf "${color}%-40s %12s %12s %10s${NC}\n" "$cfg" "-" "-" "$status_icon"
        fi
    done

    echo "----------------------------------------"
    echo "Total: ${total} | Completed: ${completed} | Failed: ${failed} | Not Started: ${not_started}"

    if [[ -n "$best_config" ]]; then
        echo -e "${GREEN}Best config: ${best_config} (success_once=${best_score})${NC}"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Export Results to JSON
# -----------------------------------------------------------------------------
export_results_json() {
    local output_file=${1:-"sweep_dsrl_results.json"}
    local config_dir
    config_dir=$(get_config_dir)

    echo "{" > "$output_file"
    echo '  "timestamp": "'$(date -Iseconds)'",' >> "$output_file"
    echo '  "config_version": "'${CONFIG_VERSION}'",' >> "$output_file"
    echo '  "env_id": "'${ENV_ID}'",' >> "$output_file"
    echo '  "checkpoint": "'${CHECKPOINT}'",' >> "$output_file"
    echo '  "total_timesteps": '${TOTAL_TIMESTEPS}',' >> "$output_file"
    echo '  "algorithms": {' >> "$output_file"

    local first_algo=true
    local global_best_config=""
    local global_best_score="0"
    local global_best_algo=""

    for algorithm in "${ALL_ALGORITHMS[@]}"; do
        local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"
        if [[ ! -f "$config_file" ]]; then
            continue
        fi

        if [[ "$first_algo" == "false" ]]; then
            echo "," >> "$output_file"
        fi
        first_algo=false

        source "$config_file"

        local algo_best_config=""
        local algo_best_score="0"

        echo -n '    "'${algorithm}'": {' >> "$output_file"
        echo '"configs": [' >> "$output_file"

        local first_config=true

        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name
            config_name=$(echo "$config" | cut -d':' -f1)
            local exp_dir
            exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
            local log_file="${exp_dir}/train.log"

            local base_dir
            base_dir=$(get_exp_dir "$algorithm" "$config_name")
            if [[ ! -f "$log_file" ]] && [[ -f "${base_dir}/train.log" ]]; then
                log_file="${base_dir}/train.log"
            fi

            local status="not_started"
            local success_once=""
            local success_end=""

            if is_experiment_successful "$exp_dir"; then
                status="success"
                success_once=$(parse_best_success_rate "$log_file")
                success_end=$(parse_success_at_end "$log_file")

                if [[ -n "$success_once" ]]; then
                    if (( $(echo "$success_once > $algo_best_score" | bc -l 2>/dev/null || echo 0) )); then
                        algo_best_score="$success_once"
                        algo_best_config="$config_name"
                    fi
                fi
            elif is_experiment_failed "$exp_dir"; then
                status="failed"
            fi

            if [[ "$first_config" == "false" ]]; then
                echo "," >> "$output_file"
            fi
            first_config=false

            echo -n '      {"name": "'${config_name}'", "status": "'${status}'"' >> "$output_file"
            if [[ -n "$success_once" ]]; then
                echo -n ', "success_once": '${success_once}', "success_at_end": '${success_end:-0} >> "$output_file"
            fi
            echo -n '}' >> "$output_file"
        done

        echo "" >> "$output_file"
        echo -n '    ]' >> "$output_file"

        if [[ -n "$algo_best_config" ]]; then
            echo -n ', "best_config": "'${algo_best_config}'", "best_success_once": '${algo_best_score} >> "$output_file"

            if (( $(echo "$algo_best_score > $global_best_score" | bc -l 2>/dev/null || echo 0) )); then
                global_best_score="$algo_best_score"
                global_best_config="$algo_best_config"
                global_best_algo="$algorithm"
            fi
        fi

        echo -n '}' >> "$output_file"
    done

    echo "" >> "$output_file"
    echo "  }," >> "$output_file"

    # Global summary
    echo '  "summary": {' >> "$output_file"
    if [[ -n "$global_best_config" ]]; then
        echo '    "best_algorithm": "'${global_best_algo}'",' >> "$output_file"
        echo '    "best_config": "'${global_best_config}'",' >> "$output_file"
        echo '    "best_success_once": '${global_best_score} >> "$output_file"
    else
        echo '    "best_algorithm": null,' >> "$output_file"
        echo '    "best_config": null,' >> "$output_file"
        echo '    "best_success_once": null' >> "$output_file"
    fi
    echo "  }" >> "$output_file"
    echo "}" >> "$output_file"

    log_success "Results exported to ${output_file}"

    if [[ -n "$global_best_config" ]]; then
        echo ""
        echo "========================================"
        echo -e "${GREEN}GLOBAL BEST${NC}"
        echo "========================================"
        echo "Algorithm: ${global_best_algo}"
        echo "Config: ${global_best_config}"
        echo "success_once: ${global_best_score}"
    fi
}

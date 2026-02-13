#!/bin/bash
# =============================================================================
# Sweep Utilities - Helper functions for RLPD hyperparameter sweep
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

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Experiment Path Helpers
# -----------------------------------------------------------------------------
get_exp_dir() {
    local algorithm=$1
    local config_name=$2
    # Base directory without timestamp
    echo "${SWEEP_BASE_DIR}/${algorithm}/${config_name}"
}

# Find the actual experiment directory (with timestamp suffix)
# train_rlpd.py creates dirs like: runs/{exp_name}__{timestamp}
# For AWSC, cmd_run adds mode suffixes: config_name_pretrain or config_name_scratch
find_actual_exp_dir() {
    local algorithm=$1
    local config_name=$2
    local algo_dir="${SWEEP_BASE_DIR}/${algorithm}"
    
    # 1. Try exact match: config_name__timestamp
    local matches=$(find "$algo_dir" -maxdepth 1 -type d -name "${config_name}__*" 2>/dev/null | sort -r | head -1)
    if [[ -n "$matches" ]]; then
        echo "$matches"
        return
    fi
    
    # 2. Try with mode suffixes (for AWSC): config_name_pretrain__timestamp, config_name_scratch__timestamp
    for suffix in "_pretrain" "_scratch"; do
        local suffixed="${config_name}${suffix}"
        matches=$(find "$algo_dir" -maxdepth 1 -type d -name "${suffixed}__*" 2>/dev/null | sort -r | head -1)
        if [[ -n "$matches" ]]; then
            echo "$matches"
            return
        fi
        # Also check non-timestamped base dir with suffix
        if [[ -d "${algo_dir}/${suffixed}" ]] && [[ -f "${algo_dir}/${suffixed}/train.log" || -d "${algo_dir}/${suffixed}/checkpoints" ]]; then
            echo "${algo_dir}/${suffixed}"
            return
        fi
    done
    
    # 3. Fallback to base directory (for compatibility)
    echo "${algo_dir}/${config_name}"
}

get_checkpoint_path() {
    local algorithm=$1
    local config_name=$2
    local exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
    # Online RL saves best checkpoints
    if [[ -f "${exp_dir}/checkpoints/best.pt" ]]; then
        echo "${exp_dir}/checkpoints/best.pt"
    elif [[ -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]]; then
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
    
    # Check for best checkpoint (online RL saves best.pt)
    if [[ -f "${exp_dir}/checkpoints/best.pt" ]] || [[ -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]]; then
        return 0  # Success
    fi
    
    # Also check for final.pt (for compatibility)
    local final_ckpt="${exp_dir}/checkpoints/final.pt"
    if [[ -f "$final_ckpt" ]]; then
        return 0  # Success
    fi
    
    # 备选：检查日志中的完成标志 (Online RL 特有)
    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        if grep -qE "Training completed|Saving final checkpoint|100%.*${TOTAL_TIMESTEPS}/${TOTAL_TIMESTEPS}" "$log_file" 2>/dev/null; then
            return 0  # Success
        fi
    fi
    
    return 1  # Not successful
}

is_experiment_failed() {
    local exp_dir=$1
    
    # If already successful, not failed
    if is_experiment_successful "$exp_dir"; then
        return 1
    fi
    
    # Check for error indicators
    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        # Check for CUDA errors or other fatal errors
        if grep -qE "(CUDA error|RuntimeError.*CUDA|illegal memory access|Segmentation fault|OutOfMemory|OOM|PhysX Internal CUDA error)" "$log_file" 2>/dev/null; then
            return 0  # Failed
        fi
        # Check if log exists but no checkpoint (incomplete)
        if [[ -d "${exp_dir}/checkpoints" ]]; then
            return 0  # Failed (started but not completed)
        fi
    fi
    
    # Check if experiment directory exists with some files but no final checkpoint
    if [[ -d "$exp_dir" ]] && [[ -d "${exp_dir}/checkpoints" ]]; then
        return 0  # Failed
    fi
    
    return 1  # Not started or unknown
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
    
    # Parse wandb summary line: "wandb: metric_name value"
    # Use word boundary to avoid partial matches (e.g. pretrain_eval/success matching eval/success)
    local value=$(grep -P "wandb:\s+${metric_name}\s" "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}')
    
    # Skip if value is sparkline character or empty
    if [[ "$value" == "▁" ]] || [[ -z "$value" ]]; then
        echo ""
        return
    fi
    
    echo "$value"
}

# -----------------------------------------------------------------------------
# Demo File Check
# -----------------------------------------------------------------------------
check_demo_file() {
    local demo_path="${DEMO_PATH/#\~/$HOME}"  # Expand ~
    if [[ ! -f "$demo_path" ]]; then
        log_error "Demo 文件不存在: ${DEMO_PATH}"
        log_info "请确保 demo 文件存在或设置 DEMO_PATH 环境变量"
        return 1
    fi
    log_info "Demo 文件: ${DEMO_PATH}"
    return 0
}

# -----------------------------------------------------------------------------
# Run Single Experiment with Retry
# -----------------------------------------------------------------------------
run_experiment() {
    local gpu_id=$1
    local algorithm=$2
    local config_name=$3
    local extra_args=$4
    
    local base_exp_dir=$(get_exp_dir "$algorithm" "$config_name")
    
    # Check if already successful (look for existing run with timestamp)
    local actual_exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
    if [[ -d "$actual_exp_dir" ]] && is_experiment_successful "$actual_exp_dir"; then
        log_info "Skipping ${algorithm}/${config_name} (already completed)"
        return 0
    fi
    
    mkdir -p "$base_exp_dir"
    local log_file="${base_exp_dir}/train.log"
    
    # WandB 参数
    local wandb_args=""
    if [[ "${USE_WANDB}" == "true" ]]; then
        wandb_args="--track --wandb_project_name ${WANDB_PROJECT}"
    fi
    
    # Build command - 使用 train_rlpd.py (Online RL)
    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.online.train_rlpd"
    cmd+=" --algorithm ${algorithm}"
    cmd+=" --env_id ${ENV_ID}"
    cmd+=" --obs_mode ${OBS_MODE}"
    cmd+=" --control_mode ${CONTROL_MODE}"
    cmd+=" --sim_backend ${SIM_BACKEND}"
    cmd+=" --total_timesteps ${TOTAL_TIMESTEPS}"
    cmd+=" --num_envs ${NUM_ENVS}"
    cmd+=" --num_eval_envs ${NUM_EVAL_ENVS}"
    cmd+=" --eval_freq ${EVAL_FREQ}"
    cmd+=" --save_freq ${SAVE_FREQ}"
    cmd+=" --demo_path ${DEMO_PATH}"
    cmd+=" --exp_name ${EXP_NAME}/${algorithm}/${config_name}"
    cmd+=" ${wandb_args}"
    cmd+=" ${extra_args}"
    
    local attempt=0
    local success=false
    
    while [[ $attempt -lt $MAX_RETRIES ]] && [[ "$success" == "false" ]]; do
        attempt=$((attempt + 1))
        
        if [[ $attempt -gt 1 ]]; then
            log_warning "Retry ${attempt}/${MAX_RETRIES} for ${algorithm}/${config_name}"
            sleep $RETRY_DELAY
            # 备份失败的日志
            if [[ -f "$log_file" ]]; then
                mv "$log_file" "${log_file}.failed.$((attempt-1))"
            fi
        fi
        
        log_info "[GPU ${gpu_id}] Running ${algorithm}/${config_name} (attempt ${attempt}/${MAX_RETRIES})"
        
        # Run experiment
        eval "$cmd" > "${log_file}" 2>&1
        local exit_code=$?
        
        # After run, find the actual experiment directory (with timestamp)
        actual_exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
        
        if [[ $exit_code -eq 0 ]] && is_experiment_successful "$actual_exp_dir"; then
            success=true
            log_success "${algorithm}/${config_name} completed"
        else
            # Check if it's a CUDA error (retryable)
            # NOTE: Must match actual error patterns, not just any mention of CUDA/PhysX
            # (e.g. "CUDA_VISIBLE_DEVICES=4" or "physx_cuda" are NOT errors)
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
# Batch Scheduling: Run configs in parallel batches
# GPU 独占模式：每个实验独占一个 GPU
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
            local config_name=$(echo "$config" | cut -d':' -f1)
            local extra_args=$(echo "$config" | cut -d':' -f2-)
            
            # Handle case where there's no ':' separator
            if [[ "$config_name" == "$extra_args" ]]; then
                extra_args=""
            fi
            
            local gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
            
            # Run in background
            run_experiment "$gpu_id" "$algorithm" "$config_name" "$extra_args" &
            pids+=($!)
            
            gpu_idx=$((gpu_idx + 1))
        done
        
        # Wait for all processes in this batch (ignore individual failures)
        log_info "Waiting for batch ${batch_num} to complete..."
        for pid in "${pids[@]}"; do
            wait $pid || true  # Don't exit on individual experiment failure
        done
        log_info "Batch ${batch_num} completed"
    done
}

# -----------------------------------------------------------------------------
# Global Queue Scheduling: Fill GPUs across algorithms
# More efficient than per-algorithm batching when algorithms have different
# numbers of configs — GPUs are never idle waiting for a batch to finish.
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
    elif [[ "$CONFIG_VERSION" == "v4" ]]; then
        config_dir="configs_v4"
    fi
    echo "$config_dir"
}

# -----------------------------------------------------------------------------
# Load Algorithm Configs
# -----------------------------------------------------------------------------
load_algorithm_configs() {
    local algorithm=$1
    local config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: ${config_file}"
        return 1
    fi
    
    # Source the config file to get SWEEP_CONFIGS array
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
    
    local config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"
    if [[ ! -f "$config_file" ]]; then
        return
    fi
    
    source "$config_file"
    
    for config in "${SWEEP_CONFIGS[@]}"; do
        local config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
        
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
    
    local config_dir=$(get_config_dir)
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
        local config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
        
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
    local config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"
    
    if [[ ! -f "$config_file" ]]; then
        return
    fi
    
    source "$config_file"
    
    echo "========================================"
    echo -e "${CYAN}Algorithm: ${algorithm}${NC}"
    echo "========================================"
    
    # Collect all results
    local results=()
    local total=0
    local completed=0
    local failed=0
    local not_started=0
    
    for config in "${SWEEP_CONFIGS[@]}"; do
        local config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
        local log_file="${exp_dir}/train.log"
        
        # Also check base dir and mode-suffixed dirs for log file
        local base_dir=$(get_exp_dir "$algorithm" "$config_name")
        if [[ ! -f "$log_file" ]] && [[ -f "${base_dir}/train.log" ]]; then
            log_file="${base_dir}/train.log"
        fi
        # Try mode-suffixed base dirs (for AWSC: baseline -> baseline_pretrain/train.log)
        for _suffix in "_pretrain" "_scratch"; do
            if [[ ! -f "$log_file" ]] && [[ -f "${base_dir}${_suffix}/train.log" ]]; then
                log_file="${base_dir}${_suffix}/train.log"
                break
            fi
        done
        
        total=$((total + 1))
        
        local success_once=$(parse_metrics_from_log "$log_file" "final/best_success_rate")
        local success_end=$(parse_metrics_from_log "$log_file" "eval/success_at_end")
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
        
        # Store result: "success_once|success_end|config_name|status"
        if [[ -n "$success_once" ]]; then
            results+=("${success_once}|${success_end}|${config_name}|${status}")
        else
            results+=("0|0|${config_name}|${status}")
        fi
    done
    
    # Sort by success_once (descending)
    echo ""
    echo -e "${BLUE}Results sorted by success_once:${NC}"
    echo "----------------------------------------"
    printf "%-35s %12s %12s %10s\n" "Config" "success_once" "success_end" "Status"
    echo "----------------------------------------"
    
    # Sort results
    local sorted_results=($(printf '%s\n' "${results[@]}" | sort -t'|' -k1 -rn))
    
    local best_config=""
    local best_score="0"
    
    for result in "${sorted_results[@]}"; do
        local s_once=$(echo "$result" | cut -d'|' -f1)
        local s_end=$(echo "$result" | cut -d'|' -f2)
        local cfg=$(echo "$result" | cut -d'|' -f3)
        local stat=$(echo "$result" | cut -d'|' -f4)
        
        # Track best
        if [[ -z "$best_config" ]] && [[ "$stat" == "success" ]]; then
            best_config="$cfg"
            best_score="$s_once"
        fi
        
        # Color based on status
        local color=""
        local status_icon=""
        case $stat in
            success)
                color="${GREEN}"
                status_icon="✓"
                ;;
            failed)
                color="${RED}"
                status_icon="✗"
                ;;
            *)
                color="${YELLOW}"
                status_icon="○"
                ;;
        esac
        
        # Format output
        if [[ "$s_once" != "0" ]]; then
            printf "${color}%-35s %12s %12s %10s${NC}\n" "$cfg" "$s_once" "$s_end" "$status_icon"
        else
            printf "${color}%-35s %12s %12s %10s${NC}\n" "$cfg" "-" "-" "$status_icon"
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
# Export Results to JSON (with metrics and best config tracking)
# -----------------------------------------------------------------------------
export_results_json() {
    local output_file=${1:-"sweep_rlpd_results.json"}
    local config_dir=$(get_config_dir)
    
    echo "{" > "$output_file"
    echo '  "timestamp": "'$(date -Iseconds)'",' >> "$output_file"
    echo '  "config_version": "'${CONFIG_VERSION}'",' >> "$output_file"
    echo '  "env_id": "'${ENV_ID}'",' >> "$output_file"
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
            local config_name=$(echo "$config" | cut -d':' -f1)
            local exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
            local log_file="${exp_dir}/train.log"
            
            # Also check base dir and mode-suffixed dirs for log file
            local base_dir=$(get_exp_dir "$algorithm" "$config_name")
            if [[ ! -f "$log_file" ]] && [[ -f "${base_dir}/train.log" ]]; then
                log_file="${base_dir}/train.log"
            fi
            # Try mode-suffixed base dirs (for AWSC)
            for _suffix in "_pretrain" "_scratch"; do
                if [[ ! -f "$log_file" ]] && [[ -f "${base_dir}${_suffix}/train.log" ]]; then
                    log_file="${base_dir}${_suffix}/train.log"
                    break
                fi
            done
            
            local status="not_started"
            local success_once=""
            local success_end=""
            
            if is_experiment_successful "$exp_dir"; then
                status="success"
                success_once=$(parse_metrics_from_log "$log_file" "final/best_success_rate")
                success_end=$(parse_metrics_from_log "$log_file" "eval/success_at_end")
                
                # Track best for this algorithm
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
        
        # Add algorithm best
        if [[ -n "$algo_best_config" ]]; then
            echo -n ', "best_config": "'${algo_best_config}'", "best_success_once": '${algo_best_score} >> "$output_file"
            
            # Track global best
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
    
    # Add global summary
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
    
    # Print summary
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

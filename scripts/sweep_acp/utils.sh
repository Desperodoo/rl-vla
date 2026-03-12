#!/bin/bash
# =============================================================================
# ACP Sweep Utilities — Helper functions for ACP reward hyperparameter sweep
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
NC='\033[0m'

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

# -----------------------------------------------------------------------------
# Experiment Status Detection
# -----------------------------------------------------------------------------
is_experiment_successful() {
    local exp_dir=$1

    if [[ -f "${exp_dir}/checkpoints/best.pt" ]] || \
       [[ -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]] || \
       [[ -f "${exp_dir}/checkpoints/final.pt" ]]; then
        return 0
    fi

    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        if grep -qE "Training completed|Saving final checkpoint|Done\. Best success rate" "$log_file" 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

is_experiment_running() {
    local exp_dir=$1
    local algorithm=$2
    local config_name=$3

    # Check if a training process for this experiment is still alive
    if pgrep -f "exp_name.*${algorithm}/${config_name}" > /dev/null 2>&1; then
        return 0
    fi
    # Also check via exp_name pattern used in sweep
    if pgrep -f "exp_name acp_sweep/${algorithm}/${config_name}" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

is_experiment_failed() {
    local exp_dir=$1
    local algorithm=${2:-}
    local config_name=${3:-}

    if is_experiment_successful "$exp_dir"; then
        return 1
    fi

    # If the process is still running, it's not failed
    if [[ -n "$algorithm" ]] && [[ -n "$config_name" ]]; then
        if is_experiment_running "$exp_dir" "$algorithm" "$config_name"; then
            return 1
        fi
    fi

    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        if grep -qE "(CUDA error|RuntimeError.*CUDA|illegal memory access|Segmentation fault|OutOfMemory|OOM|PhysX Internal CUDA error|out of memory)" "$log_file" 2>/dev/null; then
            return 0
        fi
        # Only treat as failed if checkpoints dir exists, is empty, AND no process running
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
parse_best_success_rate() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    # 1. wandb summary
    local value
    value=$(grep 'wandb:.*final/best_success_rate' "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [[ -n "$value" ]] && [[ "$value" != "▁" ]]; then
        echo "$value"
        return
    fi

    # 2. "Done. Best success rate: XX.XX%"
    value=$(grep -oP 'Done\. Best success rate: \K[\d.]+' "$log_file" 2>/dev/null | tail -1)
    if [[ -n "$value" ]]; then
        value=$(echo "scale=4; $value / 100" | bc 2>/dev/null)
        echo "$value"
        return
    fi

    # 3. best eval success_once from any log line
    value=$(grep -oP 'eval/success_once.*?[\d.]+' "$log_file" 2>/dev/null | grep -oP '[\d.]+$' | sort -rn | head -1)
    if [[ -n "$value" ]]; then
        echo "$value"
        return
    fi

    echo ""
}

parse_success_at_end() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    # 1. wandb summary
    local value
    value=$(grep 'wandb:.*eval/success_at_end' "$log_file" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [[ -n "$value" ]] && [[ "$value" != "▁" ]]; then
        echo "$value"
        return
    fi

    # 2. Plain text
    value=$(grep -oP '^\s+success_at_end: \K[\d.]+' "$log_file" 2>/dev/null | tail -1)
    if [[ -n "$value" ]]; then
        echo "$value"
        return
    fi

    echo ""
}

parse_best_success_at_end() {
    local log_file=$1

    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    # Find the maximum success_at_end across all eval steps
    local value
    value=$(grep -oP 'success_at_end.*?[\d.]+' "$log_file" 2>/dev/null | grep -oP '[\d.]+$' | sort -rn | head -1)
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
        return 1
    fi
    log_info "Pretrained checkpoint: ${CHECKPOINT}"
    return 0
}

check_acp_checkpoint() {
    local acp_path=$1
    if [[ ! -f "$acp_path" ]]; then
        log_error "ACP checkpoint not found: ${acp_path}"
        return 1
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Run Single Experiment with Retry (2-GPU: train + ACP model)
# -----------------------------------------------------------------------------
run_experiment() {
    local gpu_pair=$1      # "train_gpu,acp_gpu"
    local algorithm=$2
    local config_name=$3
    local extra_args=$4

    local train_gpu="${gpu_pair%%,*}"
    local acp_gpu="${gpu_pair##*,}"

    local base_exp_dir
    base_exp_dir=$(get_exp_dir "$algorithm" "$config_name")

    local actual_exp_dir
    actual_exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
    if [[ -d "$actual_exp_dir" ]] && is_experiment_successful "$actual_exp_dir"; then
        log_info "Skipping ${algorithm}/${config_name} (already completed)"
        return 0
    fi

    mkdir -p "$base_exp_dir"
    local log_file="${base_exp_dir}/train.log"

    # Build command based on algorithm family
    local cmd=""

    case "${algorithm%%_acp*}" in
        awsc)            # train_rlpd uses --wandb_project_name (not --wandb_project)
            local wandb_args=""
            if [[ "${USE_WANDB}" == "true" ]]; then
                wandb_args="--track --wandb_project_name ${WANDB_PROJECT}"
                if [[ -n "${WANDB_ENTITY}" ]]; then
                    wandb_args+=" --wandb_entity ${WANDB_ENTITY}"
                fi
            fi
            cmd="CUDA_VISIBLE_DEVICES=${train_gpu},${acp_gpu} python -m rlft.online.train_rlpd"
            cmd+=" --algorithm awsc"
            cmd+=" --pretrain_path ${CHECKPOINT}"
            cmd+=" --pred_horizon ${PRED_HORIZON}"
            cmd+=" --load_pretrain_critic"
            cmd+=" --reward_mode acp"
            cmd+=" --acp_checkpoint ${ACP_CKPT}"
            cmd+=" --acp_device cuda:1"
            cmd+=" --acp_reward_scale ${ACP_REWARD_SCALE}"
            cmd+=" --env_id ${ENV_ID}"
            cmd+=" --num_envs ${NUM_ENVS}"
            cmd+=" --num_eval_envs ${NUM_EVAL_ENVS}"
            cmd+=" --total_timesteps ${TOTAL_STEPS_AWSC}"
            cmd+=" --max_episode_steps ${MAX_EPISODE_STEPS}"
            cmd+=" --demo_path ${DEMO_PATH}"
            cmd+=" --online_ratio ${AWSC_ONLINE_RATIO}"
            cmd+=" --utd_ratio ${AWSC_UTD_RATIO}"
            cmd+=" --lr_actor ${AWSC_LR_ACTOR}"
            cmd+=" --lr_critic ${AWSC_LR_CRITIC}"
            cmd+=" --num_qs ${AWSC_NUM_QS}"
            cmd+=" --num_min_qs ${AWSC_NUM_MIN_QS}"
            cmd+=" --awsc_beta ${AWSC_BETA}"
            cmd+=" --awsc_bc_weight ${AWSC_BC_WEIGHT}"
            cmd+=" --awsc_advantage_mode ${AWSC_ADVANTAGE_MODE}"
            cmd+=" --awsc_num_inference_steps ${AWSC_NUM_INFERENCE_STEPS}"
            cmd+=" --seed ${SEED}"
            cmd+=" --exp_name ${EXP_NAME}/${algorithm}/${config_name}"
            cmd+=" ${wandb_args}"
            cmd+=" ${extra_args}"
            ;;
        *)
            log_error "Unknown algorithm family: ${algorithm}"
            return 1
            ;;
    esac

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

        log_info "[GPU ${gpu_pair}] Running ${algorithm}/${config_name} (attempt ${attempt}/${MAX_RETRIES})"

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
# Global Queue Scheduling (GPU-pair mode)
# Each experiment uses a GPU pair (train + ACP inference).
# -----------------------------------------------------------------------------
run_sweep_queue() {
    local items=("$@")
    local total=${#items[@]}
    local idx=0
    local completed=0

    local -a free_pairs=("${GPU_PAIRS[@]}")
    declare -A pid_to_pair=()

    log_info "Running ${total} configs (GPU pairs: ${#GPU_PAIRS[@]}, 2 GPUs each)"

    while [[ $idx -lt $total || ${#pid_to_pair[@]} -gt 0 ]]; do
        # Launch as many as possible
        while [[ $idx -lt $total && ${#free_pairs[@]} -gt 0 ]]; do
            local item="${items[$idx]}"
            local algorithm="${item%%|*}"
            local rest="${item#*|}"
            local config_name="${rest%%|*}"
            local extra_args="${rest#*|}"

            local gpu_pair="${free_pairs[0]}"
            free_pairs=("${free_pairs[@]:1}")

            log_info "[GPU ${gpu_pair}] Launching ${algorithm}/${config_name} ($((idx+1))/${total})"
            run_experiment "$gpu_pair" "$algorithm" "$config_name" "$extra_args" &
            local pid=$!
            pid_to_pair[$pid]="$gpu_pair"

            idx=$((idx + 1))
        done

        # Collect finished processes
        for pid in "${!pid_to_pair[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || true
                free_pairs+=("${pid_to_pair[$pid]}")
                unset 'pid_to_pair[$pid]'
                completed=$((completed + 1))
            fi
        done

        if [[ $idx -lt $total && ${#free_pairs[@]} -eq 0 ]]; then
            sleep 1
        fi
    done

    log_success "Queue completed! (${completed}/${total})"
}

# -----------------------------------------------------------------------------
# Config Directory Resolution
# -----------------------------------------------------------------------------
get_config_dir() {
    echo "configs"
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

        if is_experiment_failed "$exp_dir" "$algorithm" "$config_name"; then
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
    local running=0
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
        elif is_experiment_running "$exp_dir" "$algorithm" "$config_name"; then
            echo -e "  ${CYAN}▸${NC} ${config_name} (running)"
            running=$((running + 1))
        elif is_experiment_failed "$exp_dir" "$algorithm" "$config_name"; then
            echo -e "  ${RED}✗${NC} ${config_name} (failed)"
            failed=$((failed + 1))
        else
            echo -e "  ${YELLOW}○${NC} ${config_name} (not started)"
            not_started=$((not_started + 1))
        fi
    done

    echo "----------------------------------------"
    echo "Total: ${total} | Success: ${success} | Running: ${running} | Failed: ${failed} | Not Started: ${not_started}"
    echo ""
}

# -----------------------------------------------------------------------------
# Analyze Results with Metrics
# Focus on BOTH success_once AND success_at_end (the core ACP evaluation axes)
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
    local running=0
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
        local best_success_end
        best_success_end=$(parse_best_success_at_end "$log_file")
        local status="not_started"

        if is_experiment_successful "$exp_dir"; then
            status="success"
            completed=$((completed + 1))
        elif is_experiment_running "$exp_dir" "$algorithm" "$config_name"; then
            status="running"
            running=$((running + 1))
        elif is_experiment_failed "$exp_dir" "$algorithm" "$config_name"; then
            status="failed"
            failed=$((failed + 1))
        else
            not_started=$((not_started + 1))
        fi

        # Store: "best_success_end|success_once|final_success_end|config_name|status"
        local sort_key="${best_success_end:-0}"
        results+=("${sort_key}|${success_once:-0}|${success_end:-0}|${config_name}|${status}")
    done

    echo ""
    echo -e "${BLUE}Results sorted by best success_at_end (core metric):${NC}"
    echo "----------------------------------------------------------------------"
    printf "%-35s %12s %12s %12s %8s\n" "Config" "best_s_end" "best_s_once" "final_s_end" "Status"
    echo "----------------------------------------------------------------------"

    local sorted_results
    sorted_results=($(printf '%s\n' "${results[@]}" | sort -t'|' -k1 -rn))

    local best_config=""
    local best_score="0"

    for result in "${sorted_results[@]}"; do
        local b_end s_once f_end cfg stat
        b_end=$(echo "$result" | cut -d'|' -f1)
        s_once=$(echo "$result" | cut -d'|' -f2)
        f_end=$(echo "$result" | cut -d'|' -f3)
        cfg=$(echo "$result" | cut -d'|' -f4)
        stat=$(echo "$result" | cut -d'|' -f5)

        if [[ -z "$best_config" ]] && [[ "$stat" == "success" ]]; then
            best_config="$cfg"
            best_score="$b_end"
        fi

        local color="" status_icon=""
        case $stat in
            success) color="${GREEN}"; status_icon="✓" ;;
            running) color="${CYAN}";  status_icon="▸" ;;
            failed)  color="${RED}";   status_icon="✗" ;;
            *)       color="${YELLOW}"; status_icon="○" ;;
        esac

        if [[ "$b_end" != "0" ]] || [[ "$s_once" != "0" ]]; then
            printf "${color}%-35s %12s %12s %12s %8s${NC}\n" "$cfg" "$b_end" "$s_once" "$f_end" "$status_icon"
        else
            printf "${color}%-35s %12s %12s %12s %8s${NC}\n" "$cfg" "-" "-" "-" "$status_icon"
        fi
    done

    echo "----------------------------------------------------------------------"
    echo "Total: ${total} | Completed: ${completed} | Running: ${running} | Failed: ${failed} | Not Started: ${not_started}"

    if [[ -n "$best_config" ]]; then
        echo -e "${GREEN}Best config (by success_at_end): ${best_config} (best_s_end=${best_score})${NC}"
    fi
    echo ""
}

# -----------------------------------------------------------------------------
# Export Results to JSON
# -----------------------------------------------------------------------------
export_results_json() {
    local output_file=${1:-"${SWEEP_BASE_DIR}/acp_sweep_results.json"}
    local config_dir
    config_dir=$(get_config_dir)

    mkdir -p "$(dirname "$output_file")"

    echo "{" > "$output_file"
    echo '  "timestamp": "'$(date -Iseconds)'",' >> "$output_file"
    echo '  "env_id": "'${ENV_ID}'",' >> "$output_file"
    echo '  "checkpoint": "'${CHECKPOINT}'",' >> "$output_file"
    echo '  "acp_checkpoint": "'${ACP_CKPT}'",' >> "$output_file"
    echo '  "seed": '${SEED}',' >> "$output_file"
    echo '  "algorithms": {' >> "$output_file"

    local first_algo=true
    local global_best_config=""
    local global_best_end="0"
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
        local algo_best_end="0"

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
            local best_end=""

            if is_experiment_successful "$exp_dir"; then
                status="success"
                success_once=$(parse_best_success_rate "$log_file")
                success_end=$(parse_success_at_end "$log_file")
                best_end=$(parse_best_success_at_end "$log_file")

                if [[ -n "$best_end" ]]; then
                    if (( $(echo "$best_end > $algo_best_end" | bc -l 2>/dev/null || echo 0) )); then
                        algo_best_end="$best_end"
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
                echo -n ', "success_once": '${success_once}', "success_at_end": '${success_end:-0}', "best_success_at_end": '${best_end:-0} >> "$output_file"
            fi
            echo -n '}' >> "$output_file"
        done

        echo "" >> "$output_file"
        echo -n '    ]' >> "$output_file"

        if [[ -n "$algo_best_config" ]]; then
            echo -n ', "best_config": "'${algo_best_config}'", "best_success_at_end": '${algo_best_end} >> "$output_file"

            if (( $(echo "$algo_best_end > $global_best_end" | bc -l 2>/dev/null || echo 0) )); then
                global_best_end="$algo_best_end"
                global_best_config="$algo_best_config"
                global_best_algo="$algorithm"
            fi
        fi

        echo -n '}' >> "$output_file"
    done

    echo "" >> "$output_file"
    echo "  }," >> "$output_file"

    # Sim baselines for comparison
    echo '  "sim_baselines": {' >> "$output_file"
    echo '    "awsc": {"best_success_at_end": 0.72, "best_success_once": 0.92},' >> "$output_file"
    echo '    "pld":  {"best_success_at_end": 0.86, "best_success_once": 1.00},' >> "$output_file"
    echo '    "dsrl": {"best_success_at_end": 0.60, "best_success_once": 0.98}' >> "$output_file"
    echo '  },' >> "$output_file"

    echo '  "summary": {' >> "$output_file"
    if [[ -n "$global_best_config" ]]; then
        echo '    "best_algorithm": "'${global_best_algo}'",' >> "$output_file"
        echo '    "best_config": "'${global_best_config}'",' >> "$output_file"
        echo '    "best_success_at_end": '${global_best_end} >> "$output_file"
    else
        echo '    "best_algorithm": null,' >> "$output_file"
        echo '    "best_config": null,' >> "$output_file"
        echo '    "best_success_at_end": null' >> "$output_file"
    fi
    echo "  }" >> "$output_file"
    echo "}" >> "$output_file"

    log_success "Results exported to ${output_file}"

    if [[ -n "$global_best_config" ]]; then
        echo ""
        echo "========================================"
        echo -e "${GREEN}GLOBAL BEST (by success_at_end)${NC}"
        echo "========================================"
        echo "Algorithm: ${global_best_algo}"
        echo "Config: ${global_best_config}"
        echo "best_success_at_end: ${global_best_end}"
    fi
}

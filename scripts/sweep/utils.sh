#!/bin/bash
# =============================================================================
# Sweep Utilities - Helper functions for hyperparameter sweep
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
find_actual_exp_dir() {
    local algorithm=$1
    local config_name=$2
    local base_dir="${SWEEP_BASE_DIR}/${algorithm}"
    
    # Look for directories matching pattern: config_name__timestamp
    local matches=$(find "$base_dir" -maxdepth 1 -type d -name "${config_name}__*" 2>/dev/null | sort -r | head -1)
    
    if [[ -n "$matches" ]]; then
        echo "$matches"
    else
        # Fallback to base directory (for compatibility)
        echo "${base_dir}/${config_name}"
    fi
}

get_checkpoint_path() {
    local algorithm=$1
    local config_name=$2
    local exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
    echo "${exp_dir}/checkpoints/best_eval_success_once.pt"
}

# -----------------------------------------------------------------------------
# Experiment Status Detection
# -----------------------------------------------------------------------------
is_experiment_successful() {
    local exp_dir=$1
    
    # Check for any best checkpoint (training saves best_eval_success_once.pt)
    if [[ -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]]; then
        return 0  # Success
    fi
    # Also check for final.pt (for compatibility)
    if [[ -f "${exp_dir}/checkpoints/final.pt" ]]; then
        return 0  # Success
    fi
    return 1  # Not successful
}

is_experiment_failed() {
    local exp_dir=$1
    
    # If already successful, not failed
    if is_experiment_successful "$exp_dir"; then
        return 1
    fi
    
    # Check for log file with "Training completed successfully"
    local log_file="${exp_dir}/train.log"
    if [[ -f "$log_file" ]]; then
        if grep -q "Training completed successfully" "$log_file" 2>/dev/null; then
            return 1  # Actually succeeded
        fi
        # Check for CUDA errors or other fatal errors
        if grep -qE "(CUDA error|RuntimeError|OutOfMemoryError|Segmentation fault|Killed)" "$log_file" 2>/dev/null; then
            return 0  # Failed
        fi
    fi
    
    # Check if experiment directory exists with checkpoints folder but no final checkpoint
    if [[ -d "$exp_dir" ]] && [[ -d "${exp_dir}/checkpoints" ]]; then
        # Has checkpoints folder but no success checkpoint
        if [[ ! -f "${exp_dir}/checkpoints/best_eval_success_once.pt" ]] && \
           [[ ! -f "${exp_dir}/checkpoints/final.pt" ]]; then
            return 0  # Failed (started but not completed)
        fi
    fi
    
    return 1  # Not started or unknown
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
    
    # Create a temporary log directory for this run
    mkdir -p "$base_exp_dir"
    local log_file="${base_exp_dir}/train.log"
    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_maniskill"
    cmd+=" --algorithm ${algorithm}"
    cmd+=" --env_id ${ENV_ID}"
    cmd+=" --obs_mode ${OBS_MODE}"
    cmd+=" --control_mode ${CONTROL_MODE}"
    cmd+=" --sim_backend ${SIM_BACKEND}"
    cmd+=" --total_iters ${TOTAL_ITERS}"
    cmd+=" --demo_path ${DEMO_PATH}"
    cmd+=" --exp_name ${EXP_NAME}/${algorithm}/${config_name}"
    cmd+=" --wandb_project_name ${EXP_NAME}"
    cmd+=" ${extra_args}"
    
    local attempt=0
    local success=false
    
    while [[ $attempt -lt $MAX_RETRIES ]] && [[ "$success" == "false" ]]; do
        attempt=$((attempt + 1))
        
        if [[ $attempt -gt 1 ]]; then
            log_warning "Retry ${attempt}/${MAX_RETRIES} for ${algorithm}/${config_name}"
            sleep $RETRY_DELAY
        fi
        
        log_info "[GPU ${gpu_id}] Running ${algorithm}/${config_name} (attempt ${attempt}/${MAX_RETRIES})"
        
        # Run experiment
        eval "$cmd" > "${log_file}" 2>&1
        local exit_code=$?
        
        # After run, find the actual experiment directory (with timestamp)
        actual_exp_dir=$(find_actual_exp_dir "$algorithm" "$config_name")
        
        # Check success by looking at log file content or checkpoint existence
        if [[ $exit_code -eq 0 ]]; then
            if is_experiment_successful "$actual_exp_dir"; then
                success=true
                log_success "${algorithm}/${config_name} completed"
            elif grep -q "Training completed successfully" "${log_file}" 2>/dev/null; then
                success=true
                log_success "${algorithm}/${config_name} completed"
            fi
        fi
        
        if [[ "$success" == "false" ]]; then
            # Check if it's a CUDA error (retryable)
            if grep -qE "(CUDA error|cuDNN|cublas)" "${log_file}" 2>/dev/null; then
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
# One config per GPU, wait for batch to complete before next batch
# -----------------------------------------------------------------------------
run_batch() {
    local algorithm=$1
    shift
    local configs=("$@")
    
    local total=${#configs[@]}
    local batch_size=${NUM_GPUS}
    local batch_num=0
    
    log_info "Running ${total} configs for ${algorithm} (batch size: ${batch_size})"
    
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
# Load Algorithm Configs
# -----------------------------------------------------------------------------
load_algorithm_configs() {
    local algorithm=$1
    local config_dir="configs"
    
    # Support config version (v1, v2, etc.)
    if [[ "$CONFIG_VERSION" == "v2" ]]; then
        config_dir="configs_v2"
    fi
    
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
    
    local configs=($(load_algorithm_configs "$algorithm"))
    
    for config in "${configs[@]}"; do
        local config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir=$(get_exp_dir "$algorithm" "$config_name")
        
        if is_experiment_failed "$exp_dir"; then
            failed+=("$config")
        fi
    done
    
    echo "${failed[@]}"
}

# -----------------------------------------------------------------------------
# Analyze Results
# -----------------------------------------------------------------------------
analyze_algorithm() {
    local algorithm=$1
    local configs=($(load_algorithm_configs "$algorithm"))
    
    echo "========================================"
    echo "Algorithm: ${algorithm}"
    echo "========================================"
    
    local total=0
    local success=0
    local failed=0
    local not_started=0
    
    for config in "${configs[@]}"; do
        local config_name=$(echo "$config" | cut -d':' -f1)
        local exp_dir=$(get_exp_dir "$algorithm" "$config_name")
        
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
# Export Best Results to JSON
# -----------------------------------------------------------------------------
export_results_json() {
    local output_file=${1:-"sweep_results.json"}
    
    echo "{" > "$output_file"
    echo '  "timestamp": "'$(date -Iseconds)'",' >> "$output_file"
    echo '  "algorithms": {' >> "$output_file"
    
    local first_algo=true
    for algorithm in "${ALL_ALGORITHMS[@]}"; do
        local config_file="${SCRIPT_DIR}/configs/${algorithm}.sh"
        if [[ ! -f "$config_file" ]]; then
            continue
        fi
        
        if [[ "$first_algo" == "false" ]]; then
            echo "," >> "$output_file"
        fi
        first_algo=false
        
        echo -n '    "'${algorithm}'": {' >> "$output_file"
        echo '"configs": [' >> "$output_file"
        
        source "$config_file"
        local first_config=true
        
        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name=$(echo "$config" | cut -d':' -f1)
            local exp_dir=$(get_exp_dir "$algorithm" "$config_name")
            local status="not_started"
            
            if is_experiment_successful "$exp_dir"; then
                status="success"
            elif is_experiment_failed "$exp_dir"; then
                status="failed"
            fi
            
            if [[ "$first_config" == "false" ]]; then
                echo "," >> "$output_file"
            fi
            first_config=false
            
            echo -n '      {"name": "'${config_name}'", "status": "'${status}'"}' >> "$output_file"
        done
        
        echo "" >> "$output_file"
        echo -n '    ]}' >> "$output_file"
    done
    
    echo "" >> "$output_file"
    echo "  }" >> "$output_file"
    echo "}" >> "$output_file"
    
    log_success "Results exported to ${output_file}"
}

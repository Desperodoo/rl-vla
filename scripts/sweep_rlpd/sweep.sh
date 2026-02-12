#!/bin/bash
# =============================================================================
# Sweep - Unified entry point for RLPD hyperparameter sweep
# =============================================================================
#
# Usage:
#   ./sweep.sh run [--algorithm ALGO] [--mode MODE] [--pretrain-path PATH]
#   ./sweep.sh retry [--algorithm ALGO]
#   ./sweep.sh status [--algorithm ALGO]
#   ./sweep.sh analyze [--algorithm ALGO] [--export FILE]
#   ./sweep.sh report [--algorithm ALGO]
#
# Examples:
#   ./sweep.sh run                                  # Run all algorithms
#   ./sweep.sh run --algorithm sac                  # Run only SAC
#   ./sweep.sh run --algorithm awsc --config-version v2  # Run AWSC v2 (auto pretrain mode)
#   ./sweep.sh run --algorithm awsc --mode scratch  # Run AWSC from scratch
#   ./sweep.sh run --algorithm awsc --mode pretrain --pretrain-path runs/shortcut_flow/best.pt
#   ./sweep.sh retry                                # Retry all failed experiments
#   ./sweep.sh retry --algorithm sac                # Retry failed SAC experiments
#   ./sweep.sh status                               # Show status of all experiments
#   ./sweep.sh analyze --algorithm awsc             # Analyze with metrics for AWSC
#   ./sweep.sh analyze --export results.json        # Export results to JSON
#   ./sweep.sh report                               # Generate Python analysis report
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pre-parse --config-version from all args
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--config-version" ]]; then
        export CONFIG_VERSION="${ARGS[$((i+1))]}"
    fi
done

source "${SCRIPT_DIR}/utils.sh"

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  run       Run sweep experiments"
    echo "  retry     Retry failed experiments"
    echo "  status    Show experiment status"
    echo "  analyze   Analyze results with metrics"
    echo "  report    Generate Python analysis report"
    echo ""
    echo "Global Options:"
    echo "  --config-version V  Use config version (v1=wave1, v2=wave2, v3=wave3, v4=wave4)"
    echo "                      v2/v3/v4 auto-sets AWSC mode to 'pretrain' (fine-tuning)"
    echo ""
    echo "Options for 'run':"
    echo "  --algorithm ALGO    Run only specified algorithm (sac, awsc)"
    echo "  --mode MODE         AWSC mode: scratch, pretrain, both"
    echo "                      (default: scratch for v1, pretrain for v2)"
    echo "  --pretrain-path P   Pretrained checkpoint path (default: Wave 3 best AWSC model)"
    echo "  --dry-run           Show what would be run without executing"
    echo ""
    echo "Options for 'retry':"
    echo "  --algorithm ALGO    Retry only specified algorithm"
    echo "  --dry-run           Show what would be retried without executing"
    echo ""
    echo "Options for 'status':"
    echo "  --algorithm ALGO    Show status for specified algorithm only"
    echo ""
    echo "Options for 'analyze':"
    echo "  --algorithm ALGO    Analyze only specified algorithm"
    echo "  --export FILE       Export results to JSON file"
    echo ""
    echo "Options for 'report':"
    echo "  --algorithm ALGO    Report only specified algorithm"
    echo ""
    echo "Environment Variables:"
    echo "  ENV_ID              Task ID (default: LiftPegUpright-v1)"
    echo "  TOTAL_TIMESTEPS     Total training timesteps (default: 500000)"
    echo "  DEMO_PATH           Demo file path"
    echo "  CUDA_VISIBLE_DEVICES  Available GPUs"
    echo "  USE_WANDB           Use WandB logging (default: false)"
    echo ""
    exit 1
}

# -----------------------------------------------------------------------------
# Run Command
# -----------------------------------------------------------------------------
cmd_run() {
    local algorithm=""
    local awsc_mode=""  # Empty means auto-detect based on config-version
    local pretrain_path=""
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm)
                algorithm=$2
                shift 2
                ;;
            --mode)
                awsc_mode=$2
                if [[ ! "$awsc_mode" =~ ^(scratch|pretrain|both)$ ]]; then
                    log_error "Invalid mode: $awsc_mode (valid: scratch, pretrain, both)"
                    exit 1
                fi
                shift 2
                ;;
            --pretrain-path)
                pretrain_path=$2
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --config-version)
                # Already handled globally, skip
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Auto-detect AWSC mode based on config-version if not explicitly set
    if [[ -z "$awsc_mode" ]]; then
        if [[ "$CONFIG_VERSION" == "v2" || "$CONFIG_VERSION" == "v3" || "$CONFIG_VERSION" == "v4" ]]; then
            # v2/v3/v4 configs are designed for fine-tuning from pretrained model
            awsc_mode="pretrain"
            log_info "Config ${CONFIG_VERSION}: auto-setting AWSC mode to 'pretrain' (designed for fine-tuning)"
        else
            awsc_mode="scratch"
        fi
    fi
    
    # Validate pretrain mode requirements
    if [[ "$algorithm" == "awsc" || -z "$algorithm" ]]; then
        if [[ "$awsc_mode" == "pretrain" || "$awsc_mode" == "both" ]]; then
            # Use default pretrain path if not explicitly provided
            if [[ -z "$pretrain_path" ]]; then
                pretrain_path="${DEFAULT_AWSC_PRETRAIN_PATH}"
                log_info "Using default AWSC pretrain path: ${pretrain_path}"
            fi
            if [[ ! -f "$pretrain_path" ]]; then
                log_error "Pretrained checkpoint not found: $pretrain_path"
                if [[ "$awsc_mode" == "both" ]]; then
                    log_warning "Falling back to scratch mode only"
                    awsc_mode="scratch"
                else
                    exit 1
                fi
            fi
        fi
    fi
    
    # Check demo file
    if ! check_demo_file; then
        exit 1
    fi
    
    local algorithms_to_run=()
    
    if [[ -n "$algorithm" ]]; then
        algorithms_to_run=("$algorithm")
    else
        algorithms_to_run=("${ALL_ALGORITHMS[@]}")
    fi
    
    log_info "Config version: ${CONFIG_VERSION}"
    log_info "Environment: ${ENV_ID}"
    log_info "Total timesteps: ${TOTAL_TIMESTEPS}"
    log_info "Algorithms to run: ${algorithms_to_run[*]}"
    log_info "AWSC mode: ${awsc_mode}"
    if [[ -n "$pretrain_path" ]]; then
        log_info "Pretrain path: ${pretrain_path}"
    fi
    log_info "Available GPUs: ${AVAILABLE_GPUS[*]} (${NUM_GPUS} total)"
    
    local sweep_items=()
    for algo in "${algorithms_to_run[@]}"; do
        log_info "=========================================="
        log_info "Processing algorithm: ${algo}"
        log_info "=========================================="
        
        local config_dir="configs"
        if [[ "$CONFIG_VERSION" == "v2" ]]; then
            config_dir="configs_v2"
        elif [[ "$CONFIG_VERSION" == "v3" ]]; then
            config_dir="configs_v3"
        elif [[ "$CONFIG_VERSION" == "v4" ]]; then
            config_dir="configs_v4"
        fi
        local config_file="${SCRIPT_DIR}/${config_dir}/${algo}.sh"
        if [[ ! -f "$config_file" ]]; then
            log_warning "Config file not found: ${config_file}, skipping"
            continue
        fi
        
        source "$config_file"
        
        if [[ ${#SWEEP_CONFIGS[@]} -eq 0 ]]; then
            log_warning "No configs found for ${algo}, skipping"
            continue
        fi
        
        # Handle AWSC modes
        if [[ "$algo" == "awsc" ]]; then
            local modes_to_run=()
            if [[ "$awsc_mode" == "both" ]]; then
                modes_to_run=("scratch" "pretrain")
            else
                modes_to_run=("$awsc_mode")
            fi
            
            for mode in "${modes_to_run[@]}"; do
                log_info "--- AWSC Mode: ${mode} ---"
                
                # Prepare configs with mode suffix and pretrain_path
                for config in "${SWEEP_CONFIGS[@]}"; do
                    local config_name=$(echo "$config" | cut -d':' -f1)
                    local extra_args=$(echo "$config" | cut -d':' -f2-)
                    if [[ "$config_name" == "$extra_args" ]]; then
                        extra_args=""
                    fi
                    
                    # Add mode suffix to config name
                    local new_name="${config_name}_${mode}"
                    
                    # Add pretrain_path and match offline checkpoint's pred_horizon for pretrain mode
                    if [[ "$mode" == "pretrain" ]]; then
                        extra_args="${extra_args} --pretrain_path ${pretrain_path} --pred_horizon 8 --load_pretrain_critic"
                    fi
                    
                    if [[ "$dry_run" == "true" ]]; then
                        echo "  - ${new_name}"
                    else
                        sweep_items+=("${algo}|${new_name}|${extra_args}")
                    fi
                done
                
                if [[ "$dry_run" == "true" ]]; then
                    log_info "[DRY RUN] Would run ${#SWEEP_CONFIGS[@]} configs for ${algo}_${mode}"
                fi
            done
        else
            # Non-AWSC algorithms
            for config in "${SWEEP_CONFIGS[@]}"; do
                local config_name=$(echo "$config" | cut -d':' -f1)
                local extra_args=$(echo "$config" | cut -d':' -f2-)
                if [[ "$config_name" == "$extra_args" ]]; then
                    extra_args=""
                fi
                
                if [[ "$dry_run" == "true" ]]; then
                    echo "  - ${config_name}"
                else
                    sweep_items+=("${algo}|${config_name}|${extra_args}")
                fi
            done
            
            if [[ "$dry_run" == "true" ]]; then
                log_info "[DRY RUN] Would run ${#SWEEP_CONFIGS[@]} configs for ${algo}"
            fi
        fi
    done
    
    # Run all queued configs using global queue scheduling
    if [[ "$dry_run" != "true" ]]; then
        if [[ ${#sweep_items[@]} -eq 0 ]]; then
            log_error "No configs found to run"
            exit 1
        fi
        run_sweep_queue "${sweep_items[@]}"
    fi
    
    log_success "Sweep completed!"
}

# -----------------------------------------------------------------------------
# Retry Command
# -----------------------------------------------------------------------------
cmd_retry() {
    local algorithm=""
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm)
                algorithm=$2
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --config-version)
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    local algorithms_to_check=()
    
    if [[ -n "$algorithm" ]]; then
        algorithms_to_check=("$algorithm")
    else
        algorithms_to_check=("${ALL_ALGORITHMS[@]}")
    fi
    
    for algo in "${algorithms_to_check[@]}"; do
        local config_dir="configs"
        if [[ "$CONFIG_VERSION" == "v2" ]]; then
            config_dir="configs_v2"
        elif [[ "$CONFIG_VERSION" == "v3" ]]; then
            config_dir="configs_v3"
        elif [[ "$CONFIG_VERSION" == "v4" ]]; then
            config_dir="configs_v4"
        fi
        local config_file="${SCRIPT_DIR}/${config_dir}/${algo}.sh"
        if [[ ! -f "$config_file" ]]; then
            continue
        fi
        
        source "$config_file"
        local failed_configs=()
        
        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name=$(echo "$config" | cut -d':' -f1)
            local exp_dir=$(find_actual_exp_dir "$algo" "$config_name")
            
            if is_experiment_failed "$exp_dir"; then
                failed_configs+=("$config")
            fi
        done
        
        if [[ ${#failed_configs[@]} -eq 0 ]]; then
            log_info "No failed experiments for ${algo}"
            continue
        fi
        
        log_info "Found ${#failed_configs[@]} failed experiments for ${algo}"
        
        if [[ "$dry_run" == "true" ]]; then
            log_info "[DRY RUN] Would retry:"
            for config in "${failed_configs[@]}"; do
                local config_name=$(echo "$config" | cut -d':' -f1)
                echo "  - ${config_name}"
            done
        else
            # Clean up failed experiment directories before retry
            for config in "${failed_configs[@]}"; do
                local config_name=$(echo "$config" | cut -d':' -f1)
                local exp_dir=$(find_actual_exp_dir "$algo" "$config_name")
                if [[ -d "$exp_dir" ]]; then
                    log_info "Cleaning up ${exp_dir}"
                    rm -rf "$exp_dir"
                fi
            done
            
            # Build queue from failed configs for efficient GPU scheduling
            local retry_items=()
            for config in "${failed_configs[@]}"; do
                local cname=$(echo "$config" | cut -d':' -f1)
                local cargs=$(echo "$config" | cut -d':' -f2-)
                if [[ "$cname" == "$cargs" ]]; then
                    cargs=""
                fi
                retry_items+=("${algo}|${cname}|${cargs}")
            done
            run_sweep_queue "${retry_items[@]}"
        fi
    done
    
    log_success "Retry completed!"
}

# -----------------------------------------------------------------------------
# Status Command
# -----------------------------------------------------------------------------
cmd_status() {
    local algorithm=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm)
                algorithm=$2
                shift 2
                ;;
            --config-version)
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    local algorithms_to_check=()
    
    if [[ -n "$algorithm" ]]; then
        algorithms_to_check=("$algorithm")
    else
        algorithms_to_check=("${ALL_ALGORITHMS[@]}")
    fi
    
    local total_all=0
    local success_all=0
    local failed_all=0
    local not_started_all=0
    
    echo ""
    echo "Config version: ${CONFIG_VERSION}"
    echo "Environment: ${ENV_ID}"
    echo "Total timesteps: ${TOTAL_TIMESTEPS}"
    echo "Sweep directory: ${SWEEP_BASE_DIR}"
    echo ""
    
    for algo in "${algorithms_to_check[@]}"; do
        local config_dir="configs"
        if [[ "$CONFIG_VERSION" == "v2" ]]; then
            config_dir="configs_v2"
        elif [[ "$CONFIG_VERSION" == "v3" ]]; then
            config_dir="configs_v3"
        elif [[ "$CONFIG_VERSION" == "v4" ]]; then
            config_dir="configs_v4"
        fi
        local config_file="${SCRIPT_DIR}/${config_dir}/${algo}.sh"
        if [[ ! -f "$config_file" ]]; then
            continue
        fi
        
        source "$config_file"
        
        echo "========================================"
        echo "Algorithm: ${algo}"
        echo "========================================"
        
        local total=0
        local success=0
        local failed=0
        local not_started=0
        
        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name=$(echo "$config" | cut -d':' -f1)
            local exp_dir=$(find_actual_exp_dir "$algo" "$config_name")
            
            total=$((total + 1))
            
            if is_experiment_successful "$exp_dir"; then
                echo -e "  ${GREEN}✓${NC} ${config_name}"
                success=$((success + 1))
            elif is_experiment_failed "$exp_dir"; then
                echo -e "  ${RED}✗${NC} ${config_name}"
                failed=$((failed + 1))
            else
                echo -e "  ${YELLOW}○${NC} ${config_name}"
                not_started=$((not_started + 1))
            fi
        done
        
        echo "----------------------------------------"
        echo "Total: ${total} | Success: ${success} | Failed: ${failed} | Not Started: ${not_started}"
        echo ""
        
        total_all=$((total_all + total))
        success_all=$((success_all + success))
        failed_all=$((failed_all + failed))
        not_started_all=$((not_started_all + not_started))
    done
    
    echo "========================================"
    echo "OVERALL SUMMARY"
    echo "========================================"
    echo "Total: ${total_all} | Success: ${success_all} | Failed: ${failed_all} | Not Started: ${not_started_all}"
    
    if [[ $total_all -gt 0 ]]; then
        local success_rate=$((success_all * 100 / total_all))
        echo "Success Rate: ${success_rate}%"
    fi
}

# -----------------------------------------------------------------------------
# Analyze Command (with metrics)
# -----------------------------------------------------------------------------
cmd_analyze() {
    local export_file=""
    local algorithm=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --export)
                export_file=$2
                shift 2
                ;;
            --algorithm)
                algorithm=$2
                shift 2
                ;;
            --config-version)
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Create analysis output directory
    local analysis_dir="${SWEEP_BASE_DIR}/analysis_results"
    mkdir -p "$analysis_dir"
    
    local algorithms_to_analyze=()
    if [[ -n "$algorithm" ]]; then
        algorithms_to_analyze=("$algorithm")
    else
        algorithms_to_analyze=("${ALL_ALGORITHMS[@]}")
    fi
    
    echo ""
    echo "Config version: ${CONFIG_VERSION}"
    echo "Environment: ${ENV_ID}"
    echo "Total timesteps: ${TOTAL_TIMESTEPS}"
    echo "Sweep directory: ${SWEEP_BASE_DIR}"
    echo ""
    
    # Run detailed analysis with metrics for each algorithm
    for algo in "${algorithms_to_analyze[@]}"; do
        analyze_algorithm_with_metrics "$algo"
    done
    
    # Save analysis summary to file (strip ANSI color codes)
    local summary_file="${analysis_dir}/analysis_summary_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "RLPD Sweep Analysis Summary"
        echo "Generated: $(date)"
        echo "Config version: ${CONFIG_VERSION}"
        echo "Environment: ${ENV_ID}"
        echo "Total timesteps: ${TOTAL_TIMESTEPS}"
        echo ""
        for algo in "${algorithms_to_analyze[@]}"; do
            analyze_algorithm_with_metrics "$algo"
        done
    } | sed 's/\x1b\[[0-9;]*m//g' > "$summary_file"
    log_info "Analysis summary saved to ${summary_file}"
    
    # Auto-export JSON
    local json_file="${export_file:-${analysis_dir}/results.json}"
    export_results_json "$json_file"
}

# -----------------------------------------------------------------------------
# Report Command (Python analysis with visualizations)
# -----------------------------------------------------------------------------
cmd_report() {
    local algorithm=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm)
                algorithm=$2
                shift 2
                ;;
            --config-version)
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    local analysis_dir="${SWEEP_BASE_DIR}/analysis_results"
    mkdir -p "$analysis_dir"
    
    local cmd="python ${SCRIPT_DIR}/analyze_sweep.py --sweep-dir ${SWEEP_BASE_DIR} --config-version ${CONFIG_VERSION} --output-dir ${analysis_dir}"
    
    if [[ -n "$algorithm" ]]; then
        cmd="${cmd} --algorithm ${algorithm}"
    fi
    
    log_info "Running Python analysis..."
    eval "$cmd"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi
    
    local command=$1
    shift
    
    case $command in
        run)
            cmd_run "$@"
            ;;
        retry)
            cmd_retry "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        analyze)
            cmd_analyze "$@"
            ;;
        report)
            cmd_report "$@"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            ;;
    esac
}

main "$@"

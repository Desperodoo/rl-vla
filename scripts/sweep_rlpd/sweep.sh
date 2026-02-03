#!/bin/bash
# =============================================================================
# Sweep - Unified entry point for RLPD hyperparameter sweep
# =============================================================================
#
# Usage:
#   ./sweep.sh run [--algorithm ALGO] [--mode MODE] [--pretrain-path PATH]
#   ./sweep.sh retry [--algorithm ALGO]
#   ./sweep.sh status [--algorithm ALGO]
#   ./sweep.sh analyze [--export FILE]
#
# Examples:
#   ./sweep.sh run                                  # Run all algorithms
#   ./sweep.sh run --algorithm sac                  # Run only SAC
#   ./sweep.sh run --algorithm awsc --mode scratch  # Run AWSC from scratch
#   ./sweep.sh run --algorithm awsc --mode pretrain --pretrain-path runs/shortcut_flow/best.pt
#   ./sweep.sh retry                                # Retry all failed experiments
#   ./sweep.sh retry --algorithm sac                # Retry failed SAC experiments
#   ./sweep.sh status                               # Show status of all experiments
#   ./sweep.sh analyze --export results.json        # Export results to JSON
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    echo "  analyze   Analyze and export results"
    echo ""
    echo "Options for 'run':"
    echo "  --algorithm ALGO    Run only specified algorithm (sac, awsc)"
    echo "  --mode MODE         AWSC mode: scratch, pretrain, both (default: scratch)"
    echo "  --pretrain-path P   Pretrained checkpoint path (required for AWSC pretrain mode)"
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
    echo "  --export FILE       Export results to JSON file"
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
    local awsc_mode="scratch"
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
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate pretrain mode requirements
    if [[ "$algorithm" == "awsc" || -z "$algorithm" ]]; then
        if [[ "$awsc_mode" == "pretrain" || "$awsc_mode" == "both" ]]; then
            if [[ -z "$pretrain_path" ]]; then
                log_error "AWSC pretrain mode requires --pretrain-path"
                if [[ "$awsc_mode" == "both" ]]; then
                    log_warning "Falling back to scratch mode only"
                    awsc_mode="scratch"
                else
                    exit 1
                fi
            elif [[ ! -f "$pretrain_path" ]]; then
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
    
    log_info "Environment: ${ENV_ID}"
    log_info "Total timesteps: ${TOTAL_TIMESTEPS}"
    log_info "Algorithms to run: ${algorithms_to_run[*]}"
    log_info "Available GPUs: ${AVAILABLE_GPUS[*]} (${NUM_GPUS} total)"
    
    for algo in "${algorithms_to_run[@]}"; do
        log_info "=========================================="
        log_info "Processing algorithm: ${algo}"
        log_info "=========================================="
        
        local config_file="${SCRIPT_DIR}/configs/${algo}.sh"
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
                local mode_configs=()
                for config in "${SWEEP_CONFIGS[@]}"; do
                    local config_name=$(echo "$config" | cut -d':' -f1)
                    local extra_args=$(echo "$config" | cut -d':' -f2-)
                    
                    # Add mode suffix to config name
                    local new_name="${config_name}_${mode}"
                    
                    # Add pretrain_path for pretrain mode
                    if [[ "$mode" == "pretrain" ]]; then
                        extra_args="${extra_args} --pretrain_path ${pretrain_path}"
                    fi
                    
                    mode_configs+=("${new_name}:${extra_args}")
                done
                
                if [[ "$dry_run" == "true" ]]; then
                    log_info "[DRY RUN] Would run ${#mode_configs[@]} configs for ${algo}_${mode}:"
                    for config in "${mode_configs[@]}"; do
                        local cname=$(echo "$config" | cut -d':' -f1)
                        echo "  - ${cname}"
                    done
                else
                    run_batch "$algo" "${mode_configs[@]}"
                fi
            done
        else
            # Non-AWSC algorithms
            if [[ "$dry_run" == "true" ]]; then
                log_info "[DRY RUN] Would run ${#SWEEP_CONFIGS[@]} configs:"
                for config in "${SWEEP_CONFIGS[@]}"; do
                    local config_name=$(echo "$config" | cut -d':' -f1)
                    echo "  - ${config_name}"
                done
            else
                run_batch "$algo" "${SWEEP_CONFIGS[@]}"
            fi
        fi
    done
    
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
        local config_file="${SCRIPT_DIR}/configs/${algo}.sh"
        if [[ ! -f "$config_file" ]]; then
            continue
        fi
        
        source "$config_file"
        local failed_configs=()
        
        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name=$(echo "$config" | cut -d':' -f1)
            local exp_dir=$(get_exp_dir "$algo" "$config_name")
            
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
                local exp_dir=$(get_exp_dir "$algo" "$config_name")
                if [[ -d "$exp_dir" ]]; then
                    log_info "Cleaning up ${exp_dir}"
                    rm -rf "$exp_dir"
                fi
            done
            
            run_batch "$algo" "${failed_configs[@]}"
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
    echo "Environment: ${ENV_ID}"
    echo "Total timesteps: ${TOTAL_TIMESTEPS}"
    echo "Sweep directory: ${SWEEP_BASE_DIR}"
    echo ""
    
    for algo in "${algorithms_to_check[@]}"; do
        local config_file="${SCRIPT_DIR}/configs/${algo}.sh"
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
            local exp_dir=$(get_exp_dir "$algo" "$config_name")
            
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
# Analyze Command
# -----------------------------------------------------------------------------
cmd_analyze() {
    local export_file=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --export)
                export_file=$2
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Show status first
    cmd_status
    
    # Export if requested
    if [[ -n "$export_file" ]]; then
        export_results_json "$export_file"
    fi
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

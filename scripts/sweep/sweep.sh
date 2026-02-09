#!/bin/bash
# =============================================================================
# Sweep - Unified entry point for hyperparameter sweep
# =============================================================================
#
# Usage:
#   ./sweep.sh run [--stage N] [--algorithm ALGO] [--config-version V]
#   ./sweep.sh retry [--algorithm ALGO] [--config-version V]
#   ./sweep.sh status [--algorithm ALGO] [--config-version V]
#   ./sweep.sh analyze [--export FILE] [--config-version V]
#   ./sweep.sh report [--algorithm ALGO] [--output-dir DIR] [--config-version V]
#
# Examples:
#   ./sweep.sh run                          # Run wave 1 (default)
#   ./sweep.sh run --config-version v2      # Run wave 2
#   ./sweep.sh run --stage 1                # Run only stage 1 (IL algorithms)
#   ./sweep.sh run --algorithm flow_matching  # Run only flow_matching
#   ./sweep.sh retry                        # Retry all failed experiments
#   ./sweep.sh retry --algorithm cpql       # Retry failed cpql experiments
#   ./sweep.sh status                       # Show status of all experiments
#   ./sweep.sh analyze --export results.json  # Export results to JSON
#   ./sweep.sh report                       # Generate analysis report with visualizations
#
# =============================================================================

# Don't use set -e because we want to continue even if some experiments fail
# set -e

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
    echo "  analyze   Analyze and export results (quick view)"
    echo "  report    Generate comprehensive analysis report with visualizations"
    echo ""
    echo "Global Options:"
    echo "  --config-version V  Use config version (v1=wave1, v2=wave2, v3=wave3, default: v1)"
    echo ""
    echo "Options for 'run':"
    echo "  --stage N         Run only stage N (1, 2, or 3)"
    echo "  --algorithm ALGO  Run only specified algorithm"
    echo "  --dry-run         Show what would be run without executing"
    echo ""
    echo "Options for 'retry':"
    echo "  --algorithm ALGO  Retry only specified algorithm"
    echo "  --dry-run         Show what would be retried without executing"
    echo ""
    echo "Options for 'status':"
    echo "  --algorithm ALGO  Show status for specified algorithm only"
    echo ""
    echo "Options for 'analyze':"
    echo "  --export FILE     Export results to JSON file"
    echo "  --algorithm ALGO  Analyze only specified algorithm"
    echo ""
    echo "Options for 'report':"
    echo "  --algorithm ALGO  Generate report for specified algorithm only"
    echo "  --output-dir DIR  Output directory for report (default: analysis_results)"
    echo ""
    exit 1
}

# -----------------------------------------------------------------------------
# Get algorithms for a stage
# -----------------------------------------------------------------------------
get_stage_algorithms() {
    local stage=$1
    case $stage in
        1) echo "${STAGE1_ALGORITHMS[@]}" ;;
        2) echo "${STAGE2_ALGORITHMS[@]}" ;;
        3) echo "${STAGE3_ALGORITHMS[@]}" ;;
        *) echo "" ;;
    esac
}

# -----------------------------------------------------------------------------
# Run Command
# -----------------------------------------------------------------------------
cmd_run() {
    local stage=""
    local algorithm=""
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stage)
                stage=$2
                shift 2
                ;;
            --algorithm)
                algorithm=$2
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
    
    log_info "Config version: ${CONFIG_VERSION}"
    local algorithms_to_run=()
    
    if [[ -n "$algorithm" ]]; then
        algorithms_to_run=("$algorithm")
    elif [[ -n "$stage" ]]; then
        algorithms_to_run=($(get_stage_algorithms "$stage"))
    else
        # Run all stages in order
        algorithms_to_run=("${ALL_ALGORITHMS[@]}")
    fi
    
    if [[ ${#algorithms_to_run[@]} -eq 0 ]]; then
        log_error "No algorithms to run"
        exit 1
    fi
    
    log_info "Algorithms to run: ${algorithms_to_run[*]}"
    
    if [[ "$dry_run" == "true" ]]; then
        for algo in "${algorithms_to_run[@]}"; do
            log_info "=========================================="
            log_info "Processing algorithm: ${algo}"
            log_info "=========================================="

            local config_dir="configs"
            if [[ "$CONFIG_VERSION" == "v2" ]]; then
                config_dir="configs_v2"
            elif [[ "$CONFIG_VERSION" == "v3" ]]; then
                config_dir="configs_v3"
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

            log_info "[DRY RUN] Would run ${#SWEEP_CONFIGS[@]} configs:"
            for config in "${SWEEP_CONFIGS[@]}"; do
                local config_name=$(echo "$config" | cut -d':' -f1)
                echo "  - ${config_name}"
            done
        done
    else
        local sweep_items=()
        for algo in "${algorithms_to_run[@]}"; do
            log_info "=========================================="
            log_info "Queueing algorithm: ${algo}"
            log_info "=========================================="

            local config_dir="configs"
            if [[ "$CONFIG_VERSION" == "v2" ]]; then
                config_dir="configs_v2"
            elif [[ "$CONFIG_VERSION" == "v3" ]]; then
                config_dir="configs_v3"
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

            for config in "${SWEEP_CONFIGS[@]}"; do
                local config_name=$(echo "$config" | cut -d':' -f1)
                local extra_args=$(echo "$config" | cut -d':' -f2-)
                if [[ "$config_name" == "$extra_args" ]]; then
                    extra_args=""
                fi
                sweep_items+=("${algo}|${config_name}|${extra_args}")
            done
        done

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
                # Already handled globally, skip
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    log_info "Config version: ${CONFIG_VERSION}"
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
                # Clean both base dir and timestamped dirs
                local base_dir=$(get_exp_dir "$algo" "$config_name")
                local actual_dir=$(find_actual_exp_dir "$algo" "$config_name")
                if [[ -d "$base_dir" ]]; then
                    log_info "Cleaning up ${base_dir}"
                    rm -rf "$base_dir"
                fi
                if [[ -d "$actual_dir" ]] && [[ "$actual_dir" != "$base_dir" ]]; then
                    log_info "Cleaning up ${actual_dir}"
                    rm -rf "$actual_dir"
                fi
            done
            
            run_batch "$algo" "${failed_configs[@]}"
        fi
    done
    
    log_success "Retry completed!"
}

# -----------------------------------------------------------------------------
# Status Command (quick overview)
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
                # Already handled globally, skip
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    log_info "Config version: ${CONFIG_VERSION}"
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
    
    for algo in "${algorithms_to_check[@]}"; do
        local config_dir="configs"
        if [[ "$CONFIG_VERSION" == "v2" ]]; then
            config_dir="configs_v2"
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
# Analyze Command (detailed analysis with metrics)
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
                # Already handled globally, skip
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    log_info "Config version: ${CONFIG_VERSION}"
    local algorithms_to_check=()
    
    if [[ -n "$algorithm" ]]; then
        algorithms_to_check=("$algorithm")
    else
        algorithms_to_check=("${ALL_ALGORITHMS[@]}")
    fi
    
    # Create analysis output directory
    local analysis_dir="${SWEEP_BASE_DIR}/analysis_results"
    mkdir -p "$analysis_dir"
    
    # Detailed analysis with metrics for each algorithm (tee to file)
    local summary_file="${analysis_dir}/analyze_summary.txt"
    {
        echo "Sweep Analysis Summary"
        echo "Config version: ${CONFIG_VERSION}"
        echo "Date: $(date -Iseconds)"
        echo "Sweep dir: ${SWEEP_BASE_DIR}"
        echo ""
    } > "$summary_file"
    
    for algo in "${algorithms_to_check[@]}"; do
        analyze_algorithm_with_metrics "$algo" | tee -a "$summary_file"
    done
    
    # Strip ANSI color codes from saved file
    sed -i 's/\x1b\[[0-9;]*m//g' "$summary_file"
    
    # Always export JSON to analysis dir
    local auto_export="${analysis_dir}/sweep_results.json"
    export_results_json "$auto_export"
    log_success "Results saved to ${analysis_dir}/"
    log_info "  - analyze_summary.txt: Text summary"
    log_info "  - sweep_results.json: JSON data"
    
    # Also export to custom file if requested
    if [[ -n "$export_file" ]]; then
        export_results_json "$export_file"
        log_info "  - ${export_file}: Custom export"
    fi
}

# -----------------------------------------------------------------------------
# Report Command (comprehensive analysis with Python)
# -----------------------------------------------------------------------------
cmd_report() {
    local algorithm=""
    local output_dir=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm)
                algorithm=$2
                shift 2
                ;;
            --output-dir)
                output_dir=$2
                shift 2
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
    
    # Default output dir: <sweep-dir>/analysis_results
    if [[ -z "$output_dir" ]]; then
        output_dir="${SWEEP_BASE_DIR}/analysis_results"
    fi
    
    log_info "Config version: ${CONFIG_VERSION}"
    log_info "Generating comprehensive analysis report..."
    
    # Build Python command
    local python_script="${SCRIPT_DIR}/analyze_sweep.py"
    
    if [[ ! -f "$python_script" ]]; then
        log_error "Analysis script not found: ${python_script}"
        exit 1
    fi
    
    local cmd="python ${python_script}"
    cmd+=" --sweep-dir ${SWEEP_BASE_DIR}"
    cmd+=" --config-version ${CONFIG_VERSION}"
    cmd+=" --output-dir ${output_dir}"
    
    if [[ -n "$algorithm" ]]; then
        cmd+=" --algorithm ${algorithm}"
    fi
    
    log_info "Running: ${cmd}"
    eval "$cmd"
    
    if [[ $? -eq 0 ]]; then
        log_success "Report generated in ${output_dir}/"
        log_info "  - analysis_report.md: Main report"
        log_info "  - analysis_data.json: Raw data"
        log_info "  - *.png: Visualization figures"
    else
        log_error "Report generation failed"
        exit 1
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

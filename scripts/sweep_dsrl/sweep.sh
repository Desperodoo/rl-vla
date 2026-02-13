#!/bin/bash
# =============================================================================
# DSRL Hyperparameter Sweep — Main Entry Point
# =============================================================================
# Orchestrates DSRL-SAC hyperparameter sweeps with GPU-parallel scheduling.
#
# DSRL (Diffusion-based SAC in Residual/Noise space) trains a SAC agent in the
# noise space of a pretrained ShortCut Flow policy. Unlike RLPD sweeps, DSRL
# always requires a pretrained checkpoint and does not need offline demos.
#
# Usage:
#   ./sweep.sh run [dsrl_sac]             # Run sweep for dsrl_sac (default)
#   ./sweep.sh run dsrl_sac --dry-run     # Preview commands without executing
#   ./sweep.sh retry [dsrl_sac]           # Retry failed experiments
#   ./sweep.sh status [dsrl_sac]          # Show experiment status
#   ./sweep.sh analyze [dsrl_sac]         # Detailed analysis with metrics
#   ./sweep.sh report                     # Export JSON + Python analysis
#
# Environment Variables:
#   GPU_IDS=0,1,2,3          Override GPU list
#   CHECKPOINT=path/to/ckpt  Override pretrained checkpoint
#   TOTAL_TIMESTEPS=500000   Override training steps
#   CONFIG_VERSION=v1        Config version (v1 default)
#   USE_WANDB=false          Disable WandB logging
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "DSRL Hyperparameter Sweep"
    echo ""
    echo "Usage: $0 <command> [algorithm] [options]"
    echo ""
    echo "Commands:"
    echo "  run      [dsrl_sac]          Run sweep (default: dsrl_sac)"
    echo "  retry    [dsrl_sac]          Retry failed experiments"
    echo "  status   [dsrl_sac]          Show experiment status"
    echo "  analyze  [dsrl_sac]          Detailed analysis with metrics"
    echo "  report                       Export JSON report + Python analysis"
    echo ""
    echo "Options:"
    echo "  --dry-run                    Preview commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0 run                       # Run all dsrl_sac sweep configs"
    echo "  $0 run dsrl_sac --dry-run    # Preview commands"
    echo "  $0 status                    # Show all experiment status"
    echo "  $0 analyze                   # Detailed metrics analysis"
    echo "  $0 report                    # Export JSON + generate plots"
    echo ""
    echo "Environment Variables:"
    echo "  GPU_IDS=0,1        Override GPU list"
    echo "  CHECKPOINT=...     Override pretrained checkpoint path"
    echo "  TOTAL_TIMESTEPS=N  Override training steps"
    echo "  CONFIG_VERSION=v1  Config version (default: v1)"
}

# =============================================================================
# Command: run
# =============================================================================
cmd_run() {
    local algorithm="dsrl_sac"
    local dry_run=false

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run) dry_run=true; shift ;;
            --*) shift ;;          # skip unknown flags
            *)  algorithm="$1"; shift ;;
        esac
    done

    echo "========================================"
    echo "DSRL Sweep — Run"
    echo "========================================"
    echo "Algorithm:       ${algorithm}"
    echo "Env:             ${ENV_ID}"
    echo "Total timesteps: ${TOTAL_TIMESTEPS}"
    echo "Checkpoint:      ${CHECKPOINT}"
    echo "Sweep dir:       ${SWEEP_BASE_DIR}"
    echo "GPUs:            ${AVAILABLE_GPUS[*]} (${NUM_GPUS} total)"
    echo "Config version:  ${CONFIG_VERSION}"
    echo "WandB:           ${USE_WANDB}"
    echo "========================================"

    # Validate checkpoint (skip in dry-run mode)
    if [[ "$dry_run" != "true" ]] && ! check_checkpoint; then
        exit 1
    fi

    # Load configs
    local config_dir
    config_dir=$(get_config_dir)
    local config_file="${SCRIPT_DIR}/${config_dir}/${algorithm}.sh"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: ${config_file}"
        log_info "Available configs:"
        ls -1 "${SCRIPT_DIR}/${config_dir}/" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    source "$config_file"

    if [[ ${#SWEEP_CONFIGS[@]} -eq 0 ]]; then
        log_error "No configs found in ${config_file}"
        exit 1
    fi

    log_info "Loaded ${#SWEEP_CONFIGS[@]} configs from ${config_file}"

    # Dry-run mode: just print commands
    if [[ "$dry_run" == "true" ]]; then
        echo ""
        log_info "=== DRY RUN — Commands that would be executed ==="
        echo ""
        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name
            config_name=$(echo "$config" | cut -d':' -f1)
            local extra_args
            extra_args=$(echo "$config" | cut -d':' -f2-)
            if [[ "$config_name" == "$extra_args" ]]; then
                extra_args=""
            fi

            echo "# ${config_name}"
            echo "python -m rlft.online.train_dsrl \\"
            echo "  --env_id ${ENV_ID} \\"
            echo "  --total_timesteps ${TOTAL_TIMESTEPS} \\"
            echo "  --checkpoint ${CHECKPOINT} \\"
            echo "  --pred_horizon ${PRED_HORIZON} \\"
            echo "  --exp_name ${EXP_NAME}/${algorithm}/${config_name} \\"
            if [[ -n "$extra_args" ]]; then
                echo "  ${extra_args}"
            fi
            echo ""
        done
        return 0
    fi

    # Change to project root
    cd "$PROJECT_ROOT"

    # Build global queue for efficient GPU scheduling
    local sweep_items=()
    for config in "${SWEEP_CONFIGS[@]}"; do
        local config_name
        config_name=$(echo "$config" | cut -d':' -f1)
        local extra_args
        extra_args=$(echo "$config" | cut -d':' -f2-)
        if [[ "$config_name" == "$extra_args" ]]; then
            extra_args=""
        fi
        sweep_items+=("${algorithm}|${config_name}|${extra_args}")
    done

    log_info "Starting DSRL sweep for ${algorithm} (${#sweep_items[@]} configs, queue mode)..."
    run_sweep_queue "${sweep_items[@]}"

    log_success "Sweep completed for ${algorithm}!"
}

# =============================================================================
# Command: retry
# =============================================================================
cmd_retry() {
    local algorithm="${1:-dsrl_sac}"

    echo "========================================"
    echo "DSRL Sweep — Retry Failed"
    echo "========================================"

    if ! check_checkpoint; then
        exit 1
    fi

    cd "$PROJECT_ROOT"

    local failed
    failed=($(find_failed_experiments "$algorithm"))

    if [[ ${#failed[@]} -eq 0 ]]; then
        log_success "No failed experiments for ${algorithm}"
        return 0
    fi

    log_info "Found ${#failed[@]} failed experiments for ${algorithm}"
    for f in "${failed[@]}"; do
        local name
        name=$(echo "$f" | cut -d':' -f1)
        echo "  - ${name}"
    done
    echo ""

    read -p "Retry these experiments? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log_info "Aborted"
        return 0
    fi

    # Build queue from failed configs for efficient GPU scheduling
    local retry_items=()
    for f in "${failed[@]}"; do
        local fname
        fname=$(echo "$f" | cut -d':' -f1)
        local fargs
        fargs=$(echo "$f" | cut -d':' -f2-)
        if [[ "$fname" == "$fargs" ]]; then
            fargs=""
        fi
        retry_items+=("${algorithm}|${fname}|${fargs}")
    done
    run_sweep_queue "${retry_items[@]}"
    log_success "Retry completed for ${algorithm}!"
}

# =============================================================================
# Command: status
# =============================================================================
cmd_status() {
    local algorithm="${1:-}"

    echo "========================================"
    echo "DSRL Sweep — Status"
    echo "========================================"
    echo "Sweep dir: ${SWEEP_BASE_DIR}"
    echo ""

    if [[ -n "$algorithm" ]]; then
        analyze_algorithm "$algorithm"
    else
        for algo in "${ALL_ALGORITHMS[@]}"; do
            analyze_algorithm "$algo"
        done
    fi
}

# =============================================================================
# Command: analyze
# =============================================================================
cmd_analyze() {
    local algorithm="${1:-}"

    echo "========================================"
    echo "DSRL Sweep — Detailed Analysis"
    echo "========================================"
    echo "Sweep dir: ${SWEEP_BASE_DIR}"
    echo ""

    if [[ -n "$algorithm" ]]; then
        analyze_algorithm_with_metrics "$algorithm"
    else
        for algo in "${ALL_ALGORITHMS[@]}"; do
            analyze_algorithm_with_metrics "$algo"
        done
    fi
}

# =============================================================================
# Command: report
# =============================================================================
cmd_report() {
    echo "========================================"
    echo "DSRL Sweep — Report Generation"
    echo "========================================"

    # 1. Export JSON results
    local json_file="${SWEEP_BASE_DIR}/sweep_dsrl_results.json"
    mkdir -p "${SWEEP_BASE_DIR}"
    export_results_json "$json_file"

    # 2. Run Python analysis
    local analyze_script="${SCRIPT_DIR}/analyze_sweep.py"
    if [[ -f "$analyze_script" ]]; then
        echo ""
        log_info "Running Python analysis..."
        cd "$PROJECT_ROOT"
        python "$analyze_script" \
            --sweep-dir "${SWEEP_BASE_DIR}" \
            --config-version "${CONFIG_VERSION}" \
            --output-dir "${SWEEP_BASE_DIR}/analysis_results"
    else
        log_warning "analyze_sweep.py not found — skipping Python analysis"
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi

    local command=$1
    shift

    case $command in
        run)      cmd_run "$@" ;;
        retry)    cmd_retry "$@" ;;
        status)   cmd_status "$@" ;;
        analyze)  cmd_analyze "$@" ;;
        report)   cmd_report "$@" ;;
        help|-h|--help) usage ;;
        *)
            log_error "Unknown command: ${command}"
            usage
            exit 1
            ;;
    esac
}

main "$@"

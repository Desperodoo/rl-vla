#!/bin/bash
# =============================================================================
# ACP Sweep — Main Entry Point
# =============================================================================
# Sweeps ACP reward parameters across AWSC, PLD-SAC, and DSRL-SAC.
# Each experiment uses 2 GPUs (RL training + ACP value model inference).
#
# Usage:
#   ./sweep.sh run [--algorithm ALGO] [--dry-run]
#   ./sweep.sh retry [--algorithm ALGO]
#   ./sweep.sh status [--algorithm ALGO]
#   ./sweep.sh analyze [--algorithm ALGO]
#   ./sweep.sh report [--algorithm ALGO]
#
# Examples:
#   ./sweep.sh run                                 # Run all algorithms
#   ./sweep.sh run --algorithm awsc_acp            # Run only AWSC
#   ./sweep.sh run --algorithm pld_acp --dry-run   # Preview PLD commands
#   ./sweep.sh run --algorithm dsrl_acp            # Run only DSRL
#   ./sweep.sh status                              # Show all status
#   ./sweep.sh analyze                             # Detailed metrics analysis
#   ./sweep.sh report                              # Python analysis + charts
#
# Environment Variables:
#   ACP_CKPT=path/to/ckpt           Override ACP checkpoint
#   ACP_REWARD_SCALE=200.0          Override default reward scale
#   CHECKPOINT=path/to/policy.pt    Override pretrained policy
#   GPU_PAIRS="0,1 2,3"             Override GPU pairs
#   SEED=42                         Random seed
#   USE_WANDB=false                 Disable WandB
#
# Environment: rlft_ms3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "ACP Reward Hyperparameter Sweep"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  run       Run sweep experiments"
    echo "  retry     Retry failed experiments"
    echo "  status    Show experiment status"
    echo "  analyze   Detailed analysis with metrics (focus: success_at_end)"
    echo "  report    Export JSON + Python analysis with charts"
    echo ""
    echo "Options:"
    echo "  --algorithm ALGO   Target algorithm: awsc_acp, pld_acp, dsrl_acp"
    echo "  --dry-run          Preview commands without executing"
    echo ""
    echo "Algorithms:"
    echo "  awsc_acp   AWSC + ACP reward (train_rlpd, 500K steps)"
    echo "  pld_acp    PLD-SAC + ACP reward (train_pld, 71K steps)"
    echo "  dsrl_acp   DSRL-SAC + ACP reward (train_dsrl, 71K steps)"
    echo ""
    echo "GPU Layout: Each experiment uses 2 GPUs (train + ACP model)."
    echo "  Default: 5 parallel experiments on 10 GPUs (pairs 0,1 | 2,3 | 4,5 | 6,7 | 8,9)"
}

# =============================================================================
# Command: run
# =============================================================================
cmd_run() {
    local algorithm=""
    local dry_run=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm) algorithm=$2; shift 2 ;;
            --dry-run) dry_run=true; shift ;;
            --*) shift ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "ACP Sweep — Run"
    echo "========================================"
    echo "Env:             ${ENV_ID}"
    echo "Checkpoint:      ${CHECKPOINT}"
    echo "ACP checkpoint:  ${ACP_CKPT}"
    echo "ACP scale:       ${ACP_REWARD_SCALE} (default, overridden per config)"
    echo "Seed:            ${SEED}"
    echo "Sweep dir:       ${SWEEP_BASE_DIR}"
    echo "GPU pairs:       ${GPU_PAIRS[*]} (${NUM_GPU_PAIRS} pairs)"
    echo "WandB:           ${USE_WANDB}"
    echo "========================================"

    # Validate
    if [[ "$dry_run" != "true" ]]; then
        if ! check_checkpoint; then exit 1; fi
        if ! check_acp_checkpoint "$ACP_CKPT"; then exit 1; fi
    fi

    local algorithms_to_run=()
    if [[ -n "$algorithm" ]]; then
        algorithms_to_run=("$algorithm")
    else
        algorithms_to_run=("${ALL_ALGORITHMS[@]}")
    fi

    # Build global queue
    local sweep_items=()
    for algo in "${algorithms_to_run[@]}"; do
        local config_dir
        config_dir=$(get_config_dir)
        local config_file="${SCRIPT_DIR}/${config_dir}/${algo}.sh"

        if [[ ! -f "$config_file" ]]; then
            log_warning "Config file not found: ${config_file}, skipping"
            continue
        fi

        source "$config_file"

        if [[ ${#SWEEP_CONFIGS[@]} -eq 0 ]]; then
            log_warning "No configs for ${algo}, skipping"
            continue
        fi

        log_info "Algorithm ${algo}: ${#SWEEP_CONFIGS[@]} configs"

        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name
            config_name=$(echo "$config" | cut -d':' -f1)
            local extra_args
            extra_args=$(echo "$config" | cut -d':' -f2-)
            if [[ "$config_name" == "$extra_args" ]]; then
                extra_args=""
            fi

            if [[ "$dry_run" == "true" ]]; then
                echo "  [${algo}] ${config_name}: ${extra_args:-<defaults>}"
            else
                sweep_items+=("${algo}|${config_name}|${extra_args}")
            fi
        done
    done

    if [[ "$dry_run" == "true" ]]; then
        local n_algos=${#algorithms_to_run[@]}
        log_info "[DRY RUN] Would run ${n_algos} algorithm(s)"
        return 0
    fi

    if [[ ${#sweep_items[@]} -eq 0 ]]; then
        log_error "No configs to run"
        exit 1
    fi

    cd "$PROJECT_ROOT"
    log_info "Starting ACP sweep (${#sweep_items[@]} total configs, queue mode)..."
    run_sweep_queue "${sweep_items[@]}"

    log_success "ACP sweep completed!"
}

# =============================================================================
# Command: retry
# =============================================================================
cmd_retry() {
    local algorithm=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm) algorithm=$2; shift 2 ;;
            --*) shift ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "ACP Sweep — Retry Failed"
    echo "========================================"

    if ! check_checkpoint; then exit 1; fi
    if ! check_acp_checkpoint "$ACP_CKPT"; then exit 1; fi

    cd "$PROJECT_ROOT"

    local algorithms_to_check=()
    if [[ -n "$algorithm" ]]; then
        algorithms_to_check=("$algorithm")
    else
        algorithms_to_check=("${ALL_ALGORITHMS[@]}")
    fi

    local retry_items=()
    for algo in "${algorithms_to_check[@]}"; do
        local failed
        failed=($(find_failed_experiments "$algo"))

        if [[ ${#failed[@]} -eq 0 ]]; then
            log_info "No failed experiments for ${algo}"
            continue
        fi

        log_info "Found ${#failed[@]} failed experiments for ${algo}"
        for f in "${failed[@]}"; do
            local fname
            fname=$(echo "$f" | cut -d':' -f1)
            echo "  - ${fname}"
            local fargs
            fargs=$(echo "$f" | cut -d':' -f2-)
            if [[ "$fname" == "$fargs" ]]; then
                fargs=""
            fi
            retry_items+=("${algo}|${fname}|${fargs}")
        done
    done

    if [[ ${#retry_items[@]} -eq 0 ]]; then
        log_success "No failed experiments to retry"
        return 0
    fi

    run_sweep_queue "${retry_items[@]}"
    log_success "Retry completed!"
}

# =============================================================================
# Command: status
# =============================================================================
cmd_status() {
    local algorithm=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm) algorithm=$2; shift 2 ;;
            --*) shift ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "ACP Sweep — Status"
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
    local algorithm=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm) algorithm=$2; shift 2 ;;
            --*) shift ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "ACP Sweep — Detailed Analysis"
    echo "========================================"
    echo "Sweep dir: ${SWEEP_BASE_DIR}"
    echo "Core metric: success_at_end (ACP's Achilles heel)"
    echo ""

    if [[ -n "$algorithm" ]]; then
        analyze_algorithm_with_metrics "$algorithm"
    else
        for algo in "${ALL_ALGORITHMS[@]}"; do
            analyze_algorithm_with_metrics "$algo"
        done
    fi

    # Print sim baselines for reference
    echo "========================================"
    echo -e "${CYAN}Sim Baselines (seed 42, for comparison):${NC}"
    echo "========================================"
    printf "%-15s %12s %12s\n" "Algorithm" "best_s_end" "best_s_once"
    echo "----------------------------------------"
    printf "%-15s %12s %12s\n" "AWSC (sim)" "0.72" "0.92"
    printf "%-15s %12s %12s\n" "PLD (sim)" "0.86" "1.00"
    printf "%-15s %12s %12s\n" "DSRL (sim)" "0.60" "0.98"
    echo ""
    echo "ACP Mirror baselines (seed 42):"
    printf "%-15s %12s %12s\n" "AWSC (ACP)" "0.66" "0.90"
    printf "%-15s %12s %12s\n" "PLD (ACP)" "0.02" "0.82"
    printf "%-15s %12s %12s\n" "DSRL (ACP)" "0.06" "0.92"
    echo ""
}

# =============================================================================
# Command: report
# =============================================================================
cmd_report() {
    local algorithm=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --algorithm) algorithm=$2; shift 2 ;;
            --*) shift ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "ACP Sweep — Report Generation"
    echo "========================================"

    # 1. Export JSON
    local json_file="${SWEEP_BASE_DIR}/acp_sweep_results.json"
    mkdir -p "${SWEEP_BASE_DIR}"
    export_results_json "$json_file"

    # 2. Run Python analysis
    local analyze_script="${SCRIPT_DIR}/analyze_sweep.py"
    if [[ -f "$analyze_script" ]]; then
        echo ""
        log_info "Running Python analysis..."
        cd "$PROJECT_ROOT"
        local cmd="python ${analyze_script} --sweep-dir ${SWEEP_BASE_DIR} --output-dir ${SWEEP_BASE_DIR}/analysis_results"
        if [[ -n "$algorithm" ]]; then
            cmd+=" --algorithm ${algorithm}"
        fi
        eval "$cmd"
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

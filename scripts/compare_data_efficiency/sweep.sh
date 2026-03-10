#!/bin/bash
# =============================================================================
# Data-Efficiency Comparison — Main Entry Point
# =============================================================================
# Fair comparison of AWSC (RLPD) vs PLD-SAC vs DSRL-SAC.
# Pipeline structure mirrors sweep_dsrl/ and sweep_pld/ for consistency.
#
# Usage:
#   ./sweep.sh run                     # Run all 3 algorithms in parallel
#   ./sweep.sh run --dry-run           # Preview commands without executing
#   ./sweep.sh retry                   # Retry failed experiments
#   ./sweep.sh status                  # Show experiment status
#   ./sweep.sh analyze                 # Detailed analysis with metrics
#   ./sweep.sh report                  # Export JSON + Python analysis
#
# Multi-seed (recommended for credible results):
#   for seed in 42 100 200; do
#     SEED=$seed ./sweep.sh run
#     sleep 10
#   done
#
# Environment Variables:
#   GPU_IDS=5,6,7,8,9        Override GPU list
#   SEED=42                   Random seed
#   COMPARISON_SCHEME=A       A (robot steps) / B (chunk decisions) / C (raw)
#   CHECKPOINT=path/to/ckpt   Override pretrained checkpoint
#   USE_WANDB=false           Disable WandB logging
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "Data-Efficiency Fair Comparison: AWSC vs PLD-SAC vs DSRL-SAC"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  run                    Run all 3 algorithms in parallel"
    echo "  retry                  Retry failed experiments"
    echo "  status                 Show experiment status"
    echo "  analyze                Detailed analysis with metrics"
    echo "  report                 Export JSON report + Python analysis"
    echo ""
    echo "Options:"
    echo "  --dry-run              Preview commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0 run                                    # Default (scheme A, seed 42)"
    echo "  $0 run --dry-run                          # Preview commands"
    echo "  SEED=100 $0 run                           # Different seed"
    echo "  COMPARISON_SCHEME=B $0 run                # Unified chunk decisions"
    echo "  $0 status                                 # Check progress"
    echo "  $0 report                                 # Generate analysis"
    echo ""
    echo "Environment Variables:"
    echo "  GPU_IDS        Override GPU list (default: 5,6,7,8,9)"
    echo "  SEED           Random seed (default: 42)"
    echo "  COMPARISON_SCHEME  A=robot steps, B=chunk decisions, C=raw"
}

# =============================================================================
# Command: run  (multi-seed)
# =============================================================================
cmd_run() {
    local dry_run=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run) dry_run=true; shift ;;
            --*) shift ;;
            *) shift ;;
        esac
    done

    # Load configs
    local config_file="${SCRIPT_DIR}/configs/fair_comparison.sh"
    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: ${config_file}"
        exit 1
    fi
    source "$config_file"

    echo "========================================================"
    echo "  Data-Efficiency Fair Comparison (Multi-Seed)"
    echo "========================================================"
    echo "Task:              ${ENV_ID}"
    echo "Seeds:             ${SEED_LIST[*]} (${NUM_SEEDS} seeds)"
    echo "Scheme:            ${COMPARISON_SCHEME}"
    echo "Checkpoint:        ${CHECKPOINT}"
    echo "Demo:              ${DEMO_PATH}"
    echo "num_envs:          ${NUM_ENVS} | batch_size: ${BATCH_SIZE}"
    echo "eval_freq:         AWSC=${EVAL_FREQ_AWSC}  PLD=${EVAL_FREQ_PLD}  DSRL=${EVAL_FREQ_DSRL}"
    echo "Steps:             AWSC=${TOTAL_STEPS_AWSC}  PLD=${TOTAL_STEPS_PLD}  DSRL=${TOTAL_STEPS_DSRL}"
    echo "GPUs:              ${AVAILABLE_GPUS[*]} (${NUM_GPUS} total)"
    echo "Sweep dir:         ${SWEEP_BASE_DIR}"
    echo "WandB:             ${USE_WANDB}"
    echo "========================================================"
    echo ""
    echo "  ┌────────┬──────────┬────────┬────────┬──────────┬──────────┐"
    echo "  │ Algo   │ Steps    │ UTD    │ num_qs │ Key Param│ EvalFreq │"
    echo "  ├────────┼──────────┼────────┼────────┼──────────┼──────────┤"
    echo "  │ AWSC   │ $(printf '%-8s' ${TOTAL_STEPS_AWSC}) │ 20     │ 10     │ β=50     │ $(printf '%-8s' ${EVAL_FREQ_AWSC}) │"
    echo "  │ PLD    │ $(printf '%-8s' ${TOTAL_STEPS_PLD}) │ 60     │ 5      │ ξ=0.3    │ $(printf '%-8s' ${EVAL_FREQ_PLD}) │"
    echo "  │ DSRL   │ $(printf '%-8s' ${TOTAL_STEPS_DSRL}) │ 60     │ 10     │ mag=2.5  │ $(printf '%-8s' ${EVAL_FREQ_DSRL}) │"
    echo "  └────────┴──────────┴────────┴────────┴──────────┴──────────┘"
    echo ""
    echo "  Total experiments: ${NUM_SEEDS} seeds × 3 algos = $((NUM_SEEDS * 3))"
    echo ""

    # Dry-run mode
    if [[ "$dry_run" == "true" ]]; then
        log_info "=== DRY RUN — Commands that would be executed ==="
        echo ""
        for seed in "${SEED_LIST[@]}"; do
            for algo in "${ALL_ALGORITHMS[@]}"; do
                local -n configs="${algo^^}_CONFIGS"
                for config in "${configs[@]}"; do
                    local config_name
                    config_name=$(echo "$config" | cut -d':' -f1)
                    local extra_args
                    extra_args=$(echo "$config" | cut -d':' -f2-)
                    if [[ "$config_name" == "$extra_args" ]]; then
                        extra_args=""
                    fi
                    local seed_config="${config_name}_s${seed}"

                    echo "# ${algo} / ${seed_config}  (seed=${seed})"
                    SEED="$seed" build_train_command "GPU" "$algo" "$seed_config" "$extra_args"
                    echo ""
                done
            done
        done
        return 0
    fi

    # Validate prerequisites
    if ! check_checkpoint; then exit 1; fi
    if ! check_demo_file; then exit 1; fi

    cd "$PROJECT_ROOT"

    # Build global queue: algorithm|config_name|extra_args
    # 策略：PLD/DSRL 快 (~15-30 min)，先跑，腾出 GPU 给 AWSC (~2h)
    local fast_items=()
    local slow_items=()

    for seed in "${SEED_LIST[@]}"; do
        for algo in "${ALL_ALGORITHMS[@]}"; do
            local -n configs="${algo^^}_CONFIGS"
            for config in "${configs[@]}"; do
                local config_name
                config_name=$(echo "$config" | cut -d':' -f1)
                local extra_args
                extra_args=$(echo "$config" | cut -d':' -f2-)
                if [[ "$config_name" == "$extra_args" ]]; then
                    extra_args=""
                fi
                # Encode seed into config_name so each seed gets its own directory
                local seed_config="${config_name}_s${seed}"
                # Also pass --seed via extra_args override
                local seed_args="${extra_args} --seed ${seed}"

                if [[ "$algo" == "awsc" ]]; then
                    slow_items+=("${algo}|${seed_config}|${seed_args}")
                else
                    fast_items+=("${algo}|${seed_config}|${seed_args}")
                fi
            done
        done
    done

    # Merge: fast (PLD/DSRL) first, then slow (AWSC)
    local sweep_items=("${fast_items[@]}" "${slow_items[@]}")

    log_info "Starting multi-seed comparison (${#sweep_items[@]} experiments, queue mode, fast-first)..."
    log_info "  Fast (PLD+DSRL): ${#fast_items[@]} experiments"
    log_info "  Slow (AWSC):     ${#slow_items[@]} experiments"
    echo ""
    run_sweep_queue "${sweep_items[@]}"

    log_success "Multi-seed comparison completed! Run '$0 report' to generate analysis."
}

# =============================================================================
# Command: retry
# =============================================================================
cmd_retry() {
    echo "========================================"
    echo "Fair Comparison — Retry Failed"
    echo "========================================"

    if ! check_checkpoint; then exit 1; fi
    if ! check_demo_file; then exit 1; fi

    cd "$PROJECT_ROOT"

    local config_file="${SCRIPT_DIR}/configs/fair_comparison.sh"
    source "$config_file"

    local retry_items=()
    for algo in "${ALL_ALGORITHMS[@]}"; do
        local -n configs="${algo^^}_CONFIGS"
        SWEEP_CONFIGS=("${configs[@]}")

        for config in "${SWEEP_CONFIGS[@]}"; do
            local config_name
            config_name=$(echo "$config" | cut -d':' -f1)
            local extra_args
            extra_args=$(echo "$config" | cut -d':' -f2-)
            if [[ "$config_name" == "$extra_args" ]]; then
                extra_args=""
            fi

            local exp_dir
            exp_dir=$(find_actual_exp_dir "$algo" "$config_name")

            if is_experiment_failed "$exp_dir"; then
                retry_items+=("${algo}|${config_name}|${extra_args}")
                echo "  - ${algo}/${config_name}"
            fi
        done
    done

    if [[ ${#retry_items[@]} -eq 0 ]]; then
        log_success "No failed experiments to retry"
        return 0
    fi

    log_info "Found ${#retry_items[@]} failed experiments"
    echo ""
    read -p "Retry these experiments? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log_info "Aborted"
        return 0
    fi

    run_sweep_queue "${retry_items[@]}"
    log_success "Retry completed!"
}

# =============================================================================
# Command: status  (multi-seed aware)
# =============================================================================
cmd_status() {
    local config_file="${SCRIPT_DIR}/configs/fair_comparison.sh"
    source "$config_file"

    echo "========================================"
    echo "Fair Comparison — Status (Multi-Seed)"
    echo "========================================"
    echo "Sweep dir: ${SWEEP_BASE_DIR}"
    echo "Seeds: ${SEED_LIST[*]} | Scheme: ${COMPARISON_SCHEME}"
    echo ""

    for algo in "${ALL_ALGORITHMS[@]}"; do
        local -n configs="${algo^^}_CONFIGS"
        echo "──── ${algo^^} ────"
        local total=0 success=0 failed=0 not_started=0

        for seed in "${SEED_LIST[@]}"; do
            for config in "${configs[@]}"; do
                local config_name
                config_name=$(echo "$config" | cut -d':' -f1)
                local seed_config="${config_name}_s${seed}"

                SEED="$seed" SWEEP_CONFIGS=("${seed_config}:") \
                    find_actual_exp_dir "$algo" "$seed_config" >/dev/null 2>&1

                local exp_dir
                exp_dir=$(find_actual_exp_dir "$algo" "$seed_config")

                total=$((total + 1))
                if is_experiment_successful "$exp_dir"; then
                    local sr
                    sr=$(parse_best_success_rate "${exp_dir}/train.log" 2>/dev/null)
                    echo -e "  ${GREEN}✓${NC} seed=${seed}  success_once=${sr:-?}"
                    success=$((success + 1))
                elif is_experiment_failed "$exp_dir"; then
                    echo -e "  ${RED}✗${NC} seed=${seed}  (failed)"
                    failed=$((failed + 1))
                else
                    echo -e "  ${YELLOW}○${NC} seed=${seed}  (not started / running)"
                    not_started=$((not_started + 1))
                fi
            done
        done
        echo "  Total: ${total} | ✓ ${success} | ✗ ${failed} | ○ ${not_started}"
        echo ""
    done
}

# =============================================================================
# Command: analyze
# =============================================================================
cmd_analyze() {
    local config_file="${SCRIPT_DIR}/configs/fair_comparison.sh"
    source "$config_file"

    echo "========================================"
    echo "Fair Comparison — Detailed Analysis"
    echo "========================================"
    echo "Sweep dir: ${SWEEP_BASE_DIR}"
    echo "Seed: ${SEED} | Scheme: ${COMPARISON_SCHEME}"
    echo ""

    for algo in "${ALL_ALGORITHMS[@]}"; do
        local -n configs="${algo^^}_CONFIGS"
        SWEEP_CONFIGS=("${configs[@]}")
        analyze_algorithm_with_metrics "$algo"
    done
}

# =============================================================================
# Command: report
# =============================================================================
cmd_report() {
    local config_file="${SCRIPT_DIR}/configs/fair_comparison.sh"
    source "$config_file"

    echo "========================================"
    echo "Fair Comparison — Report Generation"
    echo "========================================"

    # 1. Export JSON results
    local json_file="${SWEEP_BASE_DIR}/comparison_results.json"
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
            --comparison-scheme "${COMPARISON_SCHEME}" \
            --act-horizon "${ACT_HORIZON}" \
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

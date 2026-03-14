#!/bin/bash
# Dense checkpoint evaluation of WM v5 (iter1_v5) across GPUs 2-9
# Evaluates all 20 checkpoints (step 200 to 4000) in 3 rounds of 8 parallel jobs
# Usage: bash scripts/vlaw/run/eval_wm_v5_checkpoints.sh [--round 1|2|3|all]

set -euo pipefail

CONDA_BASE="$HOME/miniconda3"
PYTHON="$CONDA_BASE/envs/rlft_ms3/bin/python"
WD="/home/wjz/rl-vla"
CKPT_BASE="$WD/checkpoints/vlaw/world_model/iter1_v5"
OUT_BASE="$WD/data/vlaw/synthetic"
LOG_BASE="$WD/logs/vlaw"
SCRIPT="$WD/rlft/vlaw/scripts/run_imagination.py"
PID_FILE="$LOG_BASE/v5_eval_pids.txt"

export PYTHONPATH="$WD:$WD/ctrl_world"
cd "$WD"
mkdir -p "$LOG_BASE"

GPUS=(2 3 4 5 6 7 8 9)
NUM_TRAJS=20
VIS_COUNT=5
SEED=42

# Round definitions: checkpoint steps
ROUND1_STEPS=(200 400 600 800 1000 1200 1400 1600)
ROUND2_STEPS=(1800 2000 2200 2400 2600 2800 3000 3200)
ROUND3_STEPS=(3400 3600 3800 4000)

ROUND="${1:---round}"
ROUND_NUM="${2:-all}"
# Also accept --round=N format
if [[ "$ROUND" == "--round" ]]; then
    true  # ROUND_NUM already set
elif [[ "$ROUND" =~ ^--round=(.+)$ ]]; then
    ROUND_NUM="${BASH_REMATCH[1]}"
else
    ROUND_NUM="$ROUND"
fi

launch_round() {
    local round_id=$1
    shift
    local steps=("$@")
    local num_jobs=${#steps[@]}
    local pids=()

    echo "=========================================="
    echo "Round $round_id: Launching $num_jobs jobs"
    echo "Steps: ${steps[*]}"
    echo "=========================================="

    echo "# Round $round_id — $(date)" >> "$PID_FILE"

    for i in "${!steps[@]}"; do
        local step=${steps[$i]}
        local gpu=${GPUS[$i]}
        local out_dir="$OUT_BASE/v5_eval_step${step}"
        local log_file="$LOG_BASE/v5_eval_step${step}.log"

        CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" "$SCRIPT" \
            --wm_ckpt "$CKPT_BASE/checkpoint-${step}.pt" \
            --num_trajs $NUM_TRAJS \
            --output_dir "$out_dir" \
            --gpu_id 0 \
            --visualize --vis_count $VIS_COUNT \
            --seed $SEED \
            > "$log_file" 2>&1 &
        local pid=$!
        pids+=($pid)
        echo "  Job step-${step} on GPU ${gpu}: PID=${pid}"
        echo "step-${step} GPU=${gpu} PID=${pid}" >> "$PID_FILE"
    done

    echo ""
    echo "Waiting for round $round_id to complete (${num_jobs} jobs)..."
    local failed=0
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local step=${steps[$i]}
        if wait "$pid"; then
            echo "  [OK] step-${step} (PID ${pid}) completed"
        else
            echo "  [FAIL] step-${step} (PID ${pid}) exit code $?"
            failed=$((failed + 1))
        fi
    done

    if [ $failed -gt 0 ]; then
        echo "WARNING: $failed job(s) failed in round $round_id"
    else
        echo "Round $round_id: All $num_jobs jobs completed successfully"
    fi
    echo ""
}

# Clear PID file
echo "# WM v5 Imagination Eval — $(date)" > "$PID_FILE"

case "$ROUND_NUM" in
    1)
        launch_round 1 "${ROUND1_STEPS[@]}"
        ;;
    2)
        launch_round 2 "${ROUND2_STEPS[@]}"
        ;;
    3)
        launch_round 3 "${ROUND3_STEPS[@]}"
        ;;
    all)
        launch_round 1 "${ROUND1_STEPS[@]}"
        launch_round 2 "${ROUND2_STEPS[@]}"
        launch_round 3 "${ROUND3_STEPS[@]}"
        echo "=========================================="
        echo "All 3 rounds complete. Check outputs:"
        echo "  ls $OUT_BASE/v5_eval_step*/"
        echo "  ls $OUT_BASE/v5_eval_step*/viz/"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [--round 1|2|3|all]"
        echo "  Default: all (run rounds 1-3 sequentially)"
        exit 1
        ;;
esac

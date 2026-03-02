#!/bin/bash
# =============================================================================
# T-BC-SCALING: Demo Scaling Curve for ShortCut Flow (LiftPegUpright-v1)
#
# Train ShortCut Flow from scratch with different numbers of demos to 
# determine the data saturation point.
#
# Experiment Matrix:
#   - Demo counts: 25, 50, 100, 200, 400, 669 (all)
#   - 5000 training steps each (rough trend, not full convergence)
#   - Eval every 2500 steps (at 2500 and 5000)
#   - GPU 8 (sequential)
#
# Data: Raw ManiSkill official demos (669 trajectories, all successful)
#   ~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5
#
# Usage:
#   tmux new -s bc_scaling
#   conda activate rlft_ms3
#   bash scripts/vlaw/run_bc_scaling.sh [resume_from_ndemos]
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=8

# --- Configuration ---
DEMO_PATH="$HOME/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
ENV_ID="LiftPegUpright-v1"
ALGORITHM="shortcut_flow"
TOTAL_ITERS=5000
EVAL_FREQ=2500
SAVE_FREQ=5000
BATCH_SIZE=256
LR=3e-4
NUM_EVAL_EPISODES=100
NUM_EVAL_ENVS=25
WANDB_PROJECT="BC_Scaling_LiftPeg"
SEED=1

# Demo counts to test
DEMO_COUNTS=(25 50 100 200 400 669)

# Resume support
START_FROM=${1:-0}

# Output directories
RESULTS_DIR="/home/wjz/rl-vla/results/vlaw/bc_scaling"
LOG_DIR="/home/wjz/rl-vla/logs/vlaw/bc_scaling"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Results tracking (JSONL)
RESULTS_JSONL="$RESULTS_DIR/scaling_results.jsonl"

echo "=============================================="
echo "T-BC-SCALING: Demo Scaling Curve"
echo "=============================================="
echo "Demo counts: ${DEMO_COUNTS[*]}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Total iters: $TOTAL_ITERS"
echo "Eval freq: $EVAL_FREQ"
echo "Resume from: $START_FROM"
echo "Start time: $(date)"
echo ""

for NUM_DEMOS in "${DEMO_COUNTS[@]}"; do
    # Skip if resuming
    if [ "$NUM_DEMOS" -lt "$START_FROM" ]; then
        echo "[SKIP] $NUM_DEMOS demos (resume from $START_FROM)"
        continue
    fi

    EXP_NAME="bc_scaling_${NUM_DEMOS}demos_${TOTAL_ITERS}steps"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

    echo ""
    echo "=============================================="
    echo "[$(date +%H:%M:%S)] Training with $NUM_DEMOS demos"
    echo "=============================================="

    python -m rlft.offline.train_maniskill \
        --env_id "$ENV_ID" \
        --demo_path "$DEMO_PATH" \
        --algorithm "$ALGORITHM" \
        --num_demos "$NUM_DEMOS" \
        --total_iters "$TOTAL_ITERS" \
        --eval_freq "$EVAL_FREQ" \
        --save_freq "$SAVE_FREQ" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --num_eval_episodes "$NUM_EVAL_EPISODES" \
        --num_eval_envs "$NUM_EVAL_ENVS" \
        --seed "$SEED" \
        --exp_name "$EXP_NAME" \
        --track \
        --wandb_project_name "$WANDB_PROJECT" \
        --obs_mode rgb \
        --pred_horizon 8 \
        --obs_horizon 2 \
        --sc_fixed_step_size 0.15 \
        --sc_num_inference_steps 8 \
        --sc_self_consistency_k 0.25 \
        --sc_step_size_mode fixed \
        --ema_decay 0.9995 \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    # Extract eval success_once from log (last occurrence)
    BEST_SUCCESS=$(grep -oP "success_once.*?([0-9]+\.[0-9]+)" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+\.[0-9]+$' || echo "N/A")

    # Append to JSONL results
    echo "{\"n_demos\": $NUM_DEMOS, \"best_success_once\": \"$BEST_SUCCESS\", \"exit_code\": $EXIT_CODE, \"timestamp\": \"$(date +%Y%m%d_%H%M%S)\"}" >> "$RESULTS_JSONL"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE: $NUM_DEMOS demos -> success_once=$BEST_SUCCESS"
    else
        echo "[$(date +%H:%M:%S)] FAIL: $NUM_DEMOS demos (exit=$EXIT_CODE)"
    fi
done

echo ""
echo "=============================================="
echo "All experiments complete! $(date)"
echo "=============================================="

# --- Collect and plot results ---
echo ""
echo "=== Results Summary ==="
if [ -f "$RESULTS_JSONL" ]; then
    cat "$RESULTS_JSONL"
fi

python3 - << 'PLOT_SCRIPT'
import json
import os

results_file = "/home/wjz/rl-vla/results/vlaw/bc_scaling/scaling_results.jsonl"
if not os.path.exists(results_file):
    print("No results file found")
    exit(0)

results = []
with open(results_file) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

print("\n=== BC Scaling Curve Summary ===")
print(f"{'N Demos':>8} | {'Success Once':>12} | {'Status':>8}")
print("-" * 35)
for r in sorted(results, key=lambda x: x['n_demos']):
    status = "OK" if r.get('exit_code', 1) == 0 else "FAIL"
    print(f"{r['n_demos']:>8} | {r['best_success_once']:>12} | {status:>8}")

# Try to make a plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    demos = [r['n_demos'] for r in results if r['best_success_once'] != 'N/A']
    success = [float(r['best_success_once']) for r in results if r['best_success_once'] != 'N/A']

    if demos:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(demos, success, 'bo-', markersize=8, linewidth=2)
        ax.set_xlabel('Number of Expert Demos', fontsize=12)
        ax.set_ylabel('Success Once Rate', fontsize=12)
        ax.set_title('BC Scaling Curve: ShortCut Flow on LiftPegUpright-v1', fontsize=13)
        ax.set_xscale('log')
        ax.set_xticks(demos)
        ax.set_xticklabels([str(d) for d in demos])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        for x, y in zip(demos, success):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.tight_layout()
        plt.savefig('/home/wjz/rl-vla/results/vlaw/bc_scaling/scaling_curve.png', dpi=150)
        print(f"\nPlot saved to /home/wjz/rl-vla/results/vlaw/bc_scaling/scaling_curve.png")
except Exception as e:
    print(f"Could not generate plot: {e}")
PLOT_SCRIPT

echo "Done!"

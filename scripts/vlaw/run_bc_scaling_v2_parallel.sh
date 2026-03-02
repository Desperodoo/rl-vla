#!/bin/bash
# =============================================================================
# T-BC-SCALING-V2: Parallel orchestrator
#
# Runs 6 experiments (25, 50, 100, 200, 400, 669 demos) in 3 rounds
# of 2 parallel jobs on GPU 0 and GPU 9.
#
# Each experiment: 20K steps ShortCut Flow, eval 50 episodes
#
# Usage:
#   tmux new -s bc_scaling_v2
#   conda activate rlft_ms3
#   bash scripts/vlaw/run_bc_scaling_v2_parallel.sh
# =============================================================================
set -e

cd /home/wjz/rl-vla

SCRIPT="scripts/vlaw/run_bc_scaling_v2_single.sh"
GPU_A=0
GPU_B=9
LOG_DIR="/home/wjz/rl-vla/logs/vlaw/bc_scaling_v2"
RESULTS_DIR="/home/wjz/rl-vla/results/vlaw/bc_scaling_v2"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Define 3 rounds of (demo_count_gpu0, demo_count_gpu9)
ROUND1_A=25;  ROUND1_B=50
ROUND2_A=100; ROUND2_B=200
ROUND3_A=400; ROUND3_B=669

echo "=============================================="
echo "T-BC-SCALING-V2: Parallel Training (3 rounds)"
echo "=============================================="
echo "GPU A: $GPU_A    GPU B: $GPU_B"
echo "Round 1: ${ROUND1_A} + ${ROUND1_B} demos"
echo "Round 2: ${ROUND2_A} + ${ROUND2_B} demos"
echo "Round 3: ${ROUND3_A} + ${ROUND3_B} demos"
echo "Start: $(date)"
echo ""

run_round() {
    local round_num=$1
    local demos_a=$2
    local demos_b=$3
    
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  Round $round_num: $demos_a demos (GPU $GPU_A) + $demos_b demos (GPU $GPU_B)"
    echo "  Start: $(date)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ""
    
    # Run both in parallel
    bash "$SCRIPT" "$demos_a" "$GPU_A" &
    PID_A=$!
    
    bash "$SCRIPT" "$demos_b" "$GPU_B" &
    PID_B=$!
    
    # Wait for both
    FAIL=0
    wait $PID_A || { echo "[WARN] GPU $GPU_A ($demos_a demos) exited with error"; FAIL=1; }
    wait $PID_B || { echo "[WARN] GPU $GPU_B ($demos_b demos) exited with error"; FAIL=1; }
    
    echo ""
    echo "  Round $round_num complete: $(date)"
    if [ $FAIL -ne 0 ]; then
        echo "  [WARN] Some jobs in round $round_num failed"
    fi
    echo ""
}

# Execute 3 rounds sequentially (2 parallel per round)
run_round 1 $ROUND1_A $ROUND1_B
run_round 2 $ROUND2_A $ROUND2_B
run_round 3 $ROUND3_A $ROUND3_B

echo ""
echo "=============================================="
echo "All 3 rounds complete! $(date)"
echo "=============================================="

# --- Collect results ---
echo ""
echo "=== Results Summary ==="
if [ -f "$RESULTS_DIR/scaling_results.jsonl" ]; then
    cat "$RESULTS_DIR/scaling_results.jsonl"
fi

# Generate summary table and JSON
python3 - << 'COLLECT_SCRIPT'
import json
import os

results_file = "/home/wjz/rl-vla/results/vlaw/bc_scaling_v2/scaling_results.jsonl"
output_json = "/home/wjz/rl-vla/results/vlaw/bc_scaling_v2/bc_scaling_v2_summary.json"

if not os.path.exists(results_file):
    print("No results file found")
    exit(0)

results = []
with open(results_file) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

# Deduplicate: keep latest per n_demos
best = {}
for r in results:
    nd = r["n_demos"]
    if nd not in best or r["timestamp"] > best[nd]["timestamp"]:
        best[nd] = r

results = sorted(best.values(), key=lambda x: x["n_demos"])

print("\n=== T-BC-SCALING-V2: Demo Scaling Curve (20K steps) ===")
print(f"{'N Demos':>8} | {'Success Once':>12} | {'Success@End':>12} | {'Status':>8}")
print("-" * 50)
for r in results:
    status = "OK" if r.get("exit_code", 1) == 0 else "FAIL"
    print(f"{r['n_demos']:>8} | {r['best_success_once']:>12} | {r['best_success_at_end']:>12} | {status:>8}")

# Save JSON
summary = {
    "experiment": "T-BC-SCALING-V2",
    "task": "LiftPegUpright-v1",
    "total_iters": 20000,
    "eval_episodes": 50,
    "algorithm": "shortcut_flow",
    "results": {str(r["n_demos"]): {
        "success_once": r["best_success_once"],
        "success_at_end": r["best_success_at_end"],
        "exit_code": r["exit_code"],
        "timestamp": r["timestamp"],
    } for r in results}
}

with open(output_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nJSON summary saved to {output_json}")

# Generate markdown table
md_file = "/home/wjz/rl-vla/results/vlaw/bc_scaling_v2/bc_scaling_v2_table.md"
with open(md_file, "w") as f:
    f.write("# T-BC-SCALING-V2: Demo Scaling Curve (20K Steps)\n\n")
    f.write(f"Task: LiftPegUpright-v1 | Algorithm: ShortCut Flow | Eval: 50 episodes\n\n")
    f.write("| Demo Count | Success Once | Success@End |\n")
    f.write("|:----------:|:------------:|:-----------:|\n")
    for r in results:
        f.write(f"| {r['n_demos']} | {r['best_success_once']} | {r['best_success_at_end']} |\n")
print(f"Markdown table saved to {md_file}")

# Try plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demos = [r["n_demos"] for r in results if r["best_success_once"] != "N/A"]
    succ = [float(r["best_success_once"]) for r in results if r["best_success_once"] != "N/A"]

    if demos:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(demos, succ, "bo-", markersize=8, linewidth=2)
        ax.set_xlabel("Number of Expert Demos", fontsize=12)
        ax.set_ylabel("Success Once Rate", fontsize=12)
        ax.set_title("BC Scaling V2 (20K Steps): ShortCut Flow on LiftPegUpright-v1", fontsize=13)
        ax.set_xscale("log")
        ax.set_xticks(demos)
        ax.set_xticklabels([str(d) for d in demos])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        for x, y in zip(demos, succ):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")
        plt.tight_layout()
        plot_path = "/home/wjz/rl-vla/results/vlaw/bc_scaling_v2/scaling_curve_v2.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
except Exception as e:
    print(f"Could not generate plot: {e}")

COLLECT_SCRIPT

echo ""
echo "Done! All results in: $RESULTS_DIR"

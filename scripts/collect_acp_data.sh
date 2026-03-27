#!/bin/bash
# ============================================================================
# collect_acp_data.sh — Multi-distribution ACP raw-fps data collection
#
# Collects ACP rollouts under 4 data distributions. Output goes to
# data/vlaw/rollouts_acp/*_rawfps so it stays isolated from WM/VLAW rollouts.
#
# Usage:
#   bash scripts/collect_acp_data.sh           # all distributions, GPU 2-5
#   bash scripts/collect_acp_data.sh --type b  # only Type B on GPU 2
#   bash scripts/collect_acp_data.sh --dry-run # dry run (5 episodes each)
#
# Environment: rlft_ms3
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

CHECKPOINT="runs/fair_comparison/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt"
ENV_ID="LiftPegUpright-v1"
NUM_ENVS=32
MAX_EPISODE_STEPS=200
SAVE_EVERY_N_STEPS=1

N_B=200
N_C=200
N_D=200
N_E=100

GPU_B=2
GPU_C=3
GPU_D=4
GPU_E=5

TYPE="all"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --type) TYPE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ $DRY_RUN -eq 1 ]]; then
    N_B=5; N_C=5; N_D=5; N_E=5
    echo "[collect_acp_data] DRY-RUN mode: 5 episodes per type"
fi

collect() {
    local mode="$1" gpu="$2" n="$3" outdir="$4" extra="${5:-}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[Type ${mode^^}] GPU=${gpu}  episodes=${n}  → ${outdir}"
    CUDA_VISIBLE_DEVICES=${gpu} conda run -n rlft_ms3 python scripts/collect_acp_data.py \
        --noise_mode "${mode}" \
        --checkpoint_path "${CHECKPOINT}" \
        --env_id "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --num_episodes "${n}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --gpu_id "${gpu}" \
        --output_dir "${outdir}" \
        ${extra} \
        &
    echo "[Type ${mode^^}] PID=$!"
}

PIDS=()

if [[ "$TYPE" == "all" || "$TYPE" == "b" ]]; then
    collect none ${GPU_B} ${N_B} "data/vlaw/rollouts_acp/pretrained_policy_rawfps" ""
    PIDS+=($!)
fi

if [[ "$TYPE" == "all" || "$TYPE" == "c" ]]; then
    collect teleop ${GPU_C} ${N_C} "data/vlaw/rollouts_acp/teleop_sim_rawfps" \
        "--ou_sigma 0.07 --pause_prob 0.04"
    PIDS+=($!)
fi

if [[ "$TYPE" == "all" || "$TYPE" == "d" ]]; then
    collect rl_explore ${GPU_D} ${N_D} "data/vlaw/rollouts_acp/rl_prior_rawfps" \
        "--explore_sigma 0.25"
    PIDS+=($!)
fi

if [[ "$TYPE" == "all" || "$TYPE" == "e" ]]; then
    collect random ${GPU_E} ${N_E} "data/vlaw/rollouts_acp/random_rawfps" ""
    PIDS+=($!)
fi

FAILED=0
for pid in "${PIDS[@]:-}"; do
    if wait "${pid}"; then
        echo "[collect_acp_data] PID ${pid} done ✓"
    else
        echo "[collect_acp_data] PID ${pid} FAILED ✗"
        FAILED=1
    fi
done

if [[ $FAILED -ne 0 ]]; then
    echo "[collect_acp_data] One or more collection jobs failed. Check logs above."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[collect_acp_data] ALL DONE. Data summary:"
for d in data/vlaw/rollouts_acp/*/; do
    N=$(ls "${d}"*.h5 2>/dev/null | wc -l || echo 0)
    echo "  ${d}  →  ${N} HDF5 file(s)"
done
echo ""
echo "Next step: validate lengths, then plan ACP retraining"

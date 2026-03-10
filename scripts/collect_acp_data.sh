#!/bin/bash
# ============================================================================
# collect_acp_data.sh — Multi-distribution ACP training data collection
#
# Collects rollouts under 4 data distributions in parallel, one GPU each.
# Expected run time: ~30-60 min per distribution (200 episodes, 32 envs).
#
# Data layout after completion:
#   data/vlaw/rollouts/
#     mixed/                ← expert demos (already present, Type A)
#     pretrained_policy/    ← Type B: clean AWSC policy
#     teleop_sim/           ← Type C: OU-noise teleop simulation
#     rl_prior/             ← Type D: Gaussian noise RL exploration
#     random/               ← Type E: pure random (ablation)
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

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT="runs/fair_comparison/awsc/best_s42__1772570560/checkpoints/best.pt"
ENV_ID="LiftPegUpright-v1"
NUM_ENVS=32
MAX_EPISODE_STEPS=100
FRAME_SKIP=3

# Episodes per distribution (reduce for dry-run)
N_B=200   # Type B: pretrained policy
N_C=200   # Type C: teleop simulation
N_D=200   # Type D: RL exploration prior
N_E=100   # Type E: random (ablation)

# GPU assignments (adjust if some are busy)
GPU_B=2
GPU_C=3
GPU_D=4
GPU_E=5

# ── Parse arguments ──────────────────────────────────────────────────────────
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

# ── Helper: launch one collection job ────────────────────────────────────────
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
        --frame_skip "${FRAME_SKIP}" \
        --gpu_id "${gpu}" \
        --output_dir "${outdir}" \
        ${extra} \
        &
    echo "[Type ${mode^^}] PID=$!"
}

# ── Launch jobs based on --type ───────────────────────────────────────────────
PIDS=()

if [[ "$TYPE" == "all" || "$TYPE" == "b" ]]; then
    collect none   ${GPU_B} ${N_B} "data/vlaw/rollouts/pretrained_policy" ""
    PIDS+=($!)
fi

if [[ "$TYPE" == "all" || "$TYPE" == "c" ]]; then
    # Teleop: mild OU noise matching human motor characteristics
    collect teleop ${GPU_C} ${N_C} "data/vlaw/rollouts/teleop_sim" \
        "--ou_sigma 0.07 --pause_prob 0.04"
    PIDS+=($!)
fi

if [[ "$TYPE" == "all" || "$TYPE" == "d" ]]; then
    # RL exploration prior: moderate Gaussian noise
    collect rl_explore ${GPU_D} ${N_D} "data/vlaw/rollouts/rl_prior" \
        "--explore_sigma 0.25"
    PIDS+=($!)
fi

if [[ "$TYPE" == "all" || "$TYPE" == "e" ]]; then
    collect random ${GPU_E} ${N_E} "data/vlaw/rollouts/random" ""
    PIDS+=($!)
fi

# ── Wait for all jobs ─────────────────────────────────────────────────────────
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
for d in data/vlaw/rollouts/*/; do
    N=$(ls "${d}${ENV_ID}/"*.h5 2>/dev/null | wc -l || echo 0)
    echo "  ${d}${ENV_ID}/  →  ${N} HDF5 file(s)"
done
echo ""
echo "Next step: bash scripts/train_acp_multi.sh"

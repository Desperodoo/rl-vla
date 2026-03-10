#!/bin/bash
# ============================================================================
# train_acp_multi.sh — Train multiple ACP versions on different data distributions
#
# Trains 5 ACP value model variants to enable ablation comparisons:
#
#   v2_demo_only        (A)         expert demos only      — overfitting baseline
#   v2_pretrained_pol   (B)         clean policy rollouts
#   v2_teleop_sim       (C)         teleop-simulated (OU noise)
#   v2_rl_prior         (D)         RL exploration noise
#   v2_combined         (A+B+C+D)   ALL distributions    ← recommended for RLPD
#
# Prerequisites:
#   bash scripts/collect_acp_data.sh   (must complete first)
#
# Usage:
#   bash scripts/train_acp_multi.sh           # all 5 versions, sequential on GPU 6
#   bash scripts/train_acp_multi.sh --parallel # all 5 versions, parallel on GPU 2-6
#   bash scripts/train_acp_multi.sh --version combined  # only combined
#
# Environment: vlaw_reward
# ============================================================================
set -euo pipefail

cd /home/wjz/rl-vla
export PYTHONPATH=/home/wjz/rl-vla

# ── Common training hyperparams ─────────────────────────────────────────────
BATCH_SIZE=32
LR=5e-5
WARMUP=500
EVAL_INTERVAL=200
SAVE_INTERVAL=1000

# Steps per version (combined gets more since data is larger)
STEPS_SINGLE=8000
STEPS_COMBINED=12000

# ── Data directories ─────────────────────────────────────────────────────────
DIR_DEMO="data/vlaw/rollouts/mixed"
DIR_POL="data/vlaw/rollouts/pretrained_policy"
DIR_TELE="data/vlaw/rollouts/teleop_sim"
DIR_RL="data/vlaw/rollouts/rl_prior"
# Note: random rollouts intentionally excluded from combined — purely negative
# examples with near-zero value signal can hurt value estimation.

# ── Parse arguments ──────────────────────────────────────────────────────────
MODE="sequential"
VERSION="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) MODE="parallel"; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Helper: launch one training job ──────────────────────────────────────────
# Args: <gpu_id> <output_dir> <num_steps> <data_dirs...>
train_acp() {
    local gpu="$1"; local outdir="$2"; local steps="$3"
    shift 3
    local data_dirs=("$@")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[train_acp] ${outdir}"
    echo "            GPU=${gpu}  steps=${steps}  data=${data_dirs[*]}"

    CUDA_VISIBLE_DEVICES=${gpu} conda run -n rlft_ms3 \
        python rlft/vlaw/scripts/run_acp_train.py \
        --data_dirs "${data_dirs[@]}" \
        --output_dir "${outdir}" \
        --num_steps "${steps}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR}" \
        --warmup_steps "${WARMUP}" \
        --eval_interval "${EVAL_INTERVAL}" \
        --save_interval "${SAVE_INTERVAL}"
}

# ── Define versions (name, gpu, steps, data_dirs) ────────────────────────────
declare -A VERSIONS
# Format stored as "gpu|steps|dir1 dir2 ..."
VERSIONS["v2_demo_only"]="2|${STEPS_SINGLE}|${DIR_DEMO}"
VERSIONS["v2_pretrained_pol"]="3|${STEPS_SINGLE}|${DIR_POL}"
VERSIONS["v2_teleop_sim"]="4|${STEPS_SINGLE}|${DIR_TELE}"
VERSIONS["v2_rl_prior"]="5|${STEPS_SINGLE}|${DIR_RL}"
VERSIONS["v2_combined"]="6|${STEPS_COMBINED}|${DIR_DEMO} ${DIR_POL} ${DIR_TELE} ${DIR_RL}"

# For sequential mode, all on GPU 6
GPU_SEQUENTIAL=6

# ── Run ───────────────────────────────────────────────────────────────────────
PIDS=()

for vname in v2_demo_only v2_pretrained_pol v2_teleop_sim v2_rl_prior v2_combined; do
    if [[ "$VERSION" != "all" && "$VERSION" != "$vname" ]]; then
        echo "[train_acp_multi] Skipping ${vname} (--version=${VERSION})"
        continue
    fi

    IFS='|' read -r gpu steps dirs_str <<< "${VERSIONS[$vname]}"
    IFS=' ' read -ra dirs <<< "${dirs_str}"
    outdir="checkpoints/vlaw/acp/${vname}"

    if [[ "$MODE" == "parallel" ]]; then
        train_acp "${gpu}" "${outdir}" "${steps}" "${dirs[@]}" &
        PIDS+=($!)
        echo "[train_acp_multi] Launched ${vname} PID=$!"
    else
        echo ""
        echo ">>> Training ${vname}  (${vname} of 5)"
        train_acp "${GPU_SEQUENTIAL}" "${outdir}" "${steps}" "${dirs[@]}"
        echo ">>> ${vname} complete — best checkpoint: ${outdir}/best.safetensors"
    fi
done

# Wait for parallel jobs
if [[ "${#PIDS[@]}" -gt 0 ]]; then
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if wait "${pid}"; then
            echo "[train_acp_multi] PID ${pid} done ✓"
        else
            echo "[train_acp_multi] PID ${pid} FAILED ✗"
            FAILED=1
        fi
    done
    if [[ $FAILED -ne 0 ]]; then
        echo "[train_acp_multi] One or more training jobs failed."
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[train_acp_multi] ALL DONE. Checkpoint summary:"
for vname in v2_demo_only v2_pretrained_pol v2_teleop_sim v2_rl_prior v2_combined; do
    best="checkpoints/vlaw/acp/${vname}/best.safetensors"
    if [[ -f "$best" ]]; then
        echo "  ✓ ${vname}  →  ${best}"
    else
        echo "  ✗ ${vname}  →  NOT FOUND (maybe skipped)"
    fi
done
echo ""
echo "Recommended RLPD checkpoint: checkpoints/vlaw/acp/v2_combined/best.safetensors"
echo ""
echo "Next step:"
echo "  bash scripts/run_rlpd_sac_acp_v2.sh     # SAC + ACP v2_combined"
echo "  bash scripts/run_rlpd_awsc_acp.sh        # AWSC + pretrained + ACP v2_combined"

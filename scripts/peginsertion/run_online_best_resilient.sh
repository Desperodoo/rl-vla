#!/usr/bin/env bash
# Resilient PegInsertion online launcher.
# Keeps PLD/DSRL at the best comparison config while allowing RLPD SAC/AWSC
# to use smaller env/batch settings when RGB replay cache + eval is too large.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DEFAULT_ALGORITHMS=(sac awsc pld dsrl)
if [[ -n "${ALGORITHMS:-}" ]]; then
    IFS=',' read -ra REQUESTED <<< "${ALGORITHMS}"
else
    REQUESTED=("${DEFAULT_ALGORITHMS[@]}")
fi

GPU_IDS="${GPU_IDS:-1,5}"
IFS=',' read -ra ALL_GPUS <<< "${GPU_IDS}"

RLPD_GPU_IDS="${RLPD_GPU_IDS:-${ALL_GPUS[0]}}"
POLICY_GPU_IDS="${POLICY_GPU_IDS:-${GPU_IDS}}"

RLPD_NUM_ENVS="${RLPD_NUM_ENVS:-10}"
RLPD_NUM_EVAL_ENVS="${RLPD_NUM_EVAL_ENVS:-10}"
RLPD_NUM_EVAL_EPISODES="${RLPD_NUM_EVAL_EPISODES:-20}"
RLPD_BATCH_SIZE="${RLPD_BATCH_SIZE:-64}"
RLPD_MAX_RETRIES="${RLPD_MAX_RETRIES:-1}"
RLPD_OFFLINE_CACHE_SIZE="${RLPD_OFFLINE_CACHE_SIZE:-10000}"
RLPD_NUM_DEMOS="${RLPD_NUM_DEMOS:-}"
RLPD_NUM_SEED_STEPS="${RLPD_NUM_SEED_STEPS:-}"

join_csv() {
    local IFS=,
    echo "$*"
}

contains_rlpd=()
contains_policy=()
for algo in "${REQUESTED[@]}"; do
    case "${algo}" in
        sac|awsc) contains_rlpd+=("${algo}") ;;
        pld|dsrl) contains_policy+=("${algo}") ;;
        *) contains_policy+=("${algo}") ;;
    esac
done

status=0

if (( ${#contains_rlpd[@]} > 0 )); then
    echo "[online_resilient] RLPD algorithms: ${contains_rlpd[*]}"
    echo "[online_resilient] RLPD GPUs: ${RLPD_GPU_IDS}; envs=${RLPD_NUM_ENVS}/${RLPD_NUM_EVAL_ENVS}; batch=${RLPD_BATCH_SIZE}; cache=${RLPD_OFFLINE_CACHE_SIZE}; demos=${RLPD_NUM_DEMOS:-all}; seed_steps=${RLPD_NUM_SEED_STEPS:-default}"
    ALGORITHMS="$(join_csv "${contains_rlpd[@]}")" \
    GPU_IDS="${RLPD_GPU_IDS}" \
    RLFT_OFFLINE_CACHE_SIZE="${RLPD_OFFLINE_CACHE_SIZE}" \
    RLPD_NUM_DEMOS="${RLPD_NUM_DEMOS}" \
    RLPD_NUM_SEED_STEPS="${RLPD_NUM_SEED_STEPS}" \
    NUM_ENVS="${RLPD_NUM_ENVS}" \
    NUM_EVAL_ENVS="${RLPD_NUM_EVAL_ENVS}" \
    NUM_EVAL_EPISODES="${RLPD_NUM_EVAL_EPISODES}" \
    BATCH_SIZE="${RLPD_BATCH_SIZE}" \
    MAX_RETRIES="${RLPD_MAX_RETRIES}" \
    scripts/peginsertion/run_online_best.sh || status=1
fi

if (( ${#contains_policy[@]} > 0 )); then
    echo "[online_resilient] Policy algorithms: ${contains_policy[*]}"
    echo "[online_resilient] Policy GPUs: ${POLICY_GPU_IDS}"
    ALGORITHMS="$(join_csv "${contains_policy[@]}")" \
    GPU_IDS="${POLICY_GPU_IDS}" \
    scripts/peginsertion/run_online_best.sh || status=1
fi

exit "${status}"

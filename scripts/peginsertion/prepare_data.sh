#!/usr/bin/env bash
# Prepare PegInsertionSide-v1 replay data for rlft ManiSkill training.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_ID="${ENV_ID:-PegInsertionSide-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"
REWARD_MODE="${REWARD_MODE:-dense}"
NUM_ENVS="${NUM_ENVS:-64}"
CONDA_ENV="${CONDA_ENV:-rlft_ms3}"
FORCE_REPLAY="${FORCE_REPLAY:-0}"
ENABLE_CPU_FALLBACK="${ENABLE_CPU_FALLBACK:-1}"
CPU_FALLBACK_NUM_ENVS="${CPU_FALLBACK_NUM_ENVS:-1}"
MIN_TRAJS="${MIN_TRAJS:-10}"
REPLAY_COUNT="${REPLAY_COUNT:-}"

DEMO_DIR="${DEMO_DIR:-${HOME}/.maniskill/demos/${ENV_ID}/rl}"
RAW_TRAJ="${RAW_TRAJ:-${DEMO_DIR}/trajectory.h5}"
OUT_TRAJ="${OUT_TRAJ:-${DEMO_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5}"
FALLBACK_DEMO_DIR="${FALLBACK_DEMO_DIR:-${HOME}/.maniskill/demos/${ENV_ID}/motionplanning}"
FALLBACK_RAW_TRAJ="${FALLBACK_RAW_TRAJ:-${FALLBACK_DEMO_DIR}/trajectory.h5}"
FALLBACK_BACKEND="${FALLBACK_BACKEND:-physx_cpu}"
FALLBACK_OUT_TRAJ="${FALLBACK_OUT_TRAJ:-${FALLBACK_DEMO_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.${FALLBACK_BACKEND}.h5}"

log() { printf '[prepare_data] %s\n' "$*"; }
warn() { printf '[prepare_data][WARN] %s\n' "$*" >&2; }
die() { printf '[prepare_data][ERROR] %s\n' "$*" >&2; exit 1; }

init_env() {
    if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
        export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
        return
    fi

    for base in "${HOME}/anaconda3" "${HOME}/miniconda3" "/opt/conda"; do
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "${base}/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV}" || true
            break
        fi
    done
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
}

valid_traj() {
    local path=$1
    python - "$path" "${MIN_TRAJS}" <<'PY'
import sys
from pathlib import Path

import h5py

path = Path(sys.argv[1]).expanduser()
min_trajs = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)

try:
    with h5py.File(path, "r") as handle:
        traj_keys = [key for key in handle.keys() if key.startswith("traj_")]
        if len(traj_keys) < min_trajs:
            raise SystemExit(1)
        first = handle[traj_keys[0]]
        if "obs" not in first or "actions" not in first:
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}

run_replay() {
    local raw=$1
    local backend=$2
    local num_envs=$3
    local use_first=$4

    local use_first_arg="--use-first-env-state"
    [[ "${use_first}" == "1" ]] || use_first_arg="--no-use-first-env-state"

    local count_args=()
    if [[ -n "${REPLAY_COUNT}" ]]; then
        count_args=(--count "${REPLAY_COUNT}")
    fi

    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path "${raw}" \
        -o "${OBS_MODE}" \
        -c "${CONTROL_MODE}" \
        -b "${backend}" \
        -n "${num_envs}" \
        --record-rewards \
        --reward-mode "${REWARD_MODE}" \
        "${use_first_arg}" \
        "${count_args[@]}" \
        --save-traj
}

main() {
    cd "${ROOT}"
    init_env

    mkdir -p "${DEMO_DIR}"
    if [[ ! -f "${RAW_TRAJ}" ]]; then
        die "Raw trajectory not found: ${RAW_TRAJ}. Run: python -m mani_skill.utils.download_demo ${ENV_ID}"
    fi

    if [[ "${FORCE_REPLAY}" == "1" ]]; then
        rm -f "${OUT_TRAJ}" "${FALLBACK_OUT_TRAJ}"
    fi

    if [[ -f "${OUT_TRAJ}" && "${FORCE_REPLAY}" != "1" ]]; then
        if ! valid_traj "${OUT_TRAJ}"; then
            warn "Existing replay data is invalid, removing: ${OUT_TRAJ}"
            rm -f "${OUT_TRAJ}"
        else
            log "Found replay data: ${OUT_TRAJ}"
            exit 0
        fi
    fi

    if [[ -f "${FALLBACK_OUT_TRAJ}" && "${FORCE_REPLAY}" != "1" ]]; then
        if valid_traj "${FALLBACK_OUT_TRAJ}"; then
            ln -sf "${FALLBACK_OUT_TRAJ}" "${OUT_TRAJ}"
            log "Found replay data: ${OUT_TRAJ}"
            exit 0
        fi
        warn "Existing fallback replay data is invalid, removing: ${FALLBACK_OUT_TRAJ}"
        rm -f "${FALLBACK_OUT_TRAJ}"
    fi

    log "Replaying ${ENV_ID}"
    log "  raw:      ${RAW_TRAJ}"
    log "  output:   ${OUT_TRAJ}"
    log "  obs:      ${OBS_MODE}"
    log "  control:  ${CONTROL_MODE}"
    log "  backend:  ${SIM_BACKEND}"
    log "  envs:     ${NUM_ENVS}"

    if run_replay "${RAW_TRAJ}" "${SIM_BACKEND}" "${NUM_ENVS}" 1; then
        valid_traj "${OUT_TRAJ}" || die "Replay finished but expected output is invalid: ${OUT_TRAJ}"
        log "Done: ${OUT_TRAJ}"
        exit 0
    fi

    rm -f "${OUT_TRAJ}"
    [[ "${ENABLE_CPU_FALLBACK}" == "1" ]] || die "GPU replay failed and CPU fallback is disabled."
    [[ -f "${FALLBACK_RAW_TRAJ}" ]] || die "Fallback trajectory not found: ${FALLBACK_RAW_TRAJ}"

    warn "GPU replay failed. Falling back to CPU conversion from motionplanning demos."
    warn "This is needed when the RL demo action/state format cannot be converted to ${CONTROL_MODE} on GPU."
    log "  fallback raw:    ${FALLBACK_RAW_TRAJ}"
    log "  fallback output: ${FALLBACK_OUT_TRAJ}"

    run_replay "${FALLBACK_RAW_TRAJ}" "${FALLBACK_BACKEND}" "${CPU_FALLBACK_NUM_ENVS}" 0

    valid_traj "${FALLBACK_OUT_TRAJ}" || die "Fallback replay finished but output is invalid: ${FALLBACK_OUT_TRAJ}"
    ln -sf "${FALLBACK_OUT_TRAJ}" "${OUT_TRAJ}"

    log "Done: ${OUT_TRAJ}"
}

main "$@"

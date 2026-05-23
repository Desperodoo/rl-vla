#!/usr/bin/env bash
# Run best/default offline IL and offline RL configs on PegInsertionSide-v1.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_ID="${ENV_ID:-PegInsertionSide-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"
EXP_NAME="${EXP_NAME:-peginsertion_best1}"
SEED="${SEED:-42}"
CONDA_ENV="${CONDA_ENV:-rlft_ms3}"

DEMO_PATH="${DEMO_PATH:-${HOME}/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5}"
TOTAL_ITERS="${TOTAL_ITERS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-25}"
EVAL_FREQ="${EVAL_FREQ:-2500}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-10}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-${EXP_NAME}}"
GPU_IDS="${GPU_IDS:-1,5}"
FORCE="${FORCE:-0}"

DEFAULT_ALGORITHMS=(
    diffusion_policy
    flow_matching
    consistency_flow
    shortcut_flow
    reflected_flow
    cpql
    awcp
    aw_shortcut_flow
    sac
    dqc
)

if [[ -n "${ALGORITHMS:-}" ]]; then
    IFS=',' read -ra RUN_ALGORITHMS <<< "${ALGORITHMS}"
else
    RUN_ALGORITHMS=("${DEFAULT_ALGORITHMS[@]}")
fi
IFS=',' read -ra AVAILABLE_GPUS <<< "${GPU_IDS}"

log() { printf '[offline_best] %s\n' "$*"; }
warn() { printf '[offline_best][WARN] %s\n' "$*" >&2; }
err() { printf '[offline_best][ERROR] %s\n' "$*" >&2; }

init_env() {
    if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
        for base in "${HOME}/anaconda3" "${HOME}/miniconda3" "/opt/conda"; do
            if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
                # shellcheck disable=SC1091
                source "${base}/etc/profile.d/conda.sh"
                conda activate "${CONDA_ENV}" || true
                break
            fi
        done
    fi
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
    export RLFT_MANISKILL_CAMERA_NAMES="${RLFT_MANISKILL_CAMERA_NAMES:-base_camera}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

base_dir_for() {
    local algo=$1
    echo "${ROOT}/runs/${EXP_NAME}/${algo}/best"
}

actual_dir_for() {
    local algo=$1
    local parent="${ROOT}/runs/${EXP_NAME}/${algo}"
    find "${parent}" -maxdepth 1 -type d -name 'best__*' 2>/dev/null | sort -r | head -1
}

is_successful() {
    local dir=$1
    [[ -n "${dir}" ]] || return 1
    [[ -f "${dir}/checkpoints/best_eval_success_once.pt" ]] && return 0
    [[ -f "${dir}/checkpoints/best_eval_success_at_end.pt" ]] && return 0
    [[ -f "${dir}/checkpoints/final.pt" ]] && return 0
    find "${dir}/checkpoints" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | grep -q .
}

build_cmd() {
    local gpu=$1
    local algo=$2
    local track_arg="--track"
    [[ "${USE_WANDB}" == "true" ]] || track_arg="--no-track"

    local cmd=(
        env "CUDA_VISIBLE_DEVICES=${gpu}"
        python -m rlft.offline.train_maniskill
        --algorithm "${algo}"
        --env_id "${ENV_ID}"
        --demo_path "${DEMO_PATH}"
        --obs_mode "${OBS_MODE}"
        --control_mode "${CONTROL_MODE}"
        --sim_backend "${SIM_BACKEND}"
        --max_episode_steps "${MAX_EPISODE_STEPS}"
        --total_iters "${TOTAL_ITERS}"
        --batch_size "${BATCH_SIZE}"
        --num_eval_episodes "${NUM_EVAL_EPISODES}"
        --num_eval_envs "${NUM_EVAL_ENVS}"
        --eval_freq "${EVAL_FREQ}"
        --save_freq "${SAVE_FREQ}"
        --seed "${SEED}"
        --exp_name "${EXP_NAME}/${algo}/best"
        --wandb_project_name "${WANDB_PROJECT}"
        "${track_arg}"
    )

    printf '%q ' "${cmd[@]}"
}

run_one() {
    local gpu=$1
    local algo=$2
    local base_dir
    base_dir="$(base_dir_for "${algo}")"
    mkdir -p "${base_dir}"

    local actual_dir
    actual_dir="$(actual_dir_for "${algo}")"
    if [[ "${FORCE}" != "1" ]] && is_successful "${actual_dir}"; then
        log "Skip ${algo}: existing successful run ${actual_dir}"
        return 0
    fi

    local log_file="${base_dir}/train.log"
    local cmd
    cmd="$(build_cmd "${gpu}" "${algo}")"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        log "[GPU ${gpu}] ${cmd}"
        return 0
    fi

    local attempt=1
    while (( attempt <= MAX_RETRIES )); do
        if (( attempt > 1 )); then
            sleep "${RETRY_DELAY}"
            [[ -f "${log_file}" ]] && mv "${log_file}" "${log_file}.failed.$((attempt - 1))"
        fi
        log "[GPU ${gpu}] ${algo} attempt ${attempt}/${MAX_RETRIES}"
        eval "${cmd}" > "${log_file}" 2>&1 && {
            actual_dir="$(actual_dir_for "${algo}")"
            if is_successful "${actual_dir}"; then
                log "${algo} completed: ${actual_dir}"
                return 0
            fi
        }
        if ! grep -qE '(CUDA error|RuntimeError.*CUDA|illegal memory access|OutOfMemory|OOM|out of memory|PhysX Internal CUDA error)' "${log_file}" 2>/dev/null; then
            err "${algo} failed with non-retryable error. See ${log_file}"
            return 1
        fi
        warn "${algo} hit CUDA/OOM error; retrying"
        attempt=$((attempt + 1))
    done

    err "${algo} failed after ${MAX_RETRIES} attempts. See ${log_file}"
    return 1
}

run_queue() {
    local idx=0
    local completed=0
    local failed=0
    local total=${#RUN_ALGORITHMS[@]}
    local free_gpus=("${AVAILABLE_GPUS[@]}")
    declare -A pid_to_gpu=()

    log "Algorithms: ${RUN_ALGORITHMS[*]}"
    log "GPUs: ${AVAILABLE_GPUS[*]}"

    while (( idx < total || ${#pid_to_gpu[@]} > 0 )); do
        while (( idx < total && ${#free_gpus[@]} > 0 )); do
            local algo="${RUN_ALGORITHMS[$idx]}"
            local gpu="${free_gpus[0]}"
            free_gpus=("${free_gpus[@]:1}")
            run_one "${gpu}" "${algo}" &
            pid_to_gpu[$!]="${gpu}"
            idx=$((idx + 1))
        done

        for pid in "${!pid_to_gpu[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}" || failed=$((failed + 1))
                free_gpus+=("${pid_to_gpu[$pid]}")
                unset "pid_to_gpu[$pid]"
                completed=$((completed + 1))
            fi
        done

        if (( idx < total && ${#free_gpus[@]} == 0 )); then
            sleep 2
        fi
    done

    log "Queue complete: ${completed}/${total}, failed=${failed}"
    (( failed == 0 ))
}

main() {
    cd "${ROOT}"
    init_env
    [[ "${DRY_RUN:-0}" == "1" || -f "${DEMO_PATH}" ]] || {
        err "Demo file not found: ${DEMO_PATH}"
        err "Run scripts/peginsertion/prepare_data.sh first."
        exit 1
    }
    run_queue
}

main "$@"

#!/usr/bin/env bash
# Run best online configs on PegInsertionSide-v1 using a PegInsertion base policy.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_ID="${ENV_ID:-PegInsertionSide-v1}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
OBS_MODE="${OBS_MODE:-rgb}"
SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"
REWARD_MODE="${REWARD_MODE:-dense}"
EXP_NAME="${EXP_NAME:-peginsertion_best1}"
SEED="${SEED:-42}"
CONDA_ENV="${CONDA_ENV:-rlft_ms3}"

DEMO_PATH="${DEMO_PATH:-${HOME}/.maniskill/demos/${ENV_ID}/rl/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5}"
GPU_IDS="${GPU_IDS:-1,5}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-${EXP_NAME}-online}"
NUM_ENVS="${NUM_ENVS:-50}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-100}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
SAVE_FREQ="${SAVE_FREQ:-50000000}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-10}"

TOTAL_STEPS_SAC="${TOTAL_STEPS_SAC:-500000}"
TOTAL_STEPS_AWSC="${TOTAL_STEPS_AWSC:-500000}"
TOTAL_STEPS_PLD="${TOTAL_STEPS_PLD:-71000}"
TOTAL_STEPS_DSRL="${TOTAL_STEPS_DSRL:-71000}"
EVAL_FREQ_SAC="${EVAL_FREQ_SAC:-5000}"
EVAL_FREQ_AWSC="${EVAL_FREQ_AWSC:-5000}"
EVAL_FREQ_PLD="${EVAL_FREQ_PLD:-2000}"
EVAL_FREQ_DSRL="${EVAL_FREQ_DSRL:-2000}"

DEFAULT_ALGORITHMS=(sac awsc pld dsrl)
if [[ -n "${ALGORITHMS:-}" ]]; then
    IFS=',' read -ra RUN_ALGORITHMS <<< "${ALGORITHMS}"
else
    RUN_ALGORITHMS=("${DEFAULT_ALGORITHMS[@]}")
fi
IFS=',' read -ra AVAILABLE_GPUS <<< "${GPU_IDS}"

log() { printf '[online_best] %s\n' "$*"; }
warn() { printf '[online_best][WARN] %s\n' "$*" >&2; }
err() { printf '[online_best][ERROR] %s\n' "$*" >&2; }

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

select_base_ckpt() {
    if [[ -n "${BASE_CKPT:-}" ]]; then
        echo "${BASE_CKPT}"
        return
    fi

    local ckpt=""
    ckpt="$(find "${ROOT}/runs/${EXP_NAME}/aw_shortcut_flow" -path '*/checkpoints/best_eval_success_once.pt' -type f 2>/dev/null | sort -r | head -1)"
    [[ -n "${ckpt}" ]] || ckpt="$(find "${ROOT}/runs/${EXP_NAME}/aw_shortcut_flow" -path '*/checkpoints/best_eval_success_at_end.pt' -type f 2>/dev/null | sort -r | head -1)"
    [[ -n "${ckpt}" ]] || ckpt="$(find "${ROOT}/runs/${EXP_NAME}/aw_shortcut_flow" -path '*/checkpoints/[0-9]*.pt' -type f 2>/dev/null | sort -r | head -1)"
    [[ -n "${ckpt}" ]] || ckpt="$(find "${ROOT}/runs/${EXP_NAME}/shortcut_flow" -path '*/checkpoints/best_eval_success_once.pt' -type f 2>/dev/null | sort -r | head -1)"
    [[ -n "${ckpt}" ]] || ckpt="$(find "${ROOT}/runs/${EXP_NAME}/shortcut_flow" -path '*/checkpoints/best_eval_success_at_end.pt' -type f 2>/dev/null | sort -r | head -1)"
    [[ -n "${ckpt}" ]] || ckpt="$(find "${ROOT}/runs/${EXP_NAME}/shortcut_flow" -path '*/checkpoints/[0-9]*.pt' -type f 2>/dev/null | sort -r | head -1)"

    echo "${ckpt}"
}

run_key_for() {
    local algo=$1
    if [[ "${algo}" == "sac" ]]; then
        echo "rlpd_sac"
    else
        echo "${algo}"
    fi
}

base_dir_for() {
    local algo=$1
    local run_key
    run_key="$(run_key_for "${algo}")"
    echo "${ROOT}/runs/${EXP_NAME}/${run_key}/best"
}

actual_dir_for() {
    local algo=$1
    local run_key
    run_key="$(run_key_for "${algo}")"
    local parent="${ROOT}/runs/${EXP_NAME}/${run_key}"
    find "${parent}" -maxdepth 1 -type d -name 'best__*' 2>/dev/null | sort -r | head -1
}

is_successful() {
    local dir=$1
    [[ -n "${dir}" ]] || return 1
    [[ -f "${dir}/checkpoints/best.pt" ]] && return 0
    [[ -f "${dir}/checkpoints/best_sae.pt" ]] && return 0
    [[ -f "${dir}/checkpoints/final.pt" ]] && return 0
    return 1
}

track_args_rlpd() {
    if [[ "${USE_WANDB}" == "true" ]]; then
        printf '%q ' --track --wandb_project_name "${WANDB_PROJECT}"
    else
        printf '%q ' --no-track
    fi
}

num_demos_args_rlpd() {
    if [[ -n "${RLPD_NUM_DEMOS:-}" ]]; then
        printf '%q ' --num_demos "${RLPD_NUM_DEMOS}"
    fi
}

track_args_online() {
    if [[ "${USE_WANDB}" == "true" ]]; then
        printf '%q ' --track --wandb_project "${WANDB_PROJECT}"
    else
        printf '%q ' --no-track
    fi
}

build_cmd() {
    local gpu=$1
    local algo=$2
    local ckpt=$3
    local run_key
    run_key="$(run_key_for "${algo}")"

    case "${algo}" in
        sac)
            printf '%q ' \
                env "CUDA_VISIBLE_DEVICES=${gpu}" python -m rlft.online.train_rlpd \
                --algorithm sac \
                --env_id "${ENV_ID}" --demo_path "${DEMO_PATH}" \
                --obs_mode "${OBS_MODE}" --control_mode "${CONTROL_MODE}" --sim_backend "${SIM_BACKEND}" \
                --max_episode_steps "${MAX_EPISODE_STEPS}" \
                --num_envs "${NUM_ENVS}" --num_eval_envs "${NUM_EVAL_ENVS}" \
                --batch_size "${BATCH_SIZE}" --total_timesteps "${TOTAL_STEPS_SAC}" \
                --eval_freq "${EVAL_FREQ_SAC}" --save_freq "${SAVE_FREQ}" --num_eval_episodes "${NUM_EVAL_EPISODES}" \
                --gamma 0.9 --tau 0.005 --init_temperature 1.0 --num_qs 10 --num_min_qs 2 \
                --online_ratio 0.5 --reward_scale 1.0 \
                --seed "${SEED}" --exp_name "${EXP_NAME}/${run_key}/best"
            num_demos_args_rlpd
            track_args_rlpd
            ;;
        awsc)
            printf '%q ' \
                env "CUDA_VISIBLE_DEVICES=${gpu}" python -m rlft.online.train_rlpd \
                --algorithm awsc \
                --env_id "${ENV_ID}" --demo_path "${DEMO_PATH}" --pretrain_path "${ckpt}" --no-load_pretrain_critic \
                --obs_mode "${OBS_MODE}" --control_mode "${CONTROL_MODE}" --sim_backend "${SIM_BACKEND}" \
                --max_episode_steps "${MAX_EPISODE_STEPS}" \
                --num_envs "${NUM_ENVS}" --num_eval_envs "${NUM_EVAL_ENVS}" \
                --batch_size "${BATCH_SIZE}" --total_timesteps "${TOTAL_STEPS_AWSC}" \
                --eval_freq "${EVAL_FREQ_AWSC}" --save_freq "${SAVE_FREQ}" --num_eval_episodes "${NUM_EVAL_EPISODES}" \
                --online_ratio 0.15 --utd_ratio 20 --lr_actor 1e-4 --lr_critic 1e-4 \
                --num_qs 10 --num_min_qs 2 --awsc_beta 50.0 --awsc_bc_weight 2.0 \
                --awsc_advantage_mode per_state_v --awsc_num_inference_steps 8 \
                --seed "${SEED}" --exp_name "${EXP_NAME}/${run_key}/best"
            num_demos_args_rlpd
            track_args_rlpd
            ;;
        pld)
            printf '%q ' \
                env "CUDA_VISIBLE_DEVICES=${gpu}" python -m rlft.online.train_pld \
                --env_id "${ENV_ID}" --checkpoint "${ckpt}" \
                --obs_mode "${OBS_MODE}" --control_mode "${CONTROL_MODE}" --sim_backend "${SIM_BACKEND}" --reward_mode "${REWARD_MODE}" \
                --max_episode_steps "${MAX_EPISODE_STEPS}" \
                --num_envs "${NUM_ENVS}" --num_eval_envs "${NUM_EVAL_ENVS}" \
                --batch_size "${BATCH_SIZE}" --total_timesteps "${TOTAL_STEPS_PLD}" \
                --eval_freq "${EVAL_FREQ_PLD}" --save_freq "${SAVE_FREQ}" --num_eval_episodes "${NUM_EVAL_EPISODES}" \
                --pred_horizon 8 \
                --action_scale 0.3 --utd_ratio 60 --gamma 0.99 --target_entropy -3.5 \
                --init_temperature 0.1 --learning_rate 1e-4 --num_layers 3 --layer_size 1024 \
                --num_qs 5 --calql_pretrain_steps 1000 --calql_alpha 0.0 --online_ratio 1.0 \
                --offline_demo_episodes 50 \
                --seed "${SEED}" --exp_name "${EXP_NAME}/${run_key}/best"
            track_args_online
            ;;
        dsrl)
            printf '%q ' \
                env "CUDA_VISIBLE_DEVICES=${gpu}" python -m rlft.online.train_dsrl \
                --env_id "${ENV_ID}" --checkpoint "${ckpt}" \
                --obs_mode "${OBS_MODE}" --control_mode "${CONTROL_MODE}" --sim_backend "${SIM_BACKEND}" --reward_mode "${REWARD_MODE}" \
                --max_episode_steps "${MAX_EPISODE_STEPS}" \
                --num_envs "${NUM_ENVS}" --num_eval_envs "${NUM_EVAL_ENVS}" \
                --batch_size "${BATCH_SIZE}" --total_timesteps "${TOTAL_STEPS_DSRL}" \
                --eval_freq "${EVAL_FREQ_DSRL}" --save_freq "${SAVE_FREQ}" --num_eval_episodes "${NUM_EVAL_EPISODES}" \
                --pred_horizon 8 \
                --action_magnitude 2.5 --utd_ratio 60 --gamma 0.95 --target_entropy -3.5 \
                --log_std_init -5.0 --learning_rate 3e-4 --num_layers 3 --layer_size 2048 \
                --num_qs 10 --num_seed_steps 0 \
                --seed "${SEED}" --exp_name "${EXP_NAME}/${run_key}/best"
            track_args_online
            ;;
        *)
            err "Unknown online algorithm: ${algo}"
            return 1
            ;;
    esac
}

run_one() {
    local gpu=$1
    local algo=$2
    local ckpt=$3
    local base_dir
    base_dir="$(base_dir_for "${algo}")"
    mkdir -p "${base_dir}"

    local actual_dir
    actual_dir="$(actual_dir_for "${algo}")"
    if [[ "${FORCE:-0}" != "1" ]] && is_successful "${actual_dir}"; then
        log "Skip ${algo}: existing successful run ${actual_dir}"
        return 0
    fi

    local log_file="${base_dir}/train.log"
    local cmd
    cmd="$(build_cmd "${gpu}" "${algo}" "${ckpt}")"

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
    local ckpt=$1
    local idx=0
    local completed=0
    local failed=0
    local total=${#RUN_ALGORITHMS[@]}
    local free_gpus=("${AVAILABLE_GPUS[@]}")
    declare -A pid_to_gpu=()

    log "Base checkpoint: ${ckpt}"
    log "Algorithms: ${RUN_ALGORITHMS[*]}"
    log "GPUs: ${AVAILABLE_GPUS[*]}"

    while (( idx < total || ${#pid_to_gpu[@]} > 0 )); do
        while (( idx < total && ${#free_gpus[@]} > 0 )); do
            local algo="${RUN_ALGORITHMS[$idx]}"
            local gpu="${free_gpus[0]}"
            free_gpus=("${free_gpus[@]:1}")
            run_one "${gpu}" "${algo}" "${ckpt}" &
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

    local ckpt
    ckpt="$(select_base_ckpt)"
    if [[ "${DRY_RUN:-0}" == "1" && -z "${ckpt}" ]]; then
        ckpt="${BASE_CKPT:-/tmp/peginsertion_base_dry_run.pt}"
    fi
    [[ "${DRY_RUN:-0}" == "1" || -f "${ckpt}" ]] || {
        err "No base checkpoint found. Run offline shortcut_flow or aw_shortcut_flow first."
        exit 1
    }

    run_queue "${ckpt}"
}

main "$@"

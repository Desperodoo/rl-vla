#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR=""
TEST_DIR=""
GPUS=""
AUTO_FREE_GPUS=1
FREE_GPU_UTIL_THRESHOLD=10
FREE_GPU_MEM_THRESHOLD_MB=1024
FREE_GPU_WAIT_SEC=30
TOTAL_ITERS=100000
BATCH_SIZE=256
STATE_MODE="joint_only"
PRED_HORIZON=16
OBS_HORIZON=2
ACT_HORIZON=8
NUM_WORKERS=4
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_dir)
      TRAIN_DIR="$2"
      shift 2
      ;;
    --test_dir)
      TEST_DIR="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      AUTO_FREE_GPUS=0
      shift 2
      ;;
    --auto_free_gpus)
      AUTO_FREE_GPUS=1
      shift 1
      ;;
    --free_gpu_util_threshold)
      FREE_GPU_UTIL_THRESHOLD="$2"
      shift 2
      ;;
    --free_gpu_mem_threshold_mb)
      FREE_GPU_MEM_THRESHOLD_MB="$2"
      shift 2
      ;;
    --free_gpu_wait_sec)
      FREE_GPU_WAIT_SEC="$2"
      shift 2
      ;;
    --total_iters)
      TOTAL_ITERS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --state_mode)
      STATE_MODE="$2"
      shift 2
      ;;
    --pred_horizon)
      PRED_HORIZON="$2"
      shift 2
      ;;
    --obs_horizon)
      OBS_HORIZON="$2"
      shift 2
      ;;
    --act_horizon)
      ACT_HORIZON="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --extra_args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${TRAIN_DIR}" || -z "${TEST_DIR}" ]]; then
  echo "Usage: $0 --train_dir <path> --test_dir <path> [--gpus 0,1,2,3 | --auto_free_gpus]"
  exit 1
fi

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "train_dir not found: ${TRAIN_DIR}"
  exit 1
fi

if [[ ! -d "${TEST_DIR}" ]]; then
  echo "test_dir not found: ${TEST_DIR}"
  exit 1
fi

ALGORITHMS=("flow_matching" "consistency_flow")
BACKBONES=("resnet10" "resnet18")
REQUIRED_GPUS=$(( ${#ALGORITHMS[@]} * ${#BACKBONES[@]} ))

resolve_gpu_list() {
  local -a resolved

  if [[ -n "${GPUS}" ]]; then
    IFS=',' read -r -a resolved <<< "${GPUS}"
    echo "${resolved[@]}"
    return 0
  fi

  while true; do
    mapfile -t candidates < <(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' -v mem_t="${FREE_GPU_MEM_THRESHOLD_MB}" -v util_t="${FREE_GPU_UTIL_THRESHOLD}" '{
            gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
            if (($2+0) <= mem_t && ($3+0) <= util_t) print $1;
          }'
    )

    if (( ${#candidates[@]} >= REQUIRED_GPUS )); then
      resolved=("${candidates[@]:0:${REQUIRED_GPUS}}")
      echo "${resolved[@]}"
      return 0
    fi

    if (( AUTO_FREE_GPUS == 0 )); then
      echo "Not enough GPUs provided. Need ${REQUIRED_GPUS}, got ${#candidates[@]}" >&2
      return 1
    fi

    echo "Waiting for free GPUs (need ${REQUIRED_GPUS}, found ${#candidates[@]}). Retry in ${FREE_GPU_WAIT_SEC}s..."
    sleep "${FREE_GPU_WAIT_SEC}"
  done
}

read -r -a GPU_LIST <<< "$(resolve_gpu_list)"
NUM_GPUS=${#GPU_LIST[@]}

if (( NUM_GPUS < REQUIRED_GPUS )); then
  echo "Need at least ${REQUIRED_GPUS} GPUs, but got ${NUM_GPUS}: ${GPU_LIST[*]}"
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/carm_grid_${TIMESTAMP}"
mkdir -p "${LOG_ROOT}"

echo "Using GPUs: ${GPU_LIST[*]}" | tee "${LOG_ROOT}/gpu_selection.txt"
echo "thresholds: util<=${FREE_GPU_UTIL_THRESHOLD}, mem<=${FREE_GPU_MEM_THRESHOLD_MB}MB" | tee -a "${LOG_ROOT}/gpu_selection.txt"

pids=()
task_idx=0

for algorithm in "${ALGORITHMS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    gpu_id=${GPU_LIST[$task_idx]}
    exp_name="${algorithm}_${backbone}_seed1"
    log_file="${LOG_ROOT}/${exp_name}.log"

    cmd="source ~/anaconda3/etc/profile.d/conda.sh && conda activate maniskill && HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_carm \
      --demo_path ${TRAIN_DIR} \
      --val_demo_path ${TEST_DIR} \
      --algorithm ${algorithm} \
      --visual_encoder_type ${backbone} \
      --exp_name ${exp_name} \
      --total_iters ${TOTAL_ITERS} \
      --batch_size ${BATCH_SIZE} \
      --state_mode ${STATE_MODE} \
      --pred_horizon ${PRED_HORIZON} \
      --obs_horizon ${OBS_HORIZON} \
      --act_horizon ${ACT_HORIZON} \
      --num_dataload_workers ${NUM_WORKERS} \
      --eval_freq 2000 \
      --eval_batches 50 \
      ${EXTRA_ARGS}"

    echo "[launch] gpu=${gpu_id} algo=${algorithm} backbone=${backbone}"
    bash -lc "${cmd}" > "${log_file}" 2>&1 &
    pid=$!
    pids+=("${pid}")
    echo "${pid} gpu=${gpu_id} algo=${algorithm} backbone=${backbone} log=${log_file}" >> "${LOG_ROOT}/pids.txt"

    task_idx=$((task_idx + 1))
  done
done

echo "Launched ${task_idx} jobs. Logs: ${LOG_ROOT}"
echo "PID records: ${LOG_ROOT}/pids.txt"

failed=0
for idx in "${!pids[@]}"; do
  pid=${pids[$idx]}
  if ! wait "${pid}"; then
    echo "[failed] pid=${pid}" | tee -a "${LOG_ROOT}/failures.txt"
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "Some jobs failed. See ${LOG_ROOT}/failures.txt"
  exit 1
fi

echo "All jobs completed"

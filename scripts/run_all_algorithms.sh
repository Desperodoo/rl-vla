#!/bin/bash
# =============================================================================
# 多 GPU 批量训练脚本
# 支持同时运行多个训练进程，每个进程使用不同的 GPU
# =============================================================================

set -e

# =============================================================================
# 配置区
# =============================================================================

# 任务配置
TASK="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"

# 数据路径 (会根据 obs_mode 自动拼接)
DEMO_DIR="${HOME}/.maniskill/demos/${TASK}/rl"

# 可用 GPU 列表 (默认使用 GPU 0-9)
AVAILABLE_GPUS=(${GPUS:-0 1 2 3 4 5 6 7 8 9})
NUM_GPUS=${#AVAILABLE_GPUS[@]}

# 训练配置
TOTAL_ITERS=${TOTAL_ITERS:-25000}        # 完整训练
QUICK_TEST_ITERS=${QUICK_TEST_ITERS:-1000}  # 快速验证

# 是否使用 WandB
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"ManiSkill-RLFT-Dense-Success"}

# 日志目录
LOG_DIR="logs/training_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# 算法列表
# =============================================================================

# 模仿学习 (IL) 算法
IL_ALGORITHMS=(
    "flow_matching"
    "shortcut_flow"
    "consistency_flow"
    "diffusion_policy"
    "reflected_flow"
)

# 离线强化学习 (Offline RL) 算法
OFFLINE_RL_ALGORITHMS=(
    "cpql"
    "awcp"
    "aw_shortcut_flow"
)

# 合并所有算法
ALL_ALGORITHMS=("${IL_ALGORITHMS[@]}" "${OFFLINE_RL_ALGORITHMS[@]}")

# 观测模式
OBS_MODES=("rgb")

# =============================================================================
# 辅助函数
# =============================================================================

print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --quick           快速验证模式 (${QUICK_TEST_ITERS} 步)"
    echo "  --full            完整训练模式 (${TOTAL_ITERS} 步)"
    echo "  --obs-mode MODE   观测模式 (rgb/state/both), 默认 both"
    echo "  --algorithms LIST 指定算法 (逗号分隔), 默认全部"
    echo "  --gpus LIST       指定 GPU (逗号分隔), 默认 0-9"
    echo "  --dry-run         只打印命令不执行"
    echo "  --help            显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --quick --obs-mode rgb --gpus 0,1,2,3"
    echo "  $0 --full --algorithms flow_matching,cpql"
    echo ""
}

# =============================================================================
# 解析参数
# =============================================================================

ITERS=${TOTAL_ITERS}
SELECTED_OBS_MODES=("${OBS_MODES[@]}")
SELECTED_ALGORITHMS=("${ALL_ALGORITHMS[@]}")
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            ITERS=${QUICK_TEST_ITERS}
            shift
            ;;
        --full)
            ITERS=${TOTAL_ITERS}
            shift
            ;;
        --obs-mode)
            if [ "$2" = "both" ]; then
                SELECTED_OBS_MODES=("rgb" "state")
            else
                SELECTED_OBS_MODES=("$2")
            fi
            shift 2
            ;;
        --algorithms)
            IFS=',' read -ra SELECTED_ALGORITHMS <<< "$2"
            shift 2
            ;;
        --gpus)
            IFS=',' read -ra AVAILABLE_GPUS <<< "$2"
            NUM_GPUS=${#AVAILABLE_GPUS[@]}
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 主逻辑
# =============================================================================

print_header "多 GPU 批量训练"
echo "任务: ${TASK}"
echo "训练步数: ${ITERS}"
echo "可用 GPU: ${AVAILABLE_GPUS[*]}"
echo "观测模式: ${SELECTED_OBS_MODES[*]}"
echo "算法: ${SELECTED_ALGORITHMS[*]}"
echo "WandB: ${USE_WANDB}"
echo "日志目录: ${LOG_DIR}"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 生成任务队列
TASK_QUEUE=()
for obs_mode in "${SELECTED_OBS_MODES[@]}"; do
    demo_path="${DEMO_DIR}/trajectory.${obs_mode}.${CONTROL_MODE}.${SIM_BACKEND}.h5"
    
    # 检查数据文件是否存在
    if [ ! -f "${demo_path}" ]; then
        echo "警告: 数据文件不存在，跳过 obs_mode=${obs_mode}: ${demo_path}"
        continue
    fi
    
    for algo in "${SELECTED_ALGORITHMS[@]}"; do
        TASK_QUEUE+=("${algo}|${obs_mode}|${demo_path}")
    done
done

NUM_TASKS=${#TASK_QUEUE[@]}
echo ""
echo "总任务数: ${NUM_TASKS}"

if [ ${NUM_TASKS} -eq 0 ]; then
    echo "错误: 没有可执行的任务"
    exit 1
fi

# WandB 参数
WANDB_ARGS=""
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_ARGS="--track --wandb_project_name ${WANDB_PROJECT}"
fi

# 启动训练任务
print_header "启动训练任务"

PIDS=()
GPU_INDEX=0
TASK_INDEX=0

for task_info in "${TASK_QUEUE[@]}"; do
    IFS='|' read -r algo obs_mode demo_path <<< "${task_info}"
    
    # 选择 GPU (循环分配)
    gpu_id=${AVAILABLE_GPUS[$((GPU_INDEX % NUM_GPUS))]}
    
    # 实验名称
    exp_name="${TASK}_${algo}_${obs_mode}"
    
    # 日志文件
    log_file="${LOG_DIR}/${exp_name}.log"
    
    # 构建训练命令
    cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_maniskill \
        --env_id ${TASK} \
        --demo_path ${demo_path} \
        --algorithm ${algo} \
        --obs_mode ${obs_mode} \
        --control_mode ${CONTROL_MODE} \
        --sim_backend ${SIM_BACKEND} \
        --total_iters ${ITERS} \
        --exp_name ${exp_name} \
        ${WANDB_ARGS}"
    
    echo ""
    echo "[${TASK_INDEX}/${NUM_TASKS}] 启动: ${exp_name}"
    echo "  GPU: ${gpu_id}"
    echo "  日志: ${log_file}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  命令: ${cmd}"
    else
        # 后台运行
        eval "${cmd}" > "${log_file}" 2>&1 &
        pid=$!
        PIDS+=("${pid}:${exp_name}")
        echo "  PID: ${pid}"
        
        # 记录 PID 信息
        echo "${pid}|${exp_name}|${gpu_id}|$(date +%s)" >> "${LOG_DIR}/running_tasks.txt"
    fi
    
    GPU_INDEX=$((GPU_INDEX + 1))
    TASK_INDEX=$((TASK_INDEX + 1))
    
    # 如果 GPU 用完一轮，等待一些任务完成后再继续
    if [ $((TASK_INDEX % NUM_GPUS)) -eq 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; then
        echo ""
        echo "等待 GPU 资源... (已启动 ${TASK_INDEX} 个任务)"
        # 等待任意一个任务完成
        wait -n 2>/dev/null || true
    fi
done

# 保存任务信息
echo ""
print_header "训练任务已启动"
echo "任务信息保存在: ${LOG_DIR}/running_tasks.txt"
echo ""
echo "监控命令:"
echo "  bash scripts/monitor_training.sh ${LOG_DIR}"
echo ""
echo "查看单个日志:"
echo "  tail -f ${LOG_DIR}/<exp_name>.log"
echo ""

if [ "${DRY_RUN}" = "false" ]; then
    echo "所有进程 PID:"
    for pid_info in "${PIDS[@]}"; do
        IFS=':' read -r pid name <<< "${pid_info}"
        echo "  ${pid}: ${name}"
    done
fi

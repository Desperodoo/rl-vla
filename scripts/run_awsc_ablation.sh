#!/bin/bash
# =============================================================================
# AW-ShortCut Flow 消融实验脚本
# 将 aw_shortcut_flow 消融到 shortcut_flow，找出关键因素
# =============================================================================

set -e

# =============================================================================
# 配置区
# =============================================================================

TASK="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"
OBS_MODE="rgb"

DEMO_DIR="${HOME}/.maniskill/demos/${TASK}/rl"
DEMO_PATH="${DEMO_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5"

# 可用 GPU 列表
AVAILABLE_GPUS=(${GPUS:-0 1 2 3 4 5 6 7 8 9})
NUM_GPUS=${#AVAILABLE_GPUS[@]}

# 训练配置
TOTAL_ITERS=${TOTAL_ITERS:-25000}
QUICK_TEST_ITERS=${QUICK_TEST_ITERS:-5000}

# WandB 配置
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"ManiSkill-AWSC-Ablation"}

# 日志目录
LOG_DIR="logs/ablation_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# 消融实验配置
# 
# 目标: 找出 AWSC (work) 和 ShortCut Flow (不work) 的关键差异
# 
# 主要差异:
# 1. Q-weighting (beta): AWSC 用 Q-value 加权 BC loss
#    - beta 小: 强 Q-weighting (更倾向高 Q 样本)
#    - beta 大: 弱 Q-weighting (接近均匀权重，等价于纯 BC)
#    - beta → ∞: weight = exp(A/beta) → 1，完全等于纯 BC
#
# 2. Shortcut consistency: 两者都有，但 weight 可能不同
#
# 消融路径: AWSC (beta=10) → AWSC (beta↑) → 纯 BC ≈ ShortCut Flow
# =============================================================================

# 消融实验列表: "名称|算法|额外参数"
ABLATION_EXPERIMENTS=(
    # ===== Baseline =====
    # 纯 ShortCut Flow (不 work)
    "baseline_sc|shortcut_flow|"
    
    # 完整 AWSC (work)
    "baseline_awsc|aw_shortcut_flow|--beta 10.0 --consistency_weight 0.3"
    
    # ===== 消融 Q-weighting (核心消融) =====
    # beta 从小到大，逐步接近纯 BC
    "awsc_beta1|aw_shortcut_flow|--beta 1.0 --consistency_weight 0.3"
    "awsc_beta100|aw_shortcut_flow|--beta 100.0 --consistency_weight 0.3"
    "awsc_beta1000|aw_shortcut_flow|--beta 1000.0 --consistency_weight 0.3"
    
    # ===== 消融 Shortcut Consistency =====
    # AWSC 去掉 shortcut consistency
    "awsc_no_sc|aw_shortcut_flow|--beta 10.0 --consistency_weight 0.0"
    
    # 纯 SC 去掉 shortcut consistency (只剩 flow matching)
    "sc_no_sc|shortcut_flow|--shortcut_weight 0.0"
)

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
    echo "  --gpus LIST       指定 GPU (逗号分隔), 默认 0-9"
    echo "  --experiments LIST 指定实验 (逗号分隔), 默认全部"
    echo "  --dry-run         只打印命令不执行"
    echo "  --list            列出所有可用实验"
    echo "  --help            显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --quick --gpus 0,1,2,3"
    echo "  $0 --full --experiments baseline_shortcut_flow,baseline_awsc_beta10"
    echo ""
}

list_experiments() {
    echo "可用的消融实验:"
    echo ""
    for exp in "${ABLATION_EXPERIMENTS[@]}"; do
        IFS='|' read -r name algo params <<< "${exp}"
        echo "  ${name}"
        echo "    算法: ${algo}"
        echo "    参数: ${params:-'(默认)'}"
        echo ""
    done
}

# =============================================================================
# 解析参数
# =============================================================================

ITERS=${TOTAL_ITERS}
SELECTED_EXPERIMENTS=("${ABLATION_EXPERIMENTS[@]}")
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
        --gpus)
            IFS=',' read -ra AVAILABLE_GPUS <<< "$2"
            NUM_GPUS=${#AVAILABLE_GPUS[@]}
            shift 2
            ;;
        --experiments)
            IFS=',' read -ra exp_names <<< "$2"
            SELECTED_EXPERIMENTS=()
            for name in "${exp_names[@]}"; do
                for exp in "${ABLATION_EXPERIMENTS[@]}"; do
                    if [[ "${exp}" == "${name}|"* ]]; then
                        SELECTED_EXPERIMENTS+=("${exp}")
                        break
                    fi
                done
            done
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            list_experiments
            exit 0
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

print_header "AW-ShortCut Flow 消融实验"
echo "任务: ${TASK}"
echo "观测模式: ${OBS_MODE}"
echo "训练步数: ${ITERS}"
echo "可用 GPU: ${AVAILABLE_GPUS[*]}"
echo "实验数量: ${#SELECTED_EXPERIMENTS[@]}"
echo "WandB 项目: ${WANDB_PROJECT}"
echo "日志目录: ${LOG_DIR}"

# 检查数据文件
if [ ! -f "${DEMO_PATH}" ]; then
    echo "错误: 数据文件不存在: ${DEMO_PATH}"
    exit 1
fi

# 创建日志目录
mkdir -p "${LOG_DIR}"

# WandB 参数
WANDB_ARGS=""
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_ARGS="--track --wandb_project_name ${WANDB_PROJECT}"
fi

# 启动实验
print_header "启动消融实验"

PIDS=()
GPU_INDEX=0
TASK_INDEX=0
NUM_TASKS=${#SELECTED_EXPERIMENTS[@]}

for exp_info in "${SELECTED_EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name algo extra_params <<< "${exp_info}"
    
    # 选择 GPU
    gpu_id=${AVAILABLE_GPUS[$((GPU_INDEX % NUM_GPUS))]}
    
    # 实验名称
    full_exp_name="${TASK}_${exp_name}_${OBS_MODE}"
    
    # 日志文件
    log_file="${LOG_DIR}/${full_exp_name}.log"
    
    # 构建训练命令
    cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_maniskill \
        --env_id ${TASK} \
        --demo_path ${DEMO_PATH} \
        --algorithm ${algo} \
        --obs_mode ${OBS_MODE} \
        --control_mode ${CONTROL_MODE} \
        --sim_backend ${SIM_BACKEND} \
        --total_iters ${ITERS} \
        --exp_name ${full_exp_name} \
        ${extra_params} \
        ${WANDB_ARGS}"
    
    echo ""
    echo "[${TASK_INDEX}/${NUM_TASKS}] 启动: ${full_exp_name}"
    echo "  算法: ${algo}"
    echo "  额外参数: ${extra_params:-'(默认)'}"
    echo "  GPU: ${gpu_id}"
    echo "  日志: ${log_file}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  命令: ${cmd}"
    else
        # 后台运行
        eval "${cmd}" > "${log_file}" 2>&1 &
        pid=$!
        PIDS+=("${pid}:${full_exp_name}")
        echo "  PID: ${pid}"
        
        # 记录 PID 信息
        echo "${pid}|${full_exp_name}|${gpu_id}|${algo}|${extra_params}|$(date +%s)" >> "${LOG_DIR}/running_tasks.txt"
    fi
    
    GPU_INDEX=$((GPU_INDEX + 1))
    TASK_INDEX=$((TASK_INDEX + 1))
    
    # 如果 GPU 用完一轮，等待一些任务完成后再继续
    if [ $((TASK_INDEX % NUM_GPUS)) -eq 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; then
        echo ""
        echo "等待 GPU 资源... (已启动 ${TASK_INDEX} 个任务)"
        wait -n 2>/dev/null || true
    fi
done

# 输出信息
echo ""
print_header "消融实验已启动"
echo "任务信息保存在: ${LOG_DIR}/running_tasks.txt"
echo ""
echo "监控命令:"
echo "  bash scripts/monitor_training.sh ${LOG_DIR}"
echo ""
echo "终止所有实验:"
echo "  pkill -f 'rlft.offline.train_maniskill'"
echo ""

if [ "${DRY_RUN}" = "false" ]; then
    echo "所有进程 PID:"
    for pid_info in "${PIDS[@]}"; do
        IFS=':' read -r pid name <<< "${pid_info}"
        echo "  ${pid}: ${name}"
    done
fi

echo ""
print_header "消融实验说明"
cat << 'EOF'
目标: 找出 AWSC (work) 和 ShortCut Flow (不work) 的关键差异

消融路径:
  baseline_awsc (beta=10, work)
       ↓ 增大 beta
  awsc_beta100 (beta=100)
       ↓ 继续增大 beta
  awsc_beta1000 (beta=1000, 接近纯 BC)
       ↓ 
  baseline_sc (纯 ShortCut Flow, 不work)

实验列表 (共 8 个):
┌─────────────────┬─────────────────────────────────────┐
│ baseline_sc     │ 纯 ShortCut Flow (不work)           │
│ baseline_awsc   │ 完整 AWSC beta=10 (work)            │
├─────────────────┼─────────────────────────────────────┤
│ awsc_beta1      │ 强 Q-weighting                      │
│ awsc_beta100    │ 弱 Q-weighting                      │
│ awsc_beta1000   │ 极弱 Q-weighting (≈纯 BC)           │
├─────────────────┼─────────────────────────────────────┤
│ awsc_no_sc      │ AWSC 无 shortcut consistency        │
│ sc_no_sc        │ SC 无 shortcut (纯 flow matching)   │
└─────────────────┴─────────────────────────────────────┘

预期结果:
- 如果 beta↑ 导致性能下降 → Q-weighting 是关键
- 如果 awsc_beta1000 仍然 work → Q-weighting 不是关键
- 如果 awsc_no_sc 不 work → shortcut consistency 在 AWSC 中重要

分析: 比较 eval/success_once 和 eval/success_at_end
EOF

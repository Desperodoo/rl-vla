#!/bin/bash
# =============================================================================
# 离线算法超参数扫描脚本
# 为所有 IL 和 Offline RL 算法扫描关键超参数
# =============================================================================

# 不使用 set -e，因为 ((idx++)) 在 idx=0 时会返回 1

# =============================================================================
# 配置区
# =============================================================================

TASK="LiftPegUpright-v1"
CONTROL_MODE="pd_ee_delta_pose"
SIM_BACKEND="physx_cuda"
OBS_MODE="rgb"

DEMO_DIR="${HOME}/.maniskill/demos/${TASK}/rl"
DEMO_PATH="${DEMO_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5"

# GPU 配置
AVAILABLE_GPUS=(${GPUS:-0 1 2 3 4 5 6 7 8 9})
NUM_GPUS=${#AVAILABLE_GPUS[@]}

# 训练配置
TOTAL_ITERS=${TOTAL_ITERS:-25000}
QUICK_TEST_ITERS=${QUICK_TEST_ITERS:-5000}

# WandB 配置
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"ManiSkill-Sweep"}

# 日志目录
LOG_DIR="logs/sweep_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# 超参数网格定义
# 
# 设计原则:
# 1. 每个算法选择 2-3 个最关键的超参数
# 2. 每个参数 3-5 个值，总实验数控制在 8-12 个
# 3. 包含学习率、算法特定参数、推理步数等关键维度
# =============================================================================

# ===================== IL 算法 =====================

# Flow Matching: lr × num_flow_steps
# 关键点: lr 对收敛有重大影响; flow_steps 影响推理质量
SWEEP_flow_matching=(
    # 学习率扫描 (num_flow_steps=10)
    "--lr 1e-4 --num_flow_steps 10"
    "--lr 3e-4 --num_flow_steps 10"
    "--lr 1e-3 --num_flow_steps 10"
    # 推理步数扫描 (lr=3e-4)
    "--lr 3e-4 --num_flow_steps 5"
    "--lr 3e-4 --num_flow_steps 20"
    # obs/act horizon 扫描
    "--lr 3e-4 --obs_horizon 1 --act_horizon 4 --pred_horizon 8"
    "--lr 3e-4 --obs_horizon 4 --act_horizon 16 --pred_horizon 32"
)

# Diffusion Policy: lr × num_diffusion_iters
# 关键点: 扩散步数对质量影响大
SWEEP_diffusion_policy=(
    # 学习率扫描
    "--lr 1e-4 --num_diffusion_iters 100"
    "--lr 3e-4 --num_diffusion_iters 100"
    "--lr 1e-3 --num_diffusion_iters 100"
    # 扩散步数扫描
    "--lr 3e-4 --num_diffusion_iters 50"
    "--lr 3e-4 --num_diffusion_iters 200"
    # horizon 扫描
    "--lr 3e-4 --obs_horizon 1 --act_horizon 4 --pred_horizon 8"
    "--lr 3e-4 --obs_horizon 4 --act_horizon 16 --pred_horizon 32"
)

# ShortCut Flow: sc_self_consistency_k × sc_num_inference_steps × lr
# 关键点: consistency比例、推理步数
SWEEP_shortcut_flow=(
    # consistency比例扫描
    "--lr 3e-4 --sc_self_consistency_k 0.0 --sc_num_inference_steps 8"
    "--lr 3e-4 --sc_self_consistency_k 0.25 --sc_num_inference_steps 8"
    "--lr 3e-4 --sc_self_consistency_k 0.5 --sc_num_inference_steps 8"
    # 推理步数扫描
    "--lr 3e-4 --sc_self_consistency_k 0.25 --sc_num_inference_steps 4"
    "--lr 3e-4 --sc_self_consistency_k 0.25 --sc_num_inference_steps 16"
    # 学习率扫描
    "--lr 1e-4 --sc_self_consistency_k 0.25 --sc_num_inference_steps 8"
    "--lr 1e-3 --sc_self_consistency_k 0.25 --sc_num_inference_steps 8"
    # step size 模式
    "--lr 3e-4 --sc_step_size_mode power2 --sc_num_inference_steps 8"
)

# Consistency Flow: consistency_weight × cons_delta × cons_teacher_steps
# 关键点: consistency权重、delta大小
SWEEP_consistency_flow=(
    # consistency权重扫描
    "--lr 3e-4 --consistency_weight 0.1 --cons_delta_fixed 0.05"
    "--lr 3e-4 --consistency_weight 0.3 --cons_delta_fixed 0.05"
    "--lr 3e-4 --consistency_weight 0.5 --cons_delta_fixed 0.05"
    # delta扫描
    "--lr 3e-4 --consistency_weight 0.3 --cons_delta_fixed 0.02"
    "--lr 3e-4 --consistency_weight 0.3 --cons_delta_fixed 0.1"
    # delta模式扫描
    "--lr 3e-4 --consistency_weight 0.3 --cons_delta_mode random"
    # 学习率扫描
    "--lr 1e-4 --consistency_weight 0.3 --cons_delta_fixed 0.05"
    "--lr 1e-3 --consistency_weight 0.3 --cons_delta_fixed 0.05"
)

# Reflected Flow: boundary_reg_weight × reflection_mode × lr
# 关键点: 边界正则强度、反射模式
SWEEP_reflected_flow=(
    # boundary_reg_weight扫描 (hard mode)
    "--lr 3e-4 --reflection_mode hard --boundary_reg_weight 0.001"
    "--lr 3e-4 --reflection_mode hard --boundary_reg_weight 0.01"
    "--lr 3e-4 --reflection_mode hard --boundary_reg_weight 0.1"
    # soft mode
    "--lr 3e-4 --reflection_mode soft --boundary_reg_weight 0.01"
    "--lr 3e-4 --reflection_mode soft --boundary_reg_weight 0.1"
    # 学习率扫描
    "--lr 1e-4 --reflection_mode hard --boundary_reg_weight 0.01"
    "--lr 1e-3 --reflection_mode hard --boundary_reg_weight 0.01"
)

# ===================== Offline RL 算法 =====================

# CPQL: alpha × beta × reward_scale
# 关键点: alpha是conservative惩罚, beta是AWAC温度
SWEEP_cpql=(
    # alpha扫描 (conservative惩罚)
    "--alpha 0.001 --beta 10.0 --reward_scale 0.1"
    "--alpha 0.01 --beta 10.0 --reward_scale 0.1"
    "--alpha 0.1 --beta 10.0 --reward_scale 0.1"
    # beta扫描 (AWAC温度)
    "--alpha 0.01 --beta 1.0 --reward_scale 0.1"
    "--alpha 0.01 --beta 50.0 --reward_scale 0.1"
    # reward_scale扫描
    "--alpha 0.01 --beta 10.0 --reward_scale 0.01"
    "--alpha 0.01 --beta 10.0 --reward_scale 1.0"
    # 学习率扫描
    "--alpha 0.01 --beta 10.0 --reward_scale 0.1 --lr 1e-4"
)

# AWCP: beta × consistency_weight × reward_scale
# 关键点: beta(Q权重温度), consistency_weight
SWEEP_awcp=(
    # beta扫描
    "--beta 1.0 --consistency_weight 0.3 --reward_scale 0.1"
    "--beta 5.0 --consistency_weight 0.3 --reward_scale 0.1"
    "--beta 10.0 --consistency_weight 0.3 --reward_scale 0.1"
    "--beta 50.0 --consistency_weight 0.3 --reward_scale 0.1"
    # consistency_weight扫描
    "--beta 10.0 --consistency_weight 0.0 --reward_scale 0.1"
    "--beta 10.0 --consistency_weight 0.1 --reward_scale 0.1"
    "--beta 10.0 --consistency_weight 0.5 --reward_scale 0.1"
    # reward_scale扫描
    "--beta 10.0 --consistency_weight 0.3 --reward_scale 0.01"
    "--beta 10.0 --consistency_weight 0.3 --reward_scale 1.0"
)

# AW-ShortCut Flow: beta × sc_self_consistency_k × sc_num_inference_steps
# 关键点: beta(Q权重), consistency比例, 推理步数
SWEEP_aw_shortcut_flow=(
    # beta扫描 (Q权重温度)
    "--beta 1.0 --sc_self_consistency_k 0.25 --reward_scale 0.1"
    "--beta 5.0 --sc_self_consistency_k 0.25 --reward_scale 0.1"
    "--beta 10.0 --sc_self_consistency_k 0.25 --reward_scale 0.1"
    "--beta 50.0 --sc_self_consistency_k 0.25 --reward_scale 0.1"
    # consistency比例扫描
    "--beta 10.0 --sc_self_consistency_k 0.0 --reward_scale 0.1"
    "--beta 10.0 --sc_self_consistency_k 0.5 --reward_scale 0.1"
    # 推理步数扫描
    "--beta 10.0 --sc_self_consistency_k 0.25 --sc_num_inference_steps 4 --reward_scale 0.1"
    "--beta 10.0 --sc_self_consistency_k 0.25 --sc_num_inference_steps 16 --reward_scale 0.1"
    # reward_scale扫描
    "--beta 10.0 --sc_self_consistency_k 0.25 --reward_scale 0.01"
    "--beta 10.0 --sc_self_consistency_k 0.25 --reward_scale 1.0"
)

# =============================================================================
# 算法列表
# =============================================================================

IL_ALGORITHMS=("flow_matching" "diffusion_policy" "shortcut_flow" "consistency_flow" "reflected_flow")
OFFLINE_RL_ALGORITHMS=("cpql" "awcp" "aw_shortcut_flow")
ALL_ALGORITHMS=("${IL_ALGORITHMS[@]}" "${OFFLINE_RL_ALGORITHMS[@]}")

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
    echo "  --quick              快速验证模式 (${QUICK_TEST_ITERS} 步)"
    echo "  --full               完整训练模式 (${TOTAL_ITERS} 步)"
    echo "  --gpus LIST          指定 GPU (逗号分隔), 默认 0-9"
    echo "  --algorithms LIST    指定算法 (逗号分隔), 默认全部"
    echo "  --il-only            只扫描 IL 算法"
    echo "  --rl-only            只扫描 Offline RL 算法"
    echo "  --dry-run            只打印命令不执行"
    echo "  --list               列出所有算法和参数配置"
    echo "  --help               显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --quick --gpus 0,1,2,3,4,5,6,7"
    echo "  $0 --full --algorithms shortcut_flow,aw_shortcut_flow"
    echo "  $0 --full --il-only"
    echo ""
}

list_sweep_configs() {
    echo "超参数扫描配置:"
    echo ""
    local total_count=0
    
    for algo in "${ALL_ALGORITHMS[@]}"; do
        echo "=== ${algo} ==="
        local idx=0
        local count=0
        
        # 根据算法名手动获取配置
        case ${algo} in
            flow_matching)
                for cfg in "${SWEEP_flow_matching[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_flow_matching[@]}
                ;;
            diffusion_policy)
                for cfg in "${SWEEP_diffusion_policy[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_diffusion_policy[@]}
                ;;
            shortcut_flow)
                for cfg in "${SWEEP_shortcut_flow[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_shortcut_flow[@]}
                ;;
            consistency_flow)
                for cfg in "${SWEEP_consistency_flow[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_consistency_flow[@]}
                ;;
            reflected_flow)
                for cfg in "${SWEEP_reflected_flow[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_reflected_flow[@]}
                ;;
            cpql)
                for cfg in "${SWEEP_cpql[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_cpql[@]}
                ;;
            awcp)
                for cfg in "${SWEEP_awcp[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_awcp[@]}
                ;;
            aw_shortcut_flow)
                for cfg in "${SWEEP_aw_shortcut_flow[@]}"; do
                    echo "  [${idx}] ${cfg}"
                    ((idx++))
                done
                count=${#SWEEP_aw_shortcut_flow[@]}
                ;;
        esac
        echo "  共 ${count} 个配置"
        echo ""
        total_count=$((total_count + count))
    done
    
    echo "========================================"
    echo "总共 ${total_count} 个实验配置"
}

get_sweep_configs() {
    local algo=$1
    local sweep_var="SWEEP_${algo}[@]"
    echo "${!sweep_var}"
}

# =============================================================================
# 解析参数
# =============================================================================

ITERS=${TOTAL_ITERS}
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
        --gpus)
            IFS=',' read -ra AVAILABLE_GPUS <<< "$2"
            NUM_GPUS=${#AVAILABLE_GPUS[@]}
            shift 2
            ;;
        --algorithms)
            IFS=',' read -ra SELECTED_ALGORITHMS <<< "$2"
            shift 2
            ;;
        --il-only)
            SELECTED_ALGORITHMS=("${IL_ALGORITHMS[@]}")
            shift
            ;;
        --rl-only)
            SELECTED_ALGORITHMS=("${OFFLINE_RL_ALGORITHMS[@]}")
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            list_sweep_configs
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

print_header "离线算法超参数扫描"
echo "任务: ${TASK}"
echo "观测模式: ${OBS_MODE}"
echo "训练步数: ${ITERS}"
echo "可用 GPU: ${AVAILABLE_GPUS[*]}"
echo "算法: ${SELECTED_ALGORITHMS[*]}"
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

# 生成所有实验任务
generate_task_queue() {
    local algo=$1
    local config_idx=0
    
    case ${algo} in
        flow_matching)
            for cfg in "${SWEEP_flow_matching[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        diffusion_policy)
            for cfg in "${SWEEP_diffusion_policy[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        shortcut_flow)
            for cfg in "${SWEEP_shortcut_flow[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        consistency_flow)
            for cfg in "${SWEEP_consistency_flow[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        reflected_flow)
            for cfg in "${SWEEP_reflected_flow[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        cpql)
            for cfg in "${SWEEP_cpql[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        awcp)
            for cfg in "${SWEEP_awcp[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
        aw_shortcut_flow)
            for cfg in "${SWEEP_aw_shortcut_flow[@]}"; do
                echo "${TASK}_${algo}_cfg${config_idx}_${OBS_MODE}|${algo}|${cfg}"
                ((config_idx++))
            done
            ;;
    esac
}

TASK_QUEUE=()
for algo in "${SELECTED_ALGORITHMS[@]}"; do
    while IFS= read -r task; do
        TASK_QUEUE+=("${task}")
    done < <(generate_task_queue "${algo}")
done

NUM_TASKS=${#TASK_QUEUE[@]}

# 获取算法实验数量的函数
get_algo_count() {
    local algo=$1
    case ${algo} in
        flow_matching) echo ${#SWEEP_flow_matching[@]} ;;
        diffusion_policy) echo ${#SWEEP_diffusion_policy[@]} ;;
        shortcut_flow) echo ${#SWEEP_shortcut_flow[@]} ;;
        consistency_flow) echo ${#SWEEP_consistency_flow[@]} ;;
        reflected_flow) echo ${#SWEEP_reflected_flow[@]} ;;
        cpql) echo ${#SWEEP_cpql[@]} ;;
        awcp) echo ${#SWEEP_awcp[@]} ;;
        aw_shortcut_flow) echo ${#SWEEP_aw_shortcut_flow[@]} ;;
        *) echo 0 ;;
    esac
}

print_header "实验统计"
echo "总实验数: ${NUM_TASKS}"
echo ""
echo "各算法实验数:"
for algo in "${SELECTED_ALGORITHMS[@]}"; do
    count=$(get_algo_count "${algo}")
    echo "  ${algo}: ${count}"
done

if [ ${NUM_TASKS} -eq 0 ]; then
    echo "错误: 没有可执行的实验"
    exit 1
fi

# 启动实验
print_header "启动超参数扫描"

PIDS=()
GPU_INDEX=0
TASK_INDEX=0

for task_info in "${TASK_QUEUE[@]}"; do
    IFS='|' read -r exp_name algo extra_params <<< "${task_info}"
    
    # 选择 GPU
    gpu_id=${AVAILABLE_GPUS[$((GPU_INDEX % NUM_GPUS))]}
    
    # 日志文件
    log_file="${LOG_DIR}/${exp_name}.log"
    
    # 构建训练命令
    cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_maniskill \
        --env_id ${TASK} \
        --demo_path ${DEMO_PATH} \
        --algorithm ${algo} \
        --obs_mode ${OBS_MODE} \
        --control_mode ${CONTROL_MODE} \
        --sim_backend ${SIM_BACKEND} \
        --total_iters ${ITERS} \
        --exp_name ${exp_name} \
        ${extra_params} \
        ${WANDB_ARGS}"
    
    echo ""
    echo "[${TASK_INDEX}/${NUM_TASKS}] ${exp_name}"
    echo "  算法: ${algo} | GPU: ${gpu_id}"
    echo "  参数: ${extra_params}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  命令: ${cmd}"
    else
        # 后台运行
        eval "${cmd}" > "${log_file}" 2>&1 &
        pid=$!
        PIDS+=("${pid}:${exp_name}")
        echo "  PID: ${pid} | 日志: ${log_file}"
        
        # 记录任务信息
        echo "${pid}|${exp_name}|${algo}|${gpu_id}|${extra_params}|$(date +%s)" >> "${LOG_DIR}/running_tasks.txt"
    fi
    
    GPU_INDEX=$((GPU_INDEX + 1))
    TASK_INDEX=$((TASK_INDEX + 1))
    
    # GPU 用完一轮，等待
    if [ $((TASK_INDEX % NUM_GPUS)) -eq 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; then
        echo ""
        echo "等待 GPU 资源... (已启动 ${TASK_INDEX}/${NUM_TASKS})"
        wait -n 2>/dev/null || true
    fi
done

# 输出信息
echo ""
print_header "扫描任务已启动"
echo "任务信息: ${LOG_DIR}/running_tasks.txt"
echo ""
echo "监控命令:"
echo "  bash scripts/monitor_training.sh ${LOG_DIR}"
echo ""
echo "分析结果 (训练完成后运行):"
echo "  python scripts/analyze_sweep.py --log_dir ${LOG_DIR}"
echo ""
echo "终止所有:"
echo "  pkill -f 'rlft.offline.train_maniskill'"

if [ "${DRY_RUN}" = "false" ]; then
    # 保存配置信息供分析脚本使用
    cat > "${LOG_DIR}/sweep_config.json" << EOF
{
    "task": "${TASK}",
    "obs_mode": "${OBS_MODE}",
    "total_iters": ${ITERS},
    "algorithms": [$(printf '"%s",' "${SELECTED_ALGORITHMS[@]}" | sed 's/,$//')],
    "wandb_project": "${WANDB_PROJECT}",
    "log_dir": "${LOG_DIR}"
}
EOF
    echo ""
    echo "配置已保存: ${LOG_DIR}/sweep_config.json"
fi

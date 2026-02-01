#!/bin/bash
# =============================================================================
# RLPD Sweep 公共函数和配置
#
# 为 train_rlpd.py (Online RL) 提供 sweep 基础设施
# 支持 SAC 和 AWSC 算法，支持预训练模型和 train from scratch
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =============================================================================
# 环境配置
# =============================================================================

# 任务和数据配置（可被外部覆盖）
export TASK="${TASK:-PickCube-v1}"
export DEMO_PATH="${DEMO_PATH:-${PROJECT_ROOT}/data/demos/${TASK}/demos.h5}"
export OBS_MODE="${OBS_MODE:-rgb}"
export CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
export SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"

# 训练配置
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
export NUM_ENVS="${NUM_ENVS:-50}"
export NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-25}"
export EVAL_FREQ="${EVAL_FREQ:-10000}"
export SAVE_FREQ="${SAVE_FREQ:-50000}"

# Sweep 输出目录
export SWEEP_ROOT="${SWEEP_ROOT:-${PROJECT_ROOT}/sweep_results_rlpd}"

# WandB 配置
export USE_WANDB="${USE_WANDB:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-rlpd-sweep}"

# =============================================================================
# GPU 配置（独占模式）
# =============================================================================

init_gpu_config() {
    # 支持外部指定 GPU 列表
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        IFS=',' read -ra AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        # 自动检测可用 GPU
        local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        AVAILABLE_GPUS=()
        for ((i=0; i<gpu_count; i++)); do
            AVAILABLE_GPUS+=($i)
        done
    fi
    
    NUM_GPUS=${#AVAILABLE_GPUS[@]}
    
    if [ ${NUM_GPUS} -eq 0 ]; then
        echo "错误: 未检测到可用 GPU"
        exit 1
    fi
    
    export AVAILABLE_GPUS
    export NUM_GPUS
}

# 初始化 GPU 配置
init_gpu_config

# =============================================================================
# 重试配置（Online RL 训练时间长，增加等待时间）
# =============================================================================

export MAX_RETRIES="${MAX_RETRIES:-3}"           # 最大重试次数
export RETRY_WAIT="${RETRY_WAIT:-30}"            # 重试等待时间（秒，比 offline 更长）
export SERIAL_MODE="${SERIAL_MODE:-false}"       # 串行模式（用于失败重试）

# =============================================================================
# 日志工具
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [ "${DEBUG}" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# =============================================================================
# 检查函数
# =============================================================================

# 检查 demo 文件是否存在
check_demo_file() {
    if [ ! -f "${DEMO_PATH}" ]; then
        print_error "Demo 文件不存在: ${DEMO_PATH}"
        print_info "请设置 DEMO_PATH 环境变量指向正确的 demo 文件"
        exit 1
    fi
    print_info "Demo 文件: ${DEMO_PATH}"
}

# 检查预训练 checkpoint 是否存在
check_pretrain_checkpoint() {
    local pretrain_path=$1
    
    if [ -z "${pretrain_path}" ]; then
        return 1  # 未指定预训练路径
    fi
    
    if [ ! -f "${pretrain_path}" ]; then
        print_error "预训练 checkpoint 不存在: ${pretrain_path}"
        return 1
    fi
    
    print_info "预训练 checkpoint: ${pretrain_path}"
    return 0
}

# 检查阶段是否已完成
check_stage_completed() {
    local stage_dir=$1
    local algorithm=$2
    
    if [ -f "${stage_dir}/best_params_${algorithm}.sh" ]; then
        return 0  # 已完成
    fi
    return 1  # 未完成
}

# =============================================================================
# CUDA 错误检测
# =============================================================================

# 检测日志中是否有 CUDA 错误
check_cuda_error() {
    local log_file=$1
    if [ ! -f "${log_file}" ]; then
        return 1
    fi
    grep -qE "CUDA error|RuntimeError.*CUDA|illegal memory access|段错误|Segmentation fault|核心已转储|PhysX Internal CUDA error|out of memory|OOM" "${log_file}"
}

# 检测 RLPD 实验是否成功完成
check_rlpd_experiment_success() {
    local log_file=$1
    if [ ! -f "${log_file}" ]; then
        return 1
    fi
    # 检查 online RL 特有的完成标志
    grep -qE "eval/success_once|Training completed|Saving final checkpoint|100%.*${TOTAL_TIMESTEPS}/${TOTAL_TIMESTEPS}" "${log_file}"
}

# =============================================================================
# 实验运行函数
# =============================================================================

# 运行单个 RLPD 实验（带重试）
# 用法: run_rlpd_experiment <exp_name> <algorithm> <gpu_id> <extra_params> <log_dir>
run_rlpd_experiment() {
    local exp_name=$1
    local algorithm=$2
    local gpu_id=$3
    local extra_params=$4
    local log_dir=$5
    
    local log_file="${log_dir}/${exp_name}.log"
    
    # WandB 参数
    local wandb_args=""
    if [ "${USE_WANDB}" = "true" ]; then
        wandb_args="--track --wandb_project_name ${WANDB_PROJECT}"
    fi
    
    # 构建命令 - 使用 train_rlpd.py
    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.online.train_rlpd \
        --env_id ${TASK} \
        --demo_path ${DEMO_PATH} \
        --algorithm ${algorithm} \
        --obs_mode ${OBS_MODE} \
        --control_mode ${CONTROL_MODE} \
        --sim_backend ${SIM_BACKEND} \
        --total_timesteps ${TOTAL_TIMESTEPS} \
        --num_envs ${NUM_ENVS} \
        --num_eval_envs ${NUM_EVAL_ENVS} \
        --eval_freq ${EVAL_FREQ} \
        --save_freq ${SAVE_FREQ} \
        --exp_name ${exp_name} \
        ${extra_params} \
        ${wandb_args}"
    
    echo ""
    print_info "[${exp_name}] GPU: ${gpu_id}"
    echo "  算法: ${algorithm}"
    echo "  参数: ${extra_params}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  命令: ${cmd}"
        return 0
    fi
    
    # 串行模式：同步执行带重试
    if [ "${SERIAL_MODE}" = "true" ]; then
        run_rlpd_experiment_with_retry "${exp_name}" "${algorithm}" "${gpu_id}" "${extra_params}" "${log_dir}" "${cmd}"
        return $?
    fi
    
    # 并行模式：后台运行（GPU 独占）
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    echo "  PID: ${pid} | 日志: ${log_file}"
    
    # 记录任务信息
    echo "${pid}|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)" >> "${log_dir}/running_tasks.txt"
    
    echo "${pid}"
}

# 带重试的 RLPD 实验运行（串行）
run_rlpd_experiment_with_retry() {
    local exp_name=$1
    local algorithm=$2
    local gpu_id=$3
    local extra_params=$4
    local log_dir=$5
    local cmd=$6
    
    local log_file="${log_dir}/${exp_name}.log"
    local retry_count=0
    
    while [ ${retry_count} -lt ${MAX_RETRIES} ]; do
        # 如果存在旧日志且有错误，删除后重试
        if [ -f "${log_file}" ] && check_cuda_error "${log_file}"; then
            local backup="${log_file}.failed.${retry_count}"
            mv "${log_file}" "${backup}"
            print_warn "[${exp_name}] 发现 CUDA 错误，已备份日志到 ${backup}"
        fi
        
        # 运行实验
        print_info "[${exp_name}] 尝试 $((retry_count + 1))/${MAX_RETRIES}..."
        eval "${cmd}" > "${log_file}" 2>&1
        local exit_code=$?
        
        # 检查是否成功
        if [ ${exit_code} -eq 0 ] && check_rlpd_experiment_success "${log_file}"; then
            print_info "[${exp_name}] ✅ 成功完成"
            echo "SUCCESS|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)" >> "${log_dir}/completed_tasks.txt"
            return 0
        fi
        
        # 检查是否 CUDA 错误
        if check_cuda_error "${log_file}"; then
            retry_count=$((retry_count + 1))
            if [ ${retry_count} -lt ${MAX_RETRIES} ]; then
                print_warn "[${exp_name}] CUDA 错误，等待 ${RETRY_WAIT}s 后重试..."
                sleep ${RETRY_WAIT}
            fi
        else
            # 其他错误，不重试
            print_error "[${exp_name}] 非 CUDA 错误，退出码: ${exit_code}"
            echo "FAILED|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)|exit_code=${exit_code}" >> "${log_dir}/failed_tasks.txt"
            return ${exit_code}
        fi
    done
    
    print_error "[${exp_name}] ❌ 达到最大重试次数 (${MAX_RETRIES})"
    echo "FAILED|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)|max_retries" >> "${log_dir}/failed_tasks.txt"
    return 1
}

# 等待所有任务完成
wait_all_tasks() {
    local log_dir=$1
    
    if [ "${DRY_RUN}" = "true" ]; then
        return 0
    fi
    
    print_info "等待所有任务完成..."
    wait
    print_info "所有任务已完成"
}

# =============================================================================
# 批量运行 Sweep
# =============================================================================

# 批量运行 RLPD sweep（GPU 独占模式）
# 用法: run_rlpd_sweep <algorithm> <log_dir> <configs_array_name>
run_rlpd_sweep() {
    local algorithm=$1
    local log_dir=$2
    local -n configs=$3  # nameref to array
    
    # 重新读取 GPU 配置（支持运行时覆盖）
    init_gpu_config
    
    mkdir -p "${log_dir}"
    
    print_header "RLPD Sweep: ${algorithm}"
    echo "日志目录: ${log_dir}"
    echo "实验数量: ${#configs[@]}"
    echo "可用 GPU: ${AVAILABLE_GPUS[*]} (共 ${NUM_GPUS} 个)"
    echo "GPU 模式: 独占（每个实验独占一个 GPU）"
    
    local pids=()
    local gpu_idx=0
    local task_idx=0
    
    for config in "${configs[@]}"; do
        local exp_name="${TASK}_${algorithm}_cfg${task_idx}_${OBS_MODE}"
        local gpu_id=${AVAILABLE_GPUS[$((gpu_idx % NUM_GPUS))]}
        
        run_rlpd_experiment "${exp_name}" "${algorithm}" "${gpu_id}" "${config}" "${log_dir}"
        
        gpu_idx=$((gpu_idx + 1))
        task_idx=$((task_idx + 1))
        
        # GPU 独占：用完一轮 GPU 后，等待一个任务完成再继续
        if [ $((task_idx % NUM_GPUS)) -eq 0 ] && [ ${task_idx} -lt ${#configs[@]} ]; then
            if [ "${DRY_RUN}" != "true" ]; then
                print_info "GPU 已全部占用，等待任务完成... (已启动 ${task_idx}/${#configs[@]})"
                wait -n 2>/dev/null || true
            fi
        fi
    done
    
    # 等待所有任务完成
    wait_all_tasks "${log_dir}"
    
    # 分析并导出最优参数
    if [ "${DRY_RUN}" != "true" ]; then
        analyze_and_export_rlpd "${algorithm}" "${log_dir}"
    fi
}

# =============================================================================
# 分析和导出
# =============================================================================

# 分析 RLPD 实验结果并导出最优参数
analyze_and_export_rlpd() {
    local algorithm=$1
    local log_dir=$2
    
    print_info "分析 ${algorithm} 实验结果..."
    
    # 检查是否有分析脚本
    local analyze_script="${PROJECT_ROOT}/scripts/analyze_rlpd_sweep.py"
    if [ ! -f "${analyze_script}" ]; then
        # 使用通用分析脚本
        analyze_script="${PROJECT_ROOT}/scripts/analyze_sweep.py"
    fi
    
    if [ -f "${analyze_script}" ]; then
        python "${analyze_script}" \
            --log_dir "${log_dir}" \
            --algorithm "${algorithm}" \
            --export_best \
            --output_dir "${log_dir}" \
            --metric "eval/success_once" 2>/dev/null || true
    fi
    
    if [ -f "${log_dir}/best_params_${algorithm}.sh" ]; then
        print_info "最优参数已导出: ${log_dir}/best_params_${algorithm}.sh"
        cat "${log_dir}/best_params_${algorithm}.sh"
    else
        print_warn "未能自动导出最优参数，请手动分析日志"
        print_info "日志目录: ${log_dir}"
    fi
}

# =============================================================================
# 参数继承（从 offline sweep 或之前的 RLPD sweep）
# =============================================================================

# 加载最优参数文件
load_best_params() {
    local base_algo=$1
    local results_dir=$2
    local params_file="${results_dir}/best_params_${base_algo}.sh"
    
    if [ ! -f "${params_file}" ]; then
        echo ""
        return 1
    fi
    
    # 读取参数文件并构建命令行参数
    local params=""
    while IFS='=' read -r key value; do
        # 跳过注释和空行
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # 移除 BEST_ 前缀，转为小写，添加 --
        local param_name=$(echo "${key#BEST_}" | tr '[:upper:]' '[:lower:]')
        # 移除引号
        value="${value%\"}"
        value="${value#\"}"
        
        params="${params} --${param_name} ${value}"
    done < "${params_file}"
    
    echo "${params}"
}

# =============================================================================
# 辅助函数
# =============================================================================

# 生成实验名称
make_exp_name() {
    local algorithm=$1
    local config_id=$2
    local suffix=$3
    
    if [ -n "${suffix}" ]; then
        echo "${TASK}_${algorithm}_${suffix}_cfg${config_id}"
    else
        echo "${TASK}_${algorithm}_cfg${config_id}"
    fi
}

# 打印 sweep 配置摘要
print_sweep_summary() {
    local algorithm=$1
    local -n configs=$2
    
    print_header "Sweep 配置摘要: ${algorithm}"
    echo "任务: ${TASK}"
    echo "Demo: ${DEMO_PATH}"
    echo "观测模式: ${OBS_MODE}"
    echo "总步数: ${TOTAL_TIMESTEPS}"
    echo "实验数量: ${#configs[@]}"
    echo ""
    echo "配置列表:"
    local idx=0
    for config in "${configs[@]}"; do
        echo "  [${idx}] ${config}"
        idx=$((idx + 1))
    done
}

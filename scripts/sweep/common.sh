#!/bin/bash
# =============================================================================
# 级联超参数扫描 - 通用配置和工具函数
# =============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 默认配置
# =============================================================================

export TASK="${TASK:-LiftPegUpright-v1}"
export CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pose}"
export SIM_BACKEND="${SIM_BACKEND:-physx_cuda}"
export OBS_MODE="${OBS_MODE:-rgb}"

export DEMO_DIR="${HOME}/.maniskill/demos/${TASK}/rl"
export DEMO_PATH="${DEMO_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.${SIM_BACKEND}.h5"

# GPU 配置 - 支持运行时通过 GPUS 环境变量覆盖
init_gpu_config() {
    AVAILABLE_GPUS=(${GPUS:-0 1 2 3 4 5 6 7 8 9})
    NUM_GPUS=${#AVAILABLE_GPUS[@]}
    export AVAILABLE_GPUS
    export NUM_GPUS
}

# 初始化默认 GPU 配置
init_gpu_config

# 训练配置
export TOTAL_ITERS="${TOTAL_ITERS:-25000}"

# WandB 配置
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-ManiSkill-Sweep}"

# Sweep 根目录
export SWEEP_ROOT="${SWEEP_ROOT:-logs/cascade_sweep_$(date +%Y%m%d)}"

# =============================================================================
# 工具函数
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo "$1"
    echo -e "==============================================${NC}"
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

# 检查数据文件
check_demo_file() {
    if [ ! -f "${DEMO_PATH}" ]; then
        print_error "数据文件不存在: ${DEMO_PATH}"
        exit 1
    fi
    print_info "数据文件: ${DEMO_PATH}"
}

# 从分析结果JSON中加载最优参数
# 用法: load_best_params <algorithm> <results_dir>
# 输出: 设置环境变量 BEST_* 
load_best_params() {
    local algo=$1
    local results_dir=$2
    local params_file="${results_dir}/best_params_${algo}.sh"
    
    if [ ! -f "${params_file}" ]; then
        print_warn "未找到最优参数文件: ${params_file}"
        return 1
    fi
    
    print_info "加载 ${algo} 最优参数: ${params_file}"
    source "${params_file}"
    return 0
}

# 构建继承参数字符串
# 用法: build_inherited_params <base_algo> <results_dir>
build_inherited_params() {
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
# 重试配置
# =============================================================================

export MAX_RETRIES="${MAX_RETRIES:-3}"           # 最大重试次数
export RETRY_WAIT="${RETRY_WAIT:-15}"            # 重试等待时间（秒）
export SERIAL_MODE="${SERIAL_MODE:-false}"       # 串行模式（用于失败重试）

# 检测日志中是否有 CUDA 错误
check_cuda_error() {
    local log_file=$1
    if [ ! -f "${log_file}" ]; then
        return 1
    fi
    grep -qE "CUDA error|RuntimeError.*CUDA|illegal memory access|段错误|Segmentation fault|核心已转储|PhysX Internal CUDA error" "${log_file}"
}

# 检测实验是否成功完成（日志中有 eval/success_once 指标）
check_experiment_success() {
    local log_file=$1
    if [ ! -f "${log_file}" ]; then
        return 1
    fi
    # 检查是否有完成标志（训练结束会有 "Training completed" 或最终 eval 结果）
    grep -qE "eval/success_once|Training completed|100%.*25000/25000" "${log_file}"
}

# 运行单个实验（带重试）
# 用法: run_experiment <exp_name> <algorithm> <gpu_id> <extra_params> <log_dir>
run_experiment() {
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
    
    # 构建命令
    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_maniskill \
        --env_id ${TASK} \
        --demo_path ${DEMO_PATH} \
        --algorithm ${algorithm} \
        --obs_mode ${OBS_MODE} \
        --control_mode ${CONTROL_MODE} \
        --sim_backend ${SIM_BACKEND} \
        --total_iters ${TOTAL_ITERS} \
        --exp_name ${exp_name} \
        ${extra_params} \
        ${wandb_args}"
    
    echo ""
    print_info "[${exp_name}] GPU: ${gpu_id}"
    echo "  参数: ${extra_params}"
    
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  命令: ${cmd}"
        return 0
    fi
    
    # 串行模式：同步执行带重试
    if [ "${SERIAL_MODE}" = "true" ]; then
        run_experiment_with_retry "${exp_name}" "${algorithm}" "${gpu_id}" "${extra_params}" "${log_dir}" "${cmd}"
        return $?
    fi
    
    # 并行模式：后台运行
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    echo "  PID: ${pid} | 日志: ${log_file}"
    
    # 记录任务信息
    echo "${pid}|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)" >> "${log_dir}/running_tasks.txt"
    
    echo "${pid}"
}

# 带重试的实验运行（串行）
run_experiment_with_retry() {
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
        if [ ${exit_code} -eq 0 ] && check_experiment_success "${log_file}"; then
            print_info "[${exp_name}] ✅ 成功完成"
            # 记录成功
            echo "SUCCESS|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)" >> "${log_dir}/running_tasks.txt"
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
            echo "FAILED|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)|exit_code=${exit_code}" >> "${log_dir}/running_tasks.txt"
            return ${exit_code}
        fi
    done
    
    print_error "[${exp_name}] ❌ 达到最大重试次数 (${MAX_RETRIES})"
    echo "FAILED|${exp_name}|${algorithm}|${gpu_id}|${extra_params}|$(date +%s)|max_retries" >> "${log_dir}/running_tasks.txt"
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

# 运行分析脚本并导出最优参数
# 用法: analyze_and_export <algorithm> <log_dir>
analyze_and_export() {
    local algorithm=$1
    local log_dir=$2
    
    print_info "分析 ${algorithm} 实验结果..."
    
    python scripts/analyze_sweep.py \
        --log_dir "${log_dir}" \
        --algorithm "${algorithm}" \
        --export_best \
        --output_dir "${log_dir}"
    
    if [ -f "${log_dir}/best_params_${algorithm}.sh" ]; then
        print_info "最优参数已导出: ${log_dir}/best_params_${algorithm}.sh"
        cat "${log_dir}/best_params_${algorithm}.sh"
    else
        print_warn "未能导出最优参数"
    fi
}

# 批量运行 sweep
# 用法: run_sweep <algorithm> <log_dir> <configs_array_name>
run_sweep() {
    local algorithm=$1
    local log_dir=$2
    local -n configs=$3  # nameref to array
    
    # 重新读取 GPU 配置（支持运行时覆盖）
    init_gpu_config
    
    mkdir -p "${log_dir}"
    
    print_header "Sweep: ${algorithm}"
    echo "日志目录: ${log_dir}"
    echo "实验数量: ${#configs[@]}"
    echo "可用 GPU: ${AVAILABLE_GPUS[*]} (共 ${NUM_GPUS} 个)"
    
    local pids=()
    local gpu_idx=0
    local task_idx=0
    
    for config in "${configs[@]}"; do
        local exp_name="${TASK}_${algorithm}_cfg${task_idx}_${OBS_MODE}"
        local gpu_id=${AVAILABLE_GPUS[$((gpu_idx % NUM_GPUS))]}
        
        run_experiment "${exp_name}" "${algorithm}" "${gpu_id}" "${config}" "${log_dir}"
        
        gpu_idx=$((gpu_idx + 1))
        task_idx=$((task_idx + 1))
        
        # GPU 用完一轮，等待一个任务完成
        if [ $((task_idx % NUM_GPUS)) -eq 0 ] && [ ${task_idx} -lt ${#configs[@]} ]; then
            if [ "${DRY_RUN}" != "true" ]; then
                print_info "等待 GPU 资源... (已启动 ${task_idx}/${#configs[@]})"
                wait -n 2>/dev/null || true
            fi
        fi
    done
    
    # 等待所有任务完成
    wait_all_tasks "${log_dir}"
    
    # 分析并导出最优参数
    if [ "${DRY_RUN}" != "true" ]; then
        analyze_and_export "${algorithm}" "${log_dir}"
    fi
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

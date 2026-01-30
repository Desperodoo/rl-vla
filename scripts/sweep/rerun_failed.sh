#!/bin/bash
# =============================================================================
# 重跑失败实验
# 
# 扫描所有日志目录，找出因 CUDA 错误而失败的实验，重新运行
# 支持串行模式（带重试）和并行模式（快速但不重试）
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# 参数解析
# =============================================================================

SWEEP_DIR="${SWEEP_ROOT}"
SPECIFIC_ALGO=""
DRY_RUN=false
LIST_ONLY=false
PARALLEL_MODE=false  # 默认串行（带重试）

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --sweep-dir DIR     指定 sweep 根目录 (默认: ${SWEEP_ROOT})"
    echo "  --algorithm ALGO    只重跑指定算法"
    echo "  --parallel          并行重跑（快速，但不带重试）"
    echo "  --serial            串行重跑（默认，带重试机制）"
    echo "  --dry-run           只显示要重跑的实验，不实际运行"
    echo "  --list              只列出失败的实验"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "模式说明:"
    echo "  串行模式（默认）: 逐个重跑，每个实验失败最多重试 ${MAX_RETRIES:-3} 次"
    echo "  并行模式: 同时在所有 GPU 上运行，快速但不带重试"
    echo ""
    echo "示例:"
    echo "  $0 --list                     # 列出所有失败实验"
    echo "  $0 --algorithm reflected_flow # 只重跑 reflected_flow"
    echo "  $0 --parallel                 # 并行重跑所有失败实验"
    echo "  $0 --dry-run                  # 预览要重跑的命令"
    echo "  $0                            # 串行重跑所有失败实验（推荐）"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep-dir)
            SWEEP_DIR="$2"
            shift 2
            ;;
        --algorithm)
            SPECIFIC_ALGO="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --serial)
            PARALLEL_MODE=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 扫描失败实验
# =============================================================================

print_header "扫描失败实验"
print_info "Sweep 目录: ${SWEEP_DIR}"

# 收集所有失败的实验
declare -A FAILED_EXPERIMENTS  # key: "algo|log_dir", value: "exp_name|params"

find_failed_experiments() {
    local log_dir=$1
    local algorithm=$2
    
    # 如果指定了算法，跳过其他
    if [ -n "${SPECIFIC_ALGO}" ] && [ "${algorithm}" != "${SPECIFIC_ALGO}" ]; then
        return
    fi
    
    # 检查 running_tasks.txt
    local tasks_file="${log_dir}/running_tasks.txt"
    if [ ! -f "${tasks_file}" ]; then
        return
    fi
    
    while IFS='|' read -r pid exp_name algo gpu_id params timestamp status; do
        # 跳过已标记失败或成功的记录
        [[ "$pid" == "SUCCESS" ]] && continue
        [[ "$pid" == "FAILED" ]] && continue
        
        local log_file="${log_dir}/${exp_name}.log"
        
        # 检查日志是否存在
        if [ ! -f "${log_file}" ]; then
            echo "  ⚠️ 日志不存在: ${exp_name}"
            FAILED_EXPERIMENTS["${algo}|${log_dir}|${exp_name}"]="${params}"
            continue
        fi
        
        # 检查是否有 CUDA 错误
        if check_cuda_error "${log_file}"; then
            echo "  ❌ CUDA 错误: ${exp_name}"
            FAILED_EXPERIMENTS["${algo}|${log_dir}|${exp_name}"]="${params}"
            continue
        fi
        
        # 检查是否成功完成
        if ! check_experiment_success "${log_file}"; then
            echo "  ⚠️ 未完成: ${exp_name}"
            FAILED_EXPERIMENTS["${algo}|${log_dir}|${exp_name}"]="${params}"
        fi
    done < "${tasks_file}"
}

# 扫描所有阶段目录
for stage_dir in "${SWEEP_DIR}"/stage*; do
    [ -d "${stage_dir}" ] || continue
    echo ""
    print_info "扫描: ${stage_dir##*/}"
    
    for algo_dir in "${stage_dir}"/*; do
        [ -d "${algo_dir}" ] || continue
        algorithm=$(basename "${algo_dir}")
        find_failed_experiments "${algo_dir}" "${algorithm}"
    done
done

# =============================================================================
# 汇总
# =============================================================================

echo ""
print_header "失败实验汇总"

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    print_info "✅ 没有失败的实验!"
    exit 0
fi

echo "共 ${#FAILED_EXPERIMENTS[@]} 个失败实验:"
echo ""

# 按算法分组显示
declare -A ALGO_COUNTS
for key in "${!FAILED_EXPERIMENTS[@]}"; do
    IFS='|' read -r algo log_dir exp_name <<< "$key"
    ALGO_COUNTS[$algo]=$((${ALGO_COUNTS[$algo]:-0} + 1))
done

for algo in "${!ALGO_COUNTS[@]}"; do
    echo "  ${algo}: ${ALGO_COUNTS[$algo]} 个"
done

if [ "${LIST_ONLY}" = "true" ]; then
    echo ""
    echo "详细列表:"
    for key in "${!FAILED_EXPERIMENTS[@]}"; do
        IFS='|' read -r algo log_dir exp_name <<< "$key"
        echo "  - ${exp_name}"
        echo "    算法: ${algo}"
        echo "    参数: ${FAILED_EXPERIMENTS[$key]}"
    done
    exit 0
fi

# =============================================================================
# 重跑失败实验
# =============================================================================

print_header "重跑失败实验"

if [ "${DRY_RUN}" = "true" ]; then
    print_warn "DRY RUN 模式 - 只显示命令，不实际运行"
fi

if [ "${PARALLEL_MODE}" = "true" ]; then
    print_info "模式: 并行 (快速，无重试)"
else
    print_info "模式: 串行 (带重试机制)"
fi

check_demo_file

# 收集所有待重跑的实验
declare -a ALL_KEYS=()
for key in "${!FAILED_EXPERIMENTS[@]}"; do
    ALL_KEYS+=("$key")
done

TOTAL_EXPERIMENTS=${#ALL_KEYS[@]}
echo ""
print_info "总计 ${TOTAL_EXPERIMENTS} 个实验待重跑"

# =============================================================================
# 并行模式: 按 GPU 数量分批运行
# =============================================================================
if [ "${PARALLEL_MODE}" = "true" ]; then
    print_info "使用 ${NUM_GPUS} 个 GPU 并行执行"
    
    batch_num=0
    while [ ${#ALL_KEYS[@]} -gt 0 ]; do
        batch_num=$((batch_num + 1))
        print_header "批次 ${batch_num}"
        
        # 取出最多 NUM_GPUS 个实验
        declare -a BATCH_KEYS=()
        declare -a BATCH_PIDS=()
        
        for ((i=0; i<NUM_GPUS && ${#ALL_KEYS[@]} > 0; i++)); do
            BATCH_KEYS+=("${ALL_KEYS[0]}")
            ALL_KEYS=("${ALL_KEYS[@]:1}")  # 移除第一个元素
        done
        
        echo "本批次运行 ${#BATCH_KEYS[@]} 个实验"
        
        # 启动本批次的实验
        gpu_idx=0
        for key in "${BATCH_KEYS[@]}"; do
            IFS='|' read -r algo log_dir exp_name <<< "$key"
            params="${FAILED_EXPERIMENTS[$key]}"
            
            gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
            gpu_idx=$((gpu_idx + 1))
            
            echo ""
            print_info "[GPU ${gpu_id}] 重跑: ${exp_name}"
            echo "  算法: ${algo}"
            echo "  参数: ${params}"
            
            if [ "${DRY_RUN}" = "true" ]; then
                echo "  [DRY RUN] 将在 GPU ${gpu_id} 上运行..."
                continue
            fi
            
            # 备份旧日志
            old_log="${log_dir}/${exp_name}.log"
            if [ -f "${old_log}" ]; then
                mv "${old_log}" "${old_log}.failed.$(date +%s)"
            fi
            
            # 后台启动实验（不带重试）
            # 使用 bash -c 并将变量作为位置参数传递，避免变量被循环覆盖
            # 参数顺序: $1=gpu, $2=demo_path, $3=algo, $4=params, $5=log, $6=env_id, $7=obs_mode, $8=control_mode, $9=sim_backend, ${10}=total_iters, ${11}=exp_name
            bash -c '
                export CUDA_VISIBLE_DEVICES="$1"
                python -m rlft.offline.train_maniskill \
                    --env_id "$6" \
                    --demo_path "$2" \
                    --algorithm "$3" \
                    --obs_mode "$7" \
                    --control_mode "$8" \
                    --sim_backend "$9" \
                    --total_iters "${10}" \
                    --exp_name "${11}" \
                    $4 \
                    > "$5" 2>&1
            ' _ "${gpu_id}" "${DEMO_PATH}" "${algo}" "${params}" "${log_dir}/${exp_name}.log" "${TASK}" "${OBS_MODE}" "${CONTROL_MODE}" "${SIM_BACKEND}" "${TOTAL_ITERS}" "${exp_name}" &
            
            BATCH_PIDS+=($!)
            echo "  启动 PID: $!"
        done
        
        if [ "${DRY_RUN}" = "true" ]; then
            continue
        fi
        
        # 等待本批次完成
        echo ""
        print_info "等待批次 ${batch_num} 完成..."
        
        for pid in "${BATCH_PIDS[@]}"; do
            wait $pid
        done
        
        print_info "批次 ${batch_num} 完成"
        
        # 批次间等待（GPU 冷却）
        if [ ${#ALL_KEYS[@]} -gt 0 ]; then
            echo "等待 ${RETRY_WAIT}s 让 GPU 冷却后继续下一批..."
            sleep ${RETRY_WAIT}
        fi
    done

# =============================================================================
# 串行模式: 逐个运行，带重试
# =============================================================================
else
    exp_num=0
    for key in "${ALL_KEYS[@]}"; do
        IFS='|' read -r algo log_dir exp_name <<< "$key"
        params="${FAILED_EXPERIMENTS[$key]}"
        
        exp_num=$((exp_num + 1))
        
        # 轮询 GPU
        gpu_id=${AVAILABLE_GPUS[$((exp_num % NUM_GPUS))]}
        
        echo ""
        print_info "[${exp_num}/${TOTAL_EXPERIMENTS}] 重跑: ${exp_name}"
        echo "  算法: ${algo}"
        echo "  GPU: ${gpu_id}"
        echo "  参数: ${params}"
        
        if [ "${DRY_RUN}" = "true" ]; then
            echo "  [DRY RUN] 将运行实验..."
            continue
        fi
        
        # 备份旧日志
        old_log="${log_dir}/${exp_name}.log"
        if [ -f "${old_log}" ]; then
            mv "${old_log}" "${old_log}.failed.$(date +%s)"
            echo "  已备份旧日志"
        fi
        
        # 构建训练命令（包含所有必要参数）
        train_cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.offline.train_maniskill \
            --env_id ${TASK} \
            --demo_path '${DEMO_PATH}' \
            --algorithm ${algo} \
            --obs_mode ${OBS_MODE} \
            --control_mode ${CONTROL_MODE} \
            --sim_backend ${SIM_BACKEND} \
            --total_iters ${TOTAL_ITERS} \
            --exp_name ${exp_name} \
            ${params}"
        
        # 使用重试机制运行
        run_experiment_with_retry "${exp_name}" "${algo}" "${gpu_id}" "${params}" "${log_dir}" "${train_cmd}"
        
        # GPU 冷却
        echo "  等待 ${RETRY_WAIT}s 让 GPU 冷却..."
        sleep ${RETRY_WAIT}
    done
fi

print_header "重跑完成"

# 重新分析受影响的算法
echo ""
print_info "重新分析受影响的算法..."

for algo in "${!ALGO_COUNTS[@]}"; do
    # 找到该算法的日志目录
    for stage_dir in "${SWEEP_DIR}"/stage*; do
        algo_dir="${stage_dir}/${algo}"
        if [ -d "${algo_dir}" ]; then
            print_info "分析 ${algo}..."
            analyze_and_export "${algo}" "${algo_dir}"
        fi
    done
done

print_info "✅ 所有失败实验已重跑完成"

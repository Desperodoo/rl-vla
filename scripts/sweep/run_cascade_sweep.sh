#!/bin/bash
# =============================================================================
# 级联超参数扫描主控脚本
# 
# 按依赖顺序执行所有算法的超参数扫描：
# 
# 阶段 1 (并行): flow_matching, diffusion_policy, reflected_flow
# 阶段 2 (并行): consistency_flow, shortcut_flow (依赖 flow_matching)
# 阶段 3 (并行): cpql, awcp (依赖 consistency_flow), aw_shortcut_flow (依赖 shortcut_flow)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# 配置
# =============================================================================

# 阶段定义
STAGE1_ALGORITHMS=("flow_matching" "diffusion_policy")
STAGE2_ALGORITHMS=("consistency_flow" "shortcut_flow" "reflected_flow")
STAGE3_ALGORITHMS=("cpql" "awcp" "aw_shortcut_flow")

# =============================================================================
# 辅助函数
# =============================================================================

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --stage N        只运行阶段 N (1, 2, 或 3)"
    echo "  --algorithm ALG  只运行指定算法"
    echo "  --analyze [N]    只运行分析环节 (可选指定阶段 1/2/3，默认全部)"
    echo "  --retry-failed   重跑失败的实验 (检测 CUDA 错误等)"
    echo "  --fine-sweep     运行精细化 sweep (基于上一轮结果)"
    echo "  --dry-run        只打印命令不执行"
    echo "  --force          强制重新运行（忽略已完成状态）"
    echo "  --status         显示当前状态"
    echo "  --serial         串行运行模式（避免 GPU 争抢，默认）"
    echo "  --parallel       并行运行模式（为每个算法分配独立 GPU 子集）"
    echo "  --help           显示帮助"
    echo ""
    echo "阶段说明:"
    echo "  阶段 1: 基础 IL 算法 (flow_matching, diffusion_policy)"
    echo "  阶段 2: 依赖 IL 算法 (consistency_flow, shortcut_flow, reflected_flow)"
    echo "  阶段 3: Offline RL 算法 (cpql, awcp, aw_shortcut_flow)"
    echo ""
    echo "GPU 分配说明:"
    echo "  默认串行模式: 阶段内算法依次运行，每个算法使用全部 GPU"
    echo "  并行模式 (--parallel): 阶段内算法并行，GPU 均分给各算法"
    echo "    例: 10 GPU + 2 算法 = 每算法 5 GPU"
    echo ""
    echo "示例:"
    echo "  $0                                  # 串行运行全部阶段（推荐）"
    echo "  $0 --parallel                       # 并行运行（GPU 均分）"
    echo "  $0 --stage 1                        # 只运行阶段 1"
    echo "  $0 --algorithm awcp                 # 只运行 awcp"
    echo "  $0 --analyze                        # 只运行全部分析"
    echo "  $0 --analyze 1                      # 只分析阶段 1"
    echo "  $0 --retry-failed                   # 串行重跑失败实验（带重试）"
    echo "  $0 --retry-failed --parallel        # 并行重跑失败实验（快速）"
    echo "  $0 --retry-failed --dry-run         # 预览要重跑的实验"
    echo "  $0 --fine-sweep                     # 运行精细化 sweep"
    echo "  $0 --fine-sweep --algorithm reflected_flow  # 只精细化 reflected_flow"
    echo "  $0 --dry-run                        # 预览所有命令"
    echo "  $0 --status                         # 查看进度"
    echo ""
}

show_status() {
    print_header "级联 Sweep 状态"
    echo "Sweep 根目录: ${SWEEP_ROOT}"
    echo ""
    
    echo "=== 阶段 1: 基础 IL 算法 ==="
    for algo in "${STAGE1_ALGORITHMS[@]}"; do
        local status="❌ 未完成"
        local dir="${SWEEP_ROOT}/stage1_base_il/${algo}"
        if [ -f "${dir}/best_params_${algo}.sh" ]; then
            status="✅ 已完成"
        elif [ -d "${dir}" ]; then
            status="🔄 运行中"
        fi
        echo "  ${algo}: ${status}"
    done
    
    echo ""
    echo "=== 阶段 2: 依赖 IL 算法 ==="
    for algo in "${STAGE2_ALGORITHMS[@]}"; do
        local status="❌ 未完成"
        local dir="${SWEEP_ROOT}/stage2_dependent_il/${algo}"
        if [ -f "${dir}/best_params_${algo}.sh" ]; then
            status="✅ 已完成"
        elif [ -d "${dir}" ]; then
            status="🔄 运行中"
        fi
        echo "  ${algo}: ${status}"
    done
    
    echo ""
    echo "=== 阶段 3: Offline RL 算法 ==="
    for algo in "${STAGE3_ALGORITHMS[@]}"; do
        local status="❌ 未完成"
        local dir="${SWEEP_ROOT}/stage3_offline_rl/${algo}"
        if [ -f "${dir}/best_params_${algo}.sh" ]; then
            status="✅ 已完成"
        elif [ -d "${dir}" ]; then
            status="🔄 运行中"
        fi
        echo "  ${algo}: ${status}"
    done
}

run_stage() {
    local stage=$1
    local -n algos=$2
    local extra_args="${3:-}"
    
    print_header "阶段 ${stage}"
    
    local num_algos=${#algos[@]}
    
    # 检查是否使用并行模式（默认串行，避免 GPU 争抢）
    if [ "${PARALLEL_ALGOS}" = "true" ] && [ ${num_algos} -le ${NUM_GPUS} ]; then
        # 并行模式：为每个算法分配独立的 GPU 子集
        print_info "并行模式: ${num_algos} 个算法，${NUM_GPUS} 个 GPU"
        
        local gpus_per_algo=$((NUM_GPUS / num_algos))
        local pids=()
        local algo_idx=0
        
        for algo in "${algos[@]}"; do
            # 计算该算法可用的 GPU 范围
            local start_gpu=$((algo_idx * gpus_per_algo))
            local end_gpu=$((start_gpu + gpus_per_algo - 1))
            
            # 构建 GPU 列表
            local algo_gpus=""
            for ((i=start_gpu; i<=end_gpu; i++)); do
                algo_gpus="${algo_gpus}${AVAILABLE_GPUS[$i]} "
            done
            algo_gpus=$(echo "${algo_gpus}" | xargs)  # trim
            
            print_info "启动 ${algo} sweep (GPU: ${algo_gpus})..."
            GPUS="${algo_gpus}" bash "${SCRIPT_DIR}/sweep_${algo}.sh" ${extra_args} &
            pids+=($!)
            
            algo_idx=$((algo_idx + 1))
        done
        
        # 等待所有并行任务完成
        print_info "等待阶段 ${stage} 完成... (${#pids[@]} 个并行任务)"
        for pid in "${pids[@]}"; do
            wait ${pid}
        done
    else
        # 串行模式（默认）：避免 GPU 争抢
        print_info "串行模式: 依次运行 ${num_algos} 个算法"
        
        for algo in "${algos[@]}"; do
            print_info "启动 ${algo} sweep..."
            bash "${SCRIPT_DIR}/sweep_${algo}.sh" ${extra_args}
            print_info "${algo} sweep 完成"
        done
    fi
    
    print_info "阶段 ${stage} 完成"
}

run_single_algorithm() {
    local algo=$1
    local extra_args="${2:-}"
    
    print_header "运行单个算法: ${algo}"
    bash "${SCRIPT_DIR}/sweep_${algo}.sh" ${extra_args}
}

# 运行分析环节
run_analyze_stage() {
    local stage=$1
    local -n algos=$2
    
    print_header "分析阶段 ${stage}"
    
    for algo in "${algos[@]}"; do
        local log_dir
        case ${stage} in
            1) log_dir="${SWEEP_ROOT}/stage1_base_il/${algo}" ;;
            2) log_dir="${SWEEP_ROOT}/stage2_dependent_il/${algo}" ;;
            3) log_dir="${SWEEP_ROOT}/stage3_offline_rl/${algo}" ;;
        esac
        
        if [ -d "${log_dir}" ]; then
            print_info "分析 ${algo}..."
            analyze_and_export "${algo}" "${log_dir}"
        else
            print_warn "${algo} 日志目录不存在: ${log_dir}"
        fi
    done
}

run_all_analysis() {
    print_header "运行全部分析"
    
    run_analyze_stage 1 STAGE1_ALGORITHMS
    run_analyze_stage 2 STAGE2_ALGORITHMS
    run_analyze_stage 3 STAGE3_ALGORITHMS
    
    # 生成最终汇总报告
    print_info "生成最终汇总报告..."
    python scripts/analyze_sweep.py \
        --log_dir "${SWEEP_ROOT}" \
        --recursive \
        --output_dir "${SWEEP_ROOT}"
    
    print_info "最终报告: ${SWEEP_ROOT}/sweep_report.md"
}

# =============================================================================
# 主逻辑
# =============================================================================

# 解析参数
STAGE=""
ALGORITHM=""
ANALYZE_STAGE=""
RETRY_FAILED=false
FINE_SWEEP=false
PARALLEL_ALGOS=false  # 默认串行，避免 GPU 争抢
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE=$2
            shift 2
            ;;
        --algorithm)
            ALGORITHM=$2
            shift 2
            ;;
        --analyze)
            # 检查下一个参数是否是数字（阶段号）
            if [[ -n "$2" && "$2" =~ ^[123]$ ]]; then
                ANALYZE_STAGE=$2
                shift 2
            else
                ANALYZE_STAGE="all"
                shift
            fi
            ;;
        --retry-failed)
            RETRY_FAILED=true
            shift
            ;;
        --fine-sweep)
            FINE_SWEEP=true
            shift
            ;;
        --serial)
            PARALLEL_ALGOS=false
            shift
            ;;
        --parallel)
            PARALLEL_ALGOS=true
            shift
            ;;
        --dry-run)
            EXTRA_ARGS="${EXTRA_ARGS} --dry-run"
            export DRY_RUN=true
            shift
            ;;
        --force)
            EXTRA_ARGS="${EXTRA_ARGS} --force"
            shift
            ;;
        --status)
            show_status
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

# 检查数据文件
check_demo_file

# 创建根目录
mkdir -p "${SWEEP_ROOT}"

print_header "级联超参数扫描"
echo "任务: ${TASK}"
echo "观测模式: ${OBS_MODE}"
echo "训练步数: ${TOTAL_ITERS}"
echo "可用 GPU: ${AVAILABLE_GPUS[*]}"
echo "Sweep 根目录: ${SWEEP_ROOT}"

# 只运行分析
if [ -n "${ANALYZE_STAGE}" ]; then
    if [ "${ANALYZE_STAGE}" = "all" ]; then
        run_all_analysis
    else
        case ${ANALYZE_STAGE} in
            1) run_analyze_stage 1 STAGE1_ALGORITHMS ;;
            2) run_analyze_stage 2 STAGE2_ALGORITHMS ;;
            3) run_analyze_stage 3 STAGE3_ALGORITHMS ;;
        esac
    fi
    exit 0
fi

# 重跑失败实验
if [ "${RETRY_FAILED}" = "true" ]; then
    print_header "重跑失败实验"
    
    retry_args=""
    if [ "${DRY_RUN}" = "true" ]; then
        retry_args="--dry-run"
    fi
    if [ -n "${ALGORITHM}" ]; then
        retry_args="${retry_args} --algorithm ${ALGORITHM}"
    fi
    # 传递并行模式
    if [ "${PARALLEL_ALGOS}" = "true" ]; then
        retry_args="${retry_args} --parallel"
    else
        retry_args="${retry_args} --serial"
    fi
    
    bash "${SCRIPT_DIR}/rerun_failed.sh" --sweep-dir "${SWEEP_ROOT}" ${retry_args}
    exit $?
fi

# 精细化 sweep
if [ "${FINE_SWEEP}" = "true" ]; then
    print_header "精细化 Sweep"
    
    fine_args=""
    if [ "${DRY_RUN}" = "true" ]; then
        fine_args="--dry-run"
    fi
    if [ -n "${ALGORITHM}" ]; then
        fine_args="${fine_args} --algorithm ${ALGORITHM}"
    fi
    if [ -n "${EXTRA_ARGS}" ] && [[ "${EXTRA_ARGS}" == *"--force"* ]]; then
        fine_args="${fine_args} --force"
    fi
    
    export ORIGINAL_SWEEP_ROOT="${SWEEP_ROOT}"
    bash "${SCRIPT_DIR}/fine/run_fine_sweep.sh" ${fine_args}
    exit $?
fi

# 运行单个算法
if [ -n "${ALGORITHM}" ]; then
    run_single_algorithm "${ALGORITHM}" "${EXTRA_ARGS}"
    exit 0
fi

# 运行指定阶段
if [ -n "${STAGE}" ]; then
    case ${STAGE} in
        1)
            run_stage 1 STAGE1_ALGORITHMS "${EXTRA_ARGS}"
            ;;
        2)
            run_stage 2 STAGE2_ALGORITHMS "${EXTRA_ARGS}"
            ;;
        3)
            run_stage 3 STAGE3_ALGORITHMS "${EXTRA_ARGS}"
            ;;
        *)
            print_error "无效阶段: ${STAGE}"
            exit 1
            ;;
    esac
    exit 0
fi

# 运行全部阶段
print_header "开始全流程级联 Sweep"

# 阶段 1: 基础 IL 算法 (并行)
run_stage 1 STAGE1_ALGORITHMS "${EXTRA_ARGS}"

# 阶段 2: 依赖 IL 算法 (并行，依赖阶段 1)
run_stage 2 STAGE2_ALGORITHMS "${EXTRA_ARGS}"

# 阶段 3: Offline RL 算法 (并行，依赖阶段 2)
run_stage 3 STAGE3_ALGORITHMS "${EXTRA_ARGS}"

# 最终报告
print_header "级联 Sweep 完成"
show_status

echo ""
echo "生成最终分析报告..."
python scripts/analyze_sweep.py \
    --log_dir "${SWEEP_ROOT}" \
    --recursive \
    --output_dir "${SWEEP_ROOT}"

print_info "最终报告: ${SWEEP_ROOT}/sweep_report.md"

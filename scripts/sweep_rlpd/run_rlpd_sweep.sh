#!/bin/bash
# =============================================================================
# RLPD Sweep 主调度脚本
#
# 统一入口，支持运行 SAC 和 AWSC 算法的超参数扫描
# 
# 功能：
#   - 运行单个算法 sweep
#   - 运行全部算法 sweep
#   - 重试失败实验
#   - 显示状态和结果
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_rlpd.sh"

# =============================================================================
# 支持的算法
# =============================================================================

SUPPORTED_ALGORITHMS=("sac" "awsc")

# =============================================================================
# 帮助信息
# =============================================================================

usage() {
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "RLPD 超参数扫描主调度脚本"
    echo ""
    echo "命令:"
    echo "  run                   运行 sweep (默认)"
    echo "  status                显示当前状态"
    echo "  retry                 重试失败的实验"
    echo "  analyze               分析结果并导出最优参数"
    echo ""
    echo "选项:"
    echo "  --algorithm ALG       指定算法: sac, awsc, all (默认: all)"
    echo "  --pretrain-path PATH  预训练 checkpoint 路径 (AWSC 必需)"
    echo "  --awsc-mode MODE      AWSC 运行模式: pretrain, scratch, both (默认: both)"
    echo "  --dry-run             只显示命令，不实际运行"
    echo "  --serial              串行模式（带重试机制）"
    echo "  --parallel            并行模式（默认）"
    echo "  --force               强制重新运行（忽略已完成状态）"
    echo "  -h, --help            显示帮助"
    echo ""
    echo "环境变量:"
    echo "  TASK                  任务 ID (默认: PickCube-v1)"
    echo "  DEMO_PATH             Demo 文件路径"
    echo "  TOTAL_TIMESTEPS       总训练步数 (默认: 500000)"
    echo "  SWEEP_ROOT            结果输出目录"
    echo "  CUDA_VISIBLE_DEVICES  可用 GPU 列表"
    echo "  USE_WANDB             是否使用 WandB (默认: false)"
    echo ""
    echo "示例:"
    echo "  # 运行所有算法的 sweep"
    echo "  $0 --pretrain-path runs/shortcut_flow/best.pt"
    echo ""
    echo "  # 只运行 SAC sweep"
    echo "  $0 --algorithm sac"
    echo ""
    echo "  # 只运行 AWSC from scratch"
    echo "  $0 --algorithm awsc --awsc-mode scratch"
    echo ""
    echo "  # 预览模式"
    echo "  $0 --dry-run"
    echo ""
    echo "  # 查看状态"
    echo "  $0 status"
    echo ""
    echo "  # 重试失败实验"
    echo "  $0 retry --algorithm sac --serial"
}

# =============================================================================
# 状态显示
# =============================================================================

show_status() {
    print_header "RLPD Sweep 状态"
    echo "任务: ${TASK}"
    echo "结果目录: ${SWEEP_ROOT}"
    echo ""
    
    for algo in "${SUPPORTED_ALGORITHMS[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        if [ "${algo}" = "awsc" ]; then
            # AWSC 有两种模式
            for mode in "scratch" "pretrain"; do
                local log_dir="${SWEEP_ROOT}/${algo}_${mode}"
                local algo_suffix="${algo}_${mode}"
                
                echo -n "[$algo_suffix] "
                
                if [ -f "${log_dir}/best_params_${algo_suffix}.sh" ]; then
                    echo -e "${GREEN}✅ 完成${NC}"
                    echo "  最优参数: ${log_dir}/best_params_${algo_suffix}.sh"
                elif [ -d "${log_dir}" ]; then
                    local total=$(ls -1 "${log_dir}"/*.log 2>/dev/null | wc -l)
                    local failed=$(grep -l "CUDA error\|OOM\|Segmentation fault" "${log_dir}"/*.log 2>/dev/null | wc -l)
                    echo -e "${YELLOW}⏳ 进行中${NC} (日志: ${total}, 失败: ${failed})"
                else
                    echo -e "${BLUE}⏸️ 未开始${NC}"
                fi
            done
        else
            local log_dir="${SWEEP_ROOT}/${algo}"
            
            echo -n "[$algo] "
            
            if [ -f "${log_dir}/best_params_${algo}.sh" ]; then
                echo -e "${GREEN}✅ 完成${NC}"
                echo "  最优参数: ${log_dir}/best_params_${algo}.sh"
            elif [ -d "${log_dir}" ]; then
                local total=$(ls -1 "${log_dir}"/*.log 2>/dev/null | wc -l)
                local failed=$(grep -l "CUDA error\|OOM\|Segmentation fault" "${log_dir}"/*.log 2>/dev/null | wc -l)
                echo -e "${YELLOW}⏳ 进行中${NC} (日志: ${total}, 失败: ${failed})"
            else
                echo -e "${BLUE}⏸️ 未开始${NC}"
            fi
        fi
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# =============================================================================
# 重试失败实验
# =============================================================================

retry_failed() {
    local algorithm=$1
    
    print_header "重试失败实验: ${algorithm}"
    
    local log_dirs=()
    
    if [ "${algorithm}" = "all" ]; then
        log_dirs+=("${SWEEP_ROOT}/sac")
        log_dirs+=("${SWEEP_ROOT}/awsc_scratch")
        log_dirs+=("${SWEEP_ROOT}/awsc_pretrain")
    elif [ "${algorithm}" = "awsc" ]; then
        log_dirs+=("${SWEEP_ROOT}/awsc_scratch")
        log_dirs+=("${SWEEP_ROOT}/awsc_pretrain")
    else
        log_dirs+=("${SWEEP_ROOT}/${algorithm}")
    fi
    
    for log_dir in "${log_dirs[@]}"; do
        if [ ! -d "${log_dir}" ]; then
            print_warn "目录不存在: ${log_dir}"
            continue
        fi
        
        print_info "扫描: ${log_dir}"
        
        # 查找失败的实验
        local failed_logs=$(grep -l "CUDA error\|OOM\|Segmentation fault\|RuntimeError" "${log_dir}"/*.log 2>/dev/null)
        
        if [ -z "${failed_logs}" ]; then
            print_info "未发现失败实验"
            continue
        fi
        
        echo "发现失败实验:"
        for log in ${failed_logs}; do
            local exp_name=$(basename "${log}" .log)
            echo "  - ${exp_name}"
            
            if [ "${DRY_RUN}" = "true" ]; then
                continue
            fi
            
            # 从 running_tasks.txt 或 failed_tasks.txt 中提取原始命令参数
            local task_info=$(grep "|${exp_name}|" "${log_dir}/running_tasks.txt" "${log_dir}/failed_tasks.txt" 2>/dev/null | tail -1)
            
            if [ -n "${task_info}" ]; then
                local algo=$(echo "${task_info}" | cut -d'|' -f3)
                local gpu_id=$(echo "${task_info}" | cut -d'|' -f4)
                local params=$(echo "${task_info}" | cut -d'|' -f5)
                
                # 备份旧日志
                mv "${log}" "${log}.failed.$(date +%Y%m%d_%H%M%S)"
                
                # 重新运行
                print_info "重新运行: ${exp_name} on GPU ${gpu_id}"
                run_rlpd_experiment_with_retry "${exp_name}" "${algo}" "${gpu_id}" "${params}" "${log_dir}" \
                    "CUDA_VISIBLE_DEVICES=${gpu_id} python -m rlft.online.train_rlpd ${params} --exp_name ${exp_name}"
            else
                print_warn "无法找到 ${exp_name} 的原始参数，跳过"
            fi
        done
    done
}

# =============================================================================
# 分析结果
# =============================================================================

analyze_results() {
    local algorithm=$1
    
    print_header "分析结果: ${algorithm}"
    
    local log_dirs=()
    local algo_names=()
    
    if [ "${algorithm}" = "all" ]; then
        log_dirs+=("${SWEEP_ROOT}/sac")
        algo_names+=("sac")
        log_dirs+=("${SWEEP_ROOT}/awsc_scratch")
        algo_names+=("awsc_scratch")
        log_dirs+=("${SWEEP_ROOT}/awsc_pretrain")
        algo_names+=("awsc_pretrain")
    elif [ "${algorithm}" = "awsc" ]; then
        log_dirs+=("${SWEEP_ROOT}/awsc_scratch")
        algo_names+=("awsc_scratch")
        log_dirs+=("${SWEEP_ROOT}/awsc_pretrain")
        algo_names+=("awsc_pretrain")
    else
        log_dirs+=("${SWEEP_ROOT}/${algorithm}")
        algo_names+=("${algorithm}")
    fi
    
    for i in "${!log_dirs[@]}"; do
        local log_dir="${log_dirs[$i]}"
        local algo_name="${algo_names[$i]}"
        
        if [ ! -d "${log_dir}" ]; then
            print_warn "目录不存在: ${log_dir}"
            continue
        fi
        
        analyze_and_export_rlpd "${algo_name}" "${log_dir}"
    done
}

# =============================================================================
# 运行 Sweep
# =============================================================================

run_sweep_for_algorithm() {
    local algorithm=$1
    
    case "${algorithm}" in
        sac)
            print_info "启动 SAC sweep..."
            local sac_args=""
            [ "${DRY_RUN}" = "true" ] && sac_args="${sac_args} --dry-run"
            [ "${SERIAL_MODE}" = "true" ] && sac_args="${sac_args} --serial"
            [ "${FORCE}" = "true" ] && sac_args="${sac_args} --force"
            
            bash "${SCRIPT_DIR}/sweep_sac.sh" ${sac_args}
            ;;
        awsc)
            print_info "启动 AWSC sweep..."
            local awsc_args="--mode ${AWSC_MODE}"
            [ -n "${PRETRAIN_PATH}" ] && awsc_args="${awsc_args} --pretrain-path ${PRETRAIN_PATH}"
            [ "${DRY_RUN}" = "true" ] && awsc_args="${awsc_args} --dry-run"
            [ "${SERIAL_MODE}" = "true" ] && awsc_args="${awsc_args} --serial"
            [ "${FORCE}" = "true" ] && awsc_args="${awsc_args} --force"
            
            bash "${SCRIPT_DIR}/sweep_awsc.sh" ${awsc_args}
            ;;
        all)
            run_sweep_for_algorithm "sac"
            echo ""
            run_sweep_for_algorithm "awsc"
            ;;
        *)
            print_error "不支持的算法: ${algorithm}"
            echo "支持的算法: ${SUPPORTED_ALGORITHMS[*]}, all"
            exit 1
            ;;
    esac
}

# =============================================================================
# 参数解析
# =============================================================================

COMMAND="run"
ALGORITHM="all"
AWSC_MODE="both"
PRETRAIN_PATH="${PRETRAIN_PATH:-}"
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        run|status|retry|analyze)
            COMMAND="$1"
            shift
            ;;
        --algorithm|-a)
            ALGORITHM="$2"
            shift 2
            ;;
        --pretrain-path)
            PRETRAIN_PATH="$2"
            shift 2
            ;;
        --awsc-mode)
            AWSC_MODE="$2"
            shift 2
            ;;
        --dry-run)
            export DRY_RUN=true
            shift
            ;;
        --serial)
            export SERIAL_MODE=true
            shift
            ;;
        --parallel)
            export SERIAL_MODE=false
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# 主逻辑
# =============================================================================

print_header "RLPD Sweep 调度器"
echo "命令: ${COMMAND}"
echo "算法: ${ALGORITHM}"
echo "任务: ${TASK}"
echo "结果目录: ${SWEEP_ROOT}"
echo ""

case "${COMMAND}" in
    run)
        run_sweep_for_algorithm "${ALGORITHM}"
        ;;
    status)
        show_status
        ;;
    retry)
        retry_failed "${ALGORITHM}"
        ;;
    analyze)
        analyze_results "${ALGORITHM}"
        ;;
    *)
        print_error "未知命令: ${COMMAND}"
        usage
        exit 1
        ;;
esac

print_header "完成"

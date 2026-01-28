#!/bin/bash
# =============================================================================
# 训练进程监控脚本
# 监控多 GPU 训练任务的运行状态、GPU 使用情况和训练指标
# =============================================================================

# 日志目录
LOG_DIR=${1:-"logs/training_latest"}

# 刷新间隔 (秒)
REFRESH_INTERVAL=${2:-10}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=============================================="
    echo -e "$1"
    echo -e "==============================================${NC}"
}

print_status() {
    local status=$1
    local name=$2
    case $status in
        "running")
            echo -e "${GREEN}●${NC} ${name}"
            ;;
        "completed")
            echo -e "${BLUE}✓${NC} ${name}"
            ;;
        "failed")
            echo -e "${RED}✗${NC} ${name}"
            ;;
        *)
            echo -e "${YELLOW}?${NC} ${name}"
            ;;
    esac
}

# 检查日志目录
if [ ! -d "${LOG_DIR}" ]; then
    echo "错误: 日志目录不存在: ${LOG_DIR}"
    echo ""
    echo "可用的日志目录:"
    ls -d logs/training_* 2>/dev/null || echo "  无"
    exit 1
fi

# 主循环
while true; do
    clear
    
    echo ""
    print_header "训练进程监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "日志目录: ${LOG_DIR}"
    echo ""
    
    # ==========================================================================
    # 1. GPU 状态
    # ==========================================================================
    print_header "GPU 状态"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util mem_used mem_total temp; do
            # 去除空格
            idx=$(echo $idx | xargs)
            name=$(echo $name | xargs)
            util=$(echo $util | xargs)
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            temp=$(echo $temp | xargs)
            
            # 计算内存百分比
            mem_pct=$((mem_used * 100 / mem_total))
            
            # 状态着色
            if [ "$util" -gt 80 ]; then
                util_color="${GREEN}"
            elif [ "$util" -gt 20 ]; then
                util_color="${YELLOW}"
            else
                util_color="${RED}"
            fi
            
            printf "GPU %s: ${util_color}%3d%%${NC} | 内存: %5dMB / %5dMB (%2d%%) | 温度: %d°C\n" \
                "$idx" "$util" "$mem_used" "$mem_total" "$mem_pct" "$temp"
        done
    else
        echo "nvidia-smi 不可用"
    fi
    echo ""
    
    # ==========================================================================
    # 2. 任务状态
    # ==========================================================================
    print_header "任务状态"
    
    running_tasks=0
    completed_tasks=0
    failed_tasks=0
    
    if [ -f "${LOG_DIR}/running_tasks.txt" ]; then
        while IFS='|' read -r pid name gpu start_time; do
            # 检查进程是否仍在运行
            if kill -0 "$pid" 2>/dev/null; then
                print_status "running" "${name} (PID: ${pid}, GPU: ${gpu})"
                running_tasks=$((running_tasks + 1))
                
                # 获取最新的训练指标
                log_file="${LOG_DIR}/${name}.log"
                if [ -f "${log_file}" ]; then
                    # 尝试从日志中提取最新指标
                    latest_iter=$(grep -oP 'iter:\s*\K\d+' "${log_file}" 2>/dev/null | tail -1)
                    latest_loss=$(grep -oP 'loss:\s*\K[\d.]+' "${log_file}" 2>/dev/null | tail -1)
                    latest_success=$(grep -oP 'success_once:\s*\K[\d.]+' "${log_file}" 2>/dev/null | tail -1)
                    
                    if [ -n "${latest_iter}" ]; then
                        printf "    → iter: %s" "${latest_iter}"
                        [ -n "${latest_loss}" ] && printf ", loss: %.4f" "${latest_loss}"
                        [ -n "${latest_success}" ] && printf ", success: %.2f%%" "${latest_success}"
                        echo ""
                    fi
                fi
            else
                # 检查是否成功完成
                log_file="${LOG_DIR}/${name}.log"
                # 检测多种完成标志：
                # 1. 显式的完成信息
                # 2. wandb 的 "View run" 信息（表示 wandb 正常结束）
                # 3. 100% 进度条完成
                if [ -f "${log_file}" ] && ( \
                    grep -q "Training completed\|训练完成" "${log_file}" 2>/dev/null || \
                    grep -q "View run.*at:" "${log_file}" 2>/dev/null || \
                    grep -qE "100%\|██████████\|.*\[.*<.*,.*it/s" "${log_file}" 2>/dev/null \
                ); then
                    print_status "completed" "${name}"
                    completed_tasks=$((completed_tasks + 1))
                else
                    print_status "failed" "${name} (可能异常退出)"
                    failed_tasks=$((failed_tasks + 1))
                fi
            fi
        done < "${LOG_DIR}/running_tasks.txt"
    else
        echo "无任务信息文件"
    fi
    
    echo ""
    echo -e "统计: ${GREEN}运行中: ${running_tasks}${NC} | ${BLUE}完成: ${completed_tasks}${NC} | ${RED}失败: ${failed_tasks}${NC}"
    echo ""
    
    # ==========================================================================
    # 3. 最近日志
    # ==========================================================================
    print_header "最近日志 (最后 5 行)"
    
    # 找到最新更新的日志文件
    latest_log=$(ls -t ${LOG_DIR}/*.log 2>/dev/null | head -1)
    if [ -n "${latest_log}" ]; then
        echo "文件: $(basename ${latest_log})"
        echo "----------------------------------------"
        tail -5 "${latest_log}" 2>/dev/null || echo "无法读取"
    else
        echo "无日志文件"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "按 Ctrl+C 退出 | 每 ${REFRESH_INTERVAL} 秒刷新"
    echo ""
    echo "有用的命令:"
    echo "  查看完整日志: tail -f ${LOG_DIR}/<name>.log"
    echo "  查看 GPU:     watch -n 1 nvidia-smi"
    echo "  终止任务:     kill <PID>"
    echo "  终止所有:     pkill -f 'rlft.offline.train_maniskill'"
    
    sleep ${REFRESH_INTERVAL}
done

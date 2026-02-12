#!/bin/bash
#
# Monitor DSRL-SAC Sweep v2
# Shows real-time progress for UTD, Architecture, and Buffer Size ablations
#
# Usage: ./monitor_dsrl_v2.sh [--log-dir /tmp/dsrl_sweep_v2] [--interval 5]

LOG_DIR="/tmp/dsrl_sweep_v2"
REFRESH=5  # seconds

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --interval)
            REFRESH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --log-dir DIR     Log directory (default: /tmp/dsrl_sweep_v2)"
            echo "  --interval N      Refresh interval in seconds (default: 5)"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory not found: $LOG_DIR"
    echo "Waiting for sweep to start..."
    while [ ! -d "$LOG_DIR" ]; do
        sleep 5
    done
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

clear
while true; do
    echo -e "${CYAN}==========================================${NC}"
    echo -e "${CYAN}DSRL-SAC Sweep v2 Monitor - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}==========================================${NC}"
    echo ""
    
    # Count status (safe when no logs)
    total=$(find "$LOG_DIR" -maxdepth 1 -name "*.log" 2>/dev/null | wc -l)
    if [ "$total" -eq 0 ]; then
        echo -e "${YELLOW}No logs yet. Waiting for runs to start...${NC}"
        echo ""
        echo "Press Ctrl+C to exit, refreshing every 10s..."
        sleep 10
        clear
        continue
    fi
    
    completed=$(grep -l "completed" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    failed=$(grep -l "failed\|Error\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    running=$((total - completed - failed))
    
    echo -e "Total: ${BLUE}$total${NC} | Completed: ${GREEN}$completed${NC} | Running: ${YELLOW}$running${NC} | Failed: ${RED}$failed${NC}"
    if [ "$total" -gt 0 ]; then
        pct=$(( (completed + failed) * 100 / total ))
        echo -e "Progress: ${CYAN}${pct}%${NC}"
    fi
    echo ""
    
    # Show by category
    echo -e "${CYAN}=== Status by Ablation Category ===${NC}"
    echo ""
    
    # Baseline
    baseline_total=$(find "$LOG_DIR" -maxdepth 1 -name "*baseline*.log" 2>/dev/null | wc -l)
    baseline_completed=$(grep -l "completed" "$LOG_DIR"/*baseline*.log 2>/dev/null | wc -l)
    baseline_failed=$(grep -l "failed\|Error" "$LOG_DIR"/*baseline*.log 2>/dev/null | wc -l)
    echo -e "${MAGENTA}Baseline:${NC} $baseline_completed/$baseline_total completed, $baseline_failed failed"
    
    # UTD experiments
    utd_total=$(find "$LOG_DIR" -maxdepth 1 -name "*utd*.log" 2>/dev/null | wc -l)
    utd_completed=$(grep -l "completed" "$LOG_DIR"/*utd*.log 2>/dev/null | wc -l)
    utd_failed=$(grep -l "failed\|Error" "$LOG_DIR"/*utd*.log 2>/dev/null | wc -l)
    echo -e "${BLUE}UTD Ablation:${NC} $utd_completed/$utd_total completed, $utd_failed failed"
    
    # Architecture experiments
    arch_total=$(find "$LOG_DIR" -maxdepth 1 -name "*arch*.log" 2>/dev/null | wc -l)
    arch_completed=$(grep -l "completed" "$LOG_DIR"/*arch*.log 2>/dev/null | wc -l)
    arch_failed=$(grep -l "failed\|Error" "$LOG_DIR"/*arch*.log 2>/dev/null | wc -l)
    echo -e "${BLUE}Arch Ablation:${NC} $arch_completed/$arch_total completed, $arch_failed failed"
    
    # Buffer experiments
    buffer_total=$(find "$LOG_DIR" -maxdepth 1 -name "*buffer*.log" 2>/dev/null | wc -l)
    buffer_completed=$(grep -l "completed" "$LOG_DIR"/*buffer*.log 2>/dev/null | wc -l)
    buffer_failed=$(grep -l "failed\|Error" "$LOG_DIR"/*buffer*.log 2>/dev/null | wc -l)
    echo -e "${BLUE}Buffer Ablation:${NC} $buffer_completed/$buffer_total completed, $buffer_failed failed"
    
    echo ""
    
    # Show running experiments with their latest output
    echo -e "${CYAN}=== Currently Running ===${NC}"
    running_count=0
    for f in "$LOG_DIR"/*.log; do
        [ -f "$f" ] || continue
        if ! grep -q "completed\|failed" "$f" 2>/dev/null; then
            exp_name=$(basename "$f" .log)
            mod_time=$(date -r "$f" '+%H:%M:%S' 2>/dev/null || echo "??:??:??")
            
            # Extract latest metrics
            latest_timesteps=$(grep -o "total_timesteps[[:space:]]*|[[:space:]]*[0-9]*" "$f" 2>/dev/null | tail -1 | grep -o "[0-9]*$" || echo "?")
            latest_reward=$(grep -o "ep_rew_mean[[:space:]]*|[[:space:]]*[0-9.-]*" "$f" 2>/dev/null | tail -1 | grep -o "[0-9.-]*$" || echo "?")
            
            # Calculate progress percentage if timesteps available
            progress=""
            if [ "$latest_timesteps" != "?" ] && [ -n "$latest_timesteps" ]; then
                pct_done=$(echo "scale=1; $latest_timesteps * 100 / 1000000" | bc 2>/dev/null || echo "?")
                progress=" (${pct_done}%)"
            fi
            
            echo -e "  ${YELLOW}●${NC} [$mod_time] $exp_name$progress"
            echo -e "    timesteps: $latest_timesteps, reward: $latest_reward"
            
            ((running_count+=1))
            if [ $running_count -ge 4 ]; then
                remaining=$((running - running_count))
                if [ $remaining -gt 0 ]; then
                    echo -e "  ${YELLOW}... and $remaining more running${NC}"
                fi
                break
            fi
        fi
    done
    
    if [ $running_count -eq 0 ]; then
        echo -e "  ${GREEN}No experiments currently running${NC}"
    fi
    
    echo ""
    
    # Show recent completions with performance
    echo -e "${CYAN}=== Recent Completions ===${NC}"
    completed_files=$(grep -l "completed" "$LOG_DIR"/*.log 2>/dev/null | xargs -I {} stat --format='%Y %n' {} 2>/dev/null | sort -rn | head -5 | cut -d' ' -f2-)
    if [ -n "$completed_files" ]; then
        for f in $completed_files; do
            exp_name=$(basename "$f" .log)
            mod_time=$(date -r "$f" '+%H:%M:%S' 2>/dev/null || echo "??:??:??")
            # Get final reward
            final_reward=$(grep -o "ep_rew_mean[[:space:]]*|[[:space:]]*[0-9.-]*" "$f" 2>/dev/null | tail -1 | grep -o "[0-9.-]*$" || echo "?")
            echo -e "  ${GREEN}✓${NC} [$mod_time] $exp_name (final reward: $final_reward)"
        done
    else
        echo "  No completions yet"
    fi
    
    echo ""
    
    # Show failures if any
    if [ $failed -gt 0 ]; then
        echo -e "${CYAN}=== Failed Experiments ===${NC}"
        failed_files=$(grep -l "failed\|Error\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | head -5)
        for f in $failed_files; do
            exp_name=$(basename "$f" .log)
            # Get last error line
            error_line=$(grep -i "error\|exception\|traceback" "$f" 2>/dev/null | tail -1 | cut -c1-80)
            echo -e "  ${RED}✗${NC} $exp_name"
            if [ -n "$error_line" ]; then
                echo -e "    ${RED}$error_line${NC}"
            fi
        done
        echo ""
    fi
    
    # GPU utilization (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${CYAN}=== GPU Utilization ===${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read line; do
            IFS=',' read -r idx name util mem_used mem_total <<< "$line"
            util_pct=$(echo "$util" | tr -d ' ')
            mem_pct=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "?")
            echo -e "  GPU $idx: ${util_pct}% util, ${mem_used}/${mem_total} MB (${mem_pct}%)"
        done
        echo ""
    fi
    
    # Estimated time remaining
    if [ $completed -gt 0 ] && [ $running -gt 0 ]; then
        # Get average completion time from completed logs
        avg_time=$(for f in $(grep -l "completed" "$LOG_DIR"/*.log 2>/dev/null | head -3); do
            start=$(head -1 "$f" 2>/dev/null | grep -o '[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}' | head -1)
            end=$(tail -20 "$f" 2>/dev/null | grep -o '[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}' | tail -1)
            if [ -n "$start" ] && [ -n "$end" ]; then
                start_sec=$(date -d "$start" +%s 2>/dev/null || echo 0)
                end_sec=$(date -d "$end" +%s 2>/dev/null || echo 0)
                echo $((end_sec - start_sec))
            fi
        done | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')
        
        if [ "$avg_time" -gt 0 ]; then
            remaining_tasks=$((total - completed - failed))
            # Assuming 2 GPUs running in parallel
            eta_sec=$(( (remaining_tasks * avg_time) / 2 ))
            eta_min=$((eta_sec / 60))
            eta_hr=$((eta_min / 60))
            eta_min=$((eta_min % 60))
            echo -e "${CYAN}=== Estimated Time ===${NC}"
            echo -e "  Avg task time: ${avg_time}s"
            echo -e "  Remaining: ~${eta_hr}h ${eta_min}m (${remaining_tasks} tasks)"
            echo ""
        fi
    fi
    
    echo "Press Ctrl+C to exit, refreshing every ${REFRESH}s..."
    sleep "$REFRESH"
    clear
done

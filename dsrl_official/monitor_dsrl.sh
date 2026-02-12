#!/bin/bash
#
# Monitor DSRL design sweep
# Shows real-time progress and status organized by ablation category
#
# Usage: ./monitor_dsrl.sh [--log-dir /tmp/dsrl_sweep] [--interval 5]

LOG_DIR="/tmp/dsrl_sweep"
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
            echo "  --log-dir DIR     Log directory (default: /tmp/dsrl_sweep)"
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
NC='\033[0m' # No Color

clear
while true; do
    echo -e "${CYAN}==========================================${NC}"
    echo -e "${CYAN}DSRL Sweep Monitor - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
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
    echo -e "${CYAN}=== Status by Category ===${NC}"
    echo ""
    
    # SAC experiments
    sac_total=$(find "$LOG_DIR" -maxdepth 1 -name "dsrl-sac-*.log" 2>/dev/null | wc -l)
    sac_completed=$(grep -l "completed" "$LOG_DIR"/dsrl-sac-*.log 2>/dev/null | wc -l)
    sac_failed=$(grep -l "failed\|Error" "$LOG_DIR"/dsrl-sac-*.log 2>/dev/null | wc -l)
    echo -e "${BLUE}DSRL-SAC:${NC} $sac_completed/$sac_total completed, $sac_failed failed"
    
    # NA experiments
    na_total=$(find "$LOG_DIR" -maxdepth 1 -name "dsrl-na-*.log" 2>/dev/null | wc -l)
    na_completed=$(grep -l "completed" "$LOG_DIR"/dsrl-na-*.log 2>/dev/null | wc -l)
    na_failed=$(grep -l "failed\|Error" "$LOG_DIR"/dsrl-na-*.log 2>/dev/null | wc -l)
    echo -e "${BLUE}DSRL-NA:${NC} $na_completed/$na_total completed, $na_failed failed"
    
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
            
            echo -e "  ${YELLOW}●${NC} [$mod_time] $exp_name"
            echo -e "    timesteps: $latest_timesteps, reward: $latest_reward"
            
            # Show last 5 lines of log (filtered for ANSI codes and carriage returns)
            if [ -s "$f" ]; then
                echo -e "    ${YELLOW}--- Latest Log (last 5 lines) ---${NC}"
                tail -n 5 "$f" 2>/dev/null | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | sed 's/^/    /' | grep -v "^[[:space:]]*$"
                echo ""
            fi
            
            ((running_count+=1))
            if [ $running_count -ge 3 ]; then
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
    
    # Show recent completions
    echo -e "${CYAN}=== Recent Completions ===${NC}"
    completed_files=$(grep -l "completed" "$LOG_DIR"/*.log 2>/dev/null | xargs -I {} stat --format='%Y %n' {} 2>/dev/null | sort -rn | head -5 | cut -d' ' -f2-)
    if [ -n "$completed_files" ]; then
        for f in $completed_files; do
            exp_name=$(basename "$f" .log)
            mod_time=$(date -r "$f" '+%H:%M:%S' 2>/dev/null || echo "??:??:??")
            echo -e "  ${GREEN}✓${NC} [$mod_time] $exp_name"
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
    
    echo "Press Ctrl+C to exit, refreshing every ${REFRESH}s..."
    sleep "$REFRESH"
    clear
done

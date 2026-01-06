#!/bin/bash
#
# Monitor ShortCut Flow design sweep
# Shows real-time progress and status organized by ablation category
#
# Usage: ./monitor_shortcut_flow.sh [--log-dir /tmp/shortcut_flow_grid]

LOG_DIR="/tmp/shortcut_flow_grid"
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
        *)
            shift
            ;;
    esac
done

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory not found: $LOG_DIR"
    exit 1
fi

clear
while true; do
    echo "=========================================="
    echo "ShortCut Flow Sweep Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Count status (safe when no logs)
    total=$(find "$LOG_DIR" -maxdepth 1 -name "*.log" | wc -l)
    if [ "$total" -eq 0 ]; then
        echo "No logs yet. Waiting for runs to start..."
        echo ""
        echo "Press Ctrl+C to exit, refreshing every 10s..."
        sleep 10
        clear
        continue
    fi
    completed=$(grep -l "completed successfully\|DRY RUN" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    failed=$(grep -l "failed\|Error\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    running=$((total - completed - failed))
    
    echo "Total: $total | Completed: $completed | Running: $running | Failed: $failed"
    if [ "$total" -gt 0 ]; then
        echo "Progress: $(( (completed + failed) * 100 / total ))%"
    fi
    echo ""
    
    # Show recent logs
    echo "Recent activity (latest 12 logs, last 2 lines, CR-flattened):"
    ls -t "$LOG_DIR"/*.log 2>/dev/null | head -12 | while read f; do
        exp_name=$(basename "$f" .log)
        mod_time=$(date -r "$f" '+%H:%M:%S')
        echo "- [$mod_time] $exp_name"
        # Flatten carriage returns, strip ANSI, show last 2 lines, truncate long lines
        tr '\r' '\n' < "$f" | sed $'s/\x1b\[[0-9;]*[A-Za-z]//g' | tail -2 | cut -c1-160 | sed 's/^/    /'
    done
    
    echo ""
    echo "Press Ctrl+C to exit, refreshing every ${REFRESH}s..."
    sleep "$REFRESH"
    clear
done

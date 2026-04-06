#!/bin/bash
LOG_FILE="build_full.log"

while true; do
    clear
    echo "=== SQL Error-Correction Dataset Build Monitor ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if process is running
    if ps aux | grep -q "[p]ython.*build_dataset"; then
        echo "Status: RUNNING ✓"
        PID=$(ps aux | grep "[p]ython.*build_dataset" | awk '{print $2}')
        echo "PID: $PID"
        
        # CPU and memory usage
        ps aux | grep "[p]ython.*build_dataset" | awk '{printf "CPU: %s%%  Memory: %s%%\n", $3, $4}'
    else
        echo "Status: COMPLETED or NOT RUNNING"
    fi
    
    echo ""
    echo "=== Latest Log Output ==="
    tail -15 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
    
    echo ""
    echo "=== File Progress ==="
    wc -l *.jsonl 2>/dev/null | tail -5
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done

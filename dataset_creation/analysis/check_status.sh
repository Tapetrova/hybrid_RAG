#!/bin/bash

echo "========================================"
echo "STEP 11 - FULL RUN STATUS"
echo "========================================"

# Check if process is running
if ps aux | grep -v grep | grep "eval_runner.py" > /dev/null; then
    echo "✅ Evaluation process is RUNNING"
else
    echo "❌ Evaluation process is NOT running"
fi

# Count progress
if [ -f "outputs/predictions_20250820_2310.jsonl" ]; then
    COUNT=$(wc -l < outputs/predictions_20250820_2310.jsonl)
    TOTAL=2824  # 706 questions × 4 modes
    PERCENT=$((COUNT * 100 / TOTAL))
    echo "Progress: $COUNT/$TOTAL ($PERCENT%)"
    
    # Estimate time remaining
    if [ $COUNT -gt 0 ]; then
        # Get file creation time and calculate rate
        START_TIME=$(stat -f "%B" outputs/predictions_20250820_2310.jsonl)
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        
        if [ $ELAPSED -gt 0 ]; then
            RATE=$((COUNT * 3600 / ELAPSED))  # per hour
            REMAINING=$((TOTAL - COUNT))
            HOURS_LEFT=$((REMAINING / RATE))
            echo "Rate: ~$RATE evaluations/hour"
            echo "Estimated time remaining: ~$HOURS_LEFT hours"
        fi
    fi
    
    # Show last processed
    echo ""
    echo "Last processed:"
    tail -1 outputs/predictions_20250820_2310.jsonl | python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(f\"  ID: {d['id']}, Mode: {d['mode']}, HR: {d['metrics_sample']['HR']:.3f}\")"
else
    echo "No predictions file found yet"
fi

echo "========================================"
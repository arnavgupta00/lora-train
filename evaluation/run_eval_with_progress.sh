#!/bin/bash
# Run evaluation while capturing both terminal output and logs with proper progress display
# Usage: ./run_eval_with_progress.sh "eval_bird.py args..." output.log

EVAL_CMD="$1"
OUTPUT_LOG="$2"

if [ -z "$EVAL_CMD" ] || [ -z "$OUTPUT_LOG" ]; then
    echo "Usage: $0 'python3 evaluation/eval_bird.py --args...' output.log"
    exit 1
fi

# Create log directory if needed
mkdir -p "$(dirname "$OUTPUT_LOG")"

echo "Starting evaluation: $EVAL_CMD"
echo "Logging to: $OUTPUT_LOG"
echo "View progress in real-time with: tail -f $OUTPUT_LOG"
echo ""

# Run with tee to show output on terminal AND save to log
# This preserves interactive progress bars on terminal while logging everything
eval "$EVAL_CMD" 2>&1 | tee "$OUTPUT_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "Evaluation completed with exit code: $EXIT_CODE"
exit $EXIT_CODE

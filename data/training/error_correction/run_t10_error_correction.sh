#!/bin/bash
# =============================================================================
# T10 Error-Correction Launcher
# =============================================================================
# Runs the trained error-correction model against the existing T10 prediction
# run, then evaluates the repaired predictions.
#
# Default inputs:
#   - T10 baseline predictions/eval from runs/t10_baseline_3090/qwen3-1.7b/without-sampling
#   - Repair model: Qwen/Qwen3.5-2B + LoRA adapter from runs/error_correction_qwen3_5_2b_3090
#
# Usage:
#   ./run_t10_error_correction.sh
#   ./run_t10_error_correction.sh --adapter_dir ./runs/my_repair_adapter --limit 100
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_ID="Qwen/Qwen3.5-2B"
ADAPTER_DIR="$REPO_ROOT/runs/error_correction_qwen3_5_2b_3090"
PREDICTIONS="$REPO_ROOT/runs/t10_baseline_3090/qwen3-1.7b/without-sampling/predictions/predictions_t10.jsonl"
EVAL_RESULTS="$REPO_ROOT/runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/per_example_results.jsonl"
ORIGINAL_EVAL="$REPO_ROOT/runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_report_t10.json"
PROMPTS="$REPO_ROOT/data/training/t10/bird_dev_t10.jsonl"
DB_DIR="$REPO_ROOT/data/bird_eval_datasets/dev_databases"
OUTPUT_DIR="$REPO_ROOT/runs/error_correction_qwen3_5_2b_3090/t10_repair"
ENABLE_THINKING=false
LIMIT=0
MAX_REPAIR_ATTEMPTS=2
GENERATION_BATCH_SIZE=61
MIN_REPAIRABILITY_SCORE=0.5
MAX_NEW_TOKENS=256
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --adapter_dir)
            ADAPTER_DIR="$2"
            shift 2
            ;;
        --predictions)
            PREDICTIONS="$2"
            shift 2
            ;;
        --eval_results)
            EVAL_RESULTS="$2"
            shift 2
            ;;
        --original_eval)
            ORIGINAL_EVAL="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS="$2"
            shift 2
            ;;
        --db_dir)
            DB_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --max_repair_attempts)
            MAX_REPAIR_ATTEMPTS="$2"
            shift 2
            ;;
        --generation_batch_size)
            GENERATION_BATCH_SIZE="$2"
            shift 2
            ;;
        --min_repairability_score)
            MIN_REPAIRABILITY_SCORE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --enable_thinking)
            ENABLE_THINKING=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

for required in "$PREDICTIONS" "$EVAL_RESULTS" "$ORIGINAL_EVAL" "$PROMPTS"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: Required file not found: $required"
        exit 1
    fi
done

if [ ! -d "$DB_DIR" ]; then
    echo "ERROR: Database directory not found: $DB_DIR"
    exit 1
fi

if [ -n "$ADAPTER_DIR" ] && [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: Adapter directory not found: $ADAPTER_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "T10 Error-Correction Run"
echo "============================================================"
echo "Repair model:      $MODEL_ID"
echo "Repair adapter:    ${ADAPTER_DIR:-None}"
echo "Predictions:       $PREDICTIONS"
echo "Eval results:      $EVAL_RESULTS"
echo "Prompts:           $PROMPTS"
echo "DB dir:            $DB_DIR"
echo "Output dir:        $OUTPUT_DIR"
echo "Thinking mode:     $ENABLE_THINKING"
echo "Limit:             $LIMIT"
echo "============================================================"
echo ""

RUN_CMD=(
    python3 "$REPO_ROOT/data/training/t10/error-correction/run_error_correction.py"
    --predictions "$PREDICTIONS"
    --eval_results "$EVAL_RESULTS"
    --prompts "$PROMPTS"
    --db_dir "$DB_DIR"
    --output_dir "$OUTPUT_DIR"
    --model_id "$MODEL_ID"
    --adapter_dir "$ADAPTER_DIR"
    --device "$DEVICE"
    --max_repair_attempts "$MAX_REPAIR_ATTEMPTS"
    --generation_batch_size "$GENERATION_BATCH_SIZE"
    --min_repairability_score "$MIN_REPAIRABILITY_SCORE"
    --max_new_tokens "$MAX_NEW_TOKENS"
)

if [ "$ENABLE_THINKING" = true ]; then
    RUN_CMD+=(--enable_thinking)
fi

if [ "$LIMIT" -gt 0 ]; then
    RUN_CMD+=(--limit "$LIMIT")
fi

"${RUN_CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/run_error_correction.log"

python3 "$REPO_ROOT/data/training/t10/error-correction/evaluate_repaired.py" \
    --repaired_predictions "$OUTPUT_DIR/repaired_predictions_t10.jsonl" \
    --original_eval "$ORIGINAL_EVAL" \
    --prompts "$PROMPTS" \
    --db_dir "$DB_DIR" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/evaluate_repaired.log"

echo ""
echo "============================================================"
echo "Finished"
echo "============================================================"
echo "Repaired predictions: $OUTPUT_DIR/repaired_predictions_t10.jsonl"
echo "Repair summary:       $OUTPUT_DIR/repair_summary_t10.json"
echo "Evaluation report:    $OUTPUT_DIR/repair_eval_report_t10.json"

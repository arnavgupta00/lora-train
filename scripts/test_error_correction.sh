#!/bin/bash
# Sequential Error Correction Testing Script
# Tests 3 different configurations to find the best approach
#
# Test 1: SFT LoRA model (no_think mode)
# Test 2: Base model (no_think mode) 
# Test 3: Base model with thinking mode (more tokens)
#
# Usage: ./scripts/test_error_correction.sh [--limit N] [--full]

set -e

# Configuration
MODEL_ID="Qwen/Qwen3-1.7B"
ADAPTER_DIR="/workspace/outputs/qwen3-1.7b-sft-20260402_152110"
BIRD_JSON="./bird_eval/dev.json"
DB_DIR="./bird_eval/dev_databases"
LIMIT="50"  # Default 50 examples

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --full)
            LIMIT="0"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================================"
echo "ERROR CORRECTION TESTING - 3 CONFIGURATIONS"
echo "============================================================"
echo "Model: $MODEL_ID"
echo "Adapter: $ADAPTER_DIR"
if [[ "$LIMIT" == "0" ]]; then
    echo "Limit: ALL 1534 examples"
else
    echo "Limit: $LIMIT examples (use --full for all)"
fi
echo "============================================================"
echo ""

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./eval_ec_comparison_${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"

echo "Results will be saved to: $BASE_OUTPUT_DIR"
echo ""

# ============================================================
# TEST 1: SFT LoRA Model (no_think mode)
# ============================================================
echo "============================================================"
echo "[TEST 1/3] SFT LoRA Model - no_think mode"
echo "============================================================"
echo "Config:"
echo "  - Model: SFT fine-tuned with LoRA"
echo "  - Initial: no_think, 256 tokens"
echo "  - Correction: no_think, 512 tokens"
echo ""

OUTPUT_DIR_1="${BASE_OUTPUT_DIR}/test1_sft_lora"

python3 evaluation/eval_error_correction.py \
    --model_id "$MODEL_ID" \
    --adapter_dir "$ADAPTER_DIR" \
    --bird_dev_json "$BIRD_JSON" \
    --db_dir "$DB_DIR" \
    --output_dir "$OUTPUT_DIR_1" \
    --initial_max_tokens 256 \
    --correction_max_tokens 512 \
    --max_retries 3 \
    --limit "$LIMIT" 2>&1 | tee "${OUTPUT_DIR_1}.log"

echo ""
echo "[TEST 1 COMPLETE]"
echo ""

# ============================================================
# TEST 2: Base Model Only (no LoRA, no_think mode)
# ============================================================
echo "============================================================"
echo "[TEST 2/3] Base Model - no_think mode"
echo "============================================================"
echo "Config:"
echo "  - Model: Base Qwen3-1.7B (NO LoRA)"
echo "  - Initial: no_think, 256 tokens"
echo "  - Correction: no_think, 512 tokens"
echo ""

OUTPUT_DIR_2="${BASE_OUTPUT_DIR}/test2_base_nothink"

python3 evaluation/eval_error_correction.py \
    --model_id "$MODEL_ID" \
    --bird_dev_json "$BIRD_JSON" \
    --db_dir "$DB_DIR" \
    --output_dir "$OUTPUT_DIR_2" \
    --initial_max_tokens 256 \
    --correction_max_tokens 512 \
    --max_retries 3 \
    --limit "$LIMIT" 2>&1 | tee "${OUTPUT_DIR_2}.log"

echo ""
echo "[TEST 2 COMPLETE]"
echo ""

# ============================================================
# TEST 3: Base Model with Thinking Mode (more tokens)
# ============================================================
echo "============================================================"
echo "[TEST 3/3] Base Model - THINKING mode"
echo "============================================================"
echo "Config:"
echo "  - Model: Base Qwen3-1.7B (NO LoRA)"
echo "  - Initial: thinking, 512 tokens (allows reasoning)"
echo "  - Correction: thinking, 1024 tokens (deep reasoning)"
echo ""

OUTPUT_DIR_3="${BASE_OUTPUT_DIR}/test3_base_thinking"

python3 evaluation/eval_error_correction.py \
    --model_id "$MODEL_ID" \
    --bird_dev_json "$BIRD_JSON" \
    --db_dir "$DB_DIR" \
    --output_dir "$OUTPUT_DIR_3" \
    --initial_max_tokens 512 \
    --correction_max_tokens 1024 \
    --max_retries 3 \
    --thinking_mode think \
    --limit "$LIMIT" 2>&1 | tee "${OUTPUT_DIR_3}.log"

echo ""
echo "[TEST 3 COMPLETE]"
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "ALL TESTS COMPLETE - RESULTS COMPARISON"
echo "============================================================"
echo ""

# Extract and compare results
echo "| Test | Config | Initial Acc | Final Acc | Improvement | Fixed |"
echo "|------|--------|-------------|-----------|-------------|-------|"

for test_dir in "$OUTPUT_DIR_1" "$OUTPUT_DIR_2" "$OUTPUT_DIR_3"; do
    if [[ -f "$test_dir/evaluation_report.json" ]]; then
        name=$(basename "$test_dir")
        python3 -c "
import json
with open('$test_dir/evaluation_report.json') as f:
    d = json.load(f)
init = d.get('initial_execution_accuracy', 0)
final = d.get('final_execution_accuracy', 0)
improve = d.get('accuracy_improvement', 0)
fixed = d.get('correction_stats', {}).get('fixed_total', 0)
print(f'| $name | {init}% | {final}% | +{improve}% | {fixed} |')
" 2>/dev/null || echo "| $name | ERROR reading results |"
    fi
done

echo ""
echo "============================================================"
echo "Full results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "To view individual reports:"
echo "  cat $OUTPUT_DIR_1/evaluation_report.json | python3 -m json.tool"
echo "  cat $OUTPUT_DIR_2/evaluation_report.json | python3 -m json.tool"
echo "  cat $OUTPUT_DIR_3/evaluation_report.json | python3 -m json.tool"
echo "============================================================"

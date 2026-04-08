#!/bin/bash
set -euo pipefail

# One-shot end-to-end run for BIRD new dev set.
# Defaults use requested HF adapters and high batching (25).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
PRED_ADAPTER="${PRED_ADAPTER:-Arnav3035/garuda-sql-2b}"
REPAIR_ADAPTER="${REPAIR_ADAPTER:-Arnav3035/error-correction-qwen3-5-2b-3090}"
BATCH_SIZE="${BATCH_SIZE:-25}"
EVAL_WORKERS="${EVAL_WORKERS:-25}"
RUN_ROOT="${1:-runs/t12_bird_new_auto}"

NEW_DEV_JSON="data/bird_eval_datasets_new/data/dev_20251106-00000-of-00001.json"
DB_DIR="data/bird_eval_datasets/dev_databases"
PROMPTS_FILE="data/training/t12/bird_new/bird_new_dev_t12.jsonl"

PRED_DIR="$RUN_ROOT/predictions"
EVAL_DIR="$RUN_ROOT/eval"
REPAIR_DIR="$RUN_ROOT/error_correction"
REPAIR_EVAL_DIR="$RUN_ROOT/error_correction/eval"

echo "[1/6] Fetching BIRD new dataset from HF..."
hf download birdsql/bird_sql_dev_20251106 \
  --repo-type dataset \
  --local-dir data/bird_eval_datasets_new

echo "[2/6] Building prompts..."
"$PYTHON_BIN" data/training/t12/build_eval_prompts.py \
  --bird_dev_json "$NEW_DEV_JSON" \
  --db_dir "$DB_DIR" \
  --output "$PROMPTS_FILE"

echo "[3/6] Predicting with adapter: $PRED_ADAPTER (batch=$BATCH_SIZE)..."
"$PYTHON_BIN" data/training/t12/predict_t12.py \
  --model_id "$MODEL_ID" \
  --adapter_dir "$PRED_ADAPTER" \
  --prompts_file "$PROMPTS_FILE" \
  --output_dir "$PRED_DIR" \
  --batch_size "$BATCH_SIZE"

echo "[4/6] Evaluating baseline predictions..."
"$PYTHON_BIN" data/training/t12/evaluate_t12.py \
  --predictions_file "$PRED_DIR/predictions_t12.jsonl" \
  --output_dir "$EVAL_DIR" \
  --dev_json "$NEW_DEV_JSON" \
  --db_dir "$DB_DIR" \
  --max_workers "$EVAL_WORKERS" \
  --no_tied_append

echo "[5/6] Running error correction with adapter: $REPAIR_ADAPTER (batch=$BATCH_SIZE)..."
"$PYTHON_BIN" data/training/t12/error-correction/run_error_correction.py \
  --predictions "$PRED_DIR/predictions_t12.jsonl" \
  --eval_results "$EVAL_DIR/per_example_results.jsonl" \
  --prompts "$PROMPTS_FILE" \
  --db_dir "$DB_DIR" \
  --output_dir "$REPAIR_DIR" \
  --model_id "$MODEL_ID" \
  --adapter_path "$REPAIR_ADAPTER" \
  --max_repair_attempts 2 \
  --generation_batch_size "$BATCH_SIZE"

echo "[6/6] Evaluating repaired predictions..."
"$PYTHON_BIN" data/training/t12/error-correction/evaluate_repaired.py \
  --repaired_predictions "$REPAIR_DIR/repaired_predictions_t12.jsonl" \
  --original_eval "$EVAL_DIR/eval_report_t12.json" \
  --prompts "$PROMPTS_FILE" \
  --db_dir "$DB_DIR" \
  --output_dir "$REPAIR_EVAL_DIR"

echo ""
echo "DONE"
echo "Baseline eval: $EVAL_DIR/eval_report_t12.json"
echo "Repaired eval: $REPAIR_EVAL_DIR/repair_eval_report_t12.json"

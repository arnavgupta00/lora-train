#!/bin/bash
set -euo pipefail

# Usage:
# ./data/training/t12/bird_new/error_correction_bird_new.sh <predictions_jsonl> <eval_results_jsonl> <output_dir> <model_id> [adapter_path]

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <predictions_jsonl> <eval_results_jsonl> <output_dir> <model_id> [adapter_path]"
  exit 1
fi

PREDICTIONS="$1"
EVAL_RESULTS="$2"
OUTPUT_DIR="$3"
MODEL_ID="$4"
ADAPTER_PATH="${5:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

CMD=(python data/training/t12/error-correction/run_error_correction.py
  --predictions "$PREDICTIONS"
  --eval_results "$EVAL_RESULTS"
  --prompts data/training/t12/bird_new/bird_new_dev_t12.jsonl
  --db_dir data/bird_eval_datasets/dev_databases
  --output_dir "$OUTPUT_DIR"
  --model_id "$MODEL_ID"
  --max_repair_attempts 2
)

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter_path "$ADAPTER_PATH")
fi

"${CMD[@]}"

#!/bin/bash
set -euo pipefail

# Usage:
# ./data/training/t12/bird_new/eval_repaired_bird_new.sh <repaired_predictions_jsonl> <original_eval_report_json> <output_dir>

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <repaired_predictions_jsonl> <original_eval_report_json> <output_dir>"
  exit 1
fi

REPAIRED_PREDICTIONS="$1"
ORIGINAL_EVAL="$2"
OUTPUT_DIR="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

python data/training/t12/error-correction/evaluate_repaired.py \
  --repaired_predictions "$REPAIRED_PREDICTIONS" \
  --original_eval "$ORIGINAL_EVAL" \
  --prompts data/training/t12/bird_new/bird_new_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir "$OUTPUT_DIR"

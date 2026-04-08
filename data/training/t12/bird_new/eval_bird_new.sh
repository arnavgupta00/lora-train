#!/bin/bash
set -euo pipefail

# Usage:
# ./data/training/t12/bird_new/eval_bird_new.sh <predictions_file> <output_dir>

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <predictions_file> <output_dir>"
  exit 1
fi

PREDICTIONS_FILE="$1"
OUTPUT_DIR="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

python data/training/t12/evaluate_t12.py \
  --predictions_file "$PREDICTIONS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --dev_json data/bird_eval_datasets_new/data/dev_20251106-00000-of-00001.json \
  --db_dir data/bird_eval_datasets/dev_databases \
  --no_tied_append

#!/bin/bash
set -euo pipefail

# Usage:
# ./data/training/t12/bird_new/predict_bird_new.sh <model_id> <output_dir> [adapter_dir]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_id> <output_dir> [adapter_dir]"
  exit 1
fi

MODEL_ID="$1"
OUTPUT_DIR="$2"
ADAPTER_DIR="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

CMD=(python data/training/t12/predict_t12.py
  --model_id "$MODEL_ID"
  --prompts_file data/training/t12/bird_new/bird_new_dev_t12.jsonl
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$ADAPTER_DIR" ]]; then
  CMD+=(--adapter_dir "$ADAPTER_DIR")
fi

"${CMD[@]}"

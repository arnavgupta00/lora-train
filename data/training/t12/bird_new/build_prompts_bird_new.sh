#!/bin/bash
set -euo pipefail

# Build T12 prompts from BIRD 2025-11-06 dev set.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

python data/training/t12/build_eval_prompts.py \
  --bird_dev_json data/bird_eval_datasets_new/data/dev_20251106-00000-of-00001.json \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output data/training/t12/bird_new/bird_new_dev_t12.jsonl

echo "Built prompts: data/training/t12/bird_new/bird_new_dev_t12.jsonl"

#!/bin/bash
set -euo pipefail

# Fetch latest BIRD-SQL dev update from Hugging Face.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

hf download birdsql/bird_sql_dev_20251106 \
  --repo-type dataset \
  --local-dir data/bird_eval_datasets_new

echo "Downloaded to: data/bird_eval_datasets_new"

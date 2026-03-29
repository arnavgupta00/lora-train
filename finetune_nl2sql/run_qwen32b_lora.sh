#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d /runpod-volume ]]; then
  export HF_HOME="/runpod-volume/hf"
  export TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
  export HF_DATASETS_CACHE="/runpod-volume/hf/datasets"
  OUT_BASE="/runpod-volume/outputs"
else
  # No network volume: keep caches on the pod's /workspace volume disk (not the ephemeral container disk).
  export HF_HOME="/workspace/hf"
  export TRANSFORMERS_CACHE="/workspace/hf/transformers"
  export HF_DATASETS_CACHE="/workspace/hf/datasets"
  OUT_BASE="$ROOT_DIR/outputs"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$OUT_BASE"

python3 -m pip install -U pip
python3 -m pip install -r finetune_nl2sql/requirements.txt

if [[ ! -f finetune_nl2sql/private_key.py ]]; then
  if [[ -n "${NL2SQL_ADMIN_API_KEY:-}" ]]; then
    python3 - <<'PY'
import os
key = os.environ.get("NL2SQL_ADMIN_API_KEY", "").strip()
if not key:
  raise SystemExit("NL2SQL_ADMIN_API_KEY is empty")
with open("finetune_nl2sql/private_key.py", "w", encoding="utf-8") as f:
  f.write('ADMIN_API_KEY = "' + key.replace('"', '\\"') + '"\n')
PY
  else
    echo "Missing finetune_nl2sql/private_key.py (needed for execution-match eval)."
    echo "Set NL2SQL_ADMIN_API_KEY env var or create the file from private_key.py.example."
    exit 1
  fi
fi

MODEL_ID="Qwen/Qwen2.5-Coder-32B-Instruct"
RUN_NAME="qwen2.5-coder-32b-instruct-lora-$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_BASE/$RUN_NAME"

DATASET_DIR="${DATASET_DIR:-}"
if [[ -z "$DATASET_DIR" ]]; then
  if [[ -f dataset/t3_test1000_rebalanced/all-all-train.qwen.jsonl ]]; then
    DATASET_DIR="dataset/t3_test1000_rebalanced"
  elif [[ -f dataset/t3/all-all-train.qwen.jsonl ]]; then
    DATASET_DIR="dataset/t3"
  elif [[ "${ALLOW_DATASET_FALLBACK_T2:-0}" == "1" && -f dataset/t2/all-all-train.qwen.jsonl ]]; then
    DATASET_DIR="dataset/t2"
  else
    echo "ERROR: No t3 dataset found in repo and DATASET_DIR is not set."
    echo "Expected one of:"
    echo "  dataset/t3_test1000_rebalanced/all-all-train.qwen.jsonl"
    echo "  dataset/t3/all-all-train.qwen.jsonl"
    echo ""
    echo "Fix: copy your dataset into the pod and set DATASET_DIR to that folder."
    echo "If you intentionally want to run on t2, set ALLOW_DATASET_FALLBACK_T2=1."
    echo ""
    echo "Datasets present under ./dataset:"
    ls -la dataset || true
    exit 1
  fi
fi
TRAIN_JSONL="$DATASET_DIR/all-all-train.qwen.jsonl"
DEV_JSONL="$DATASET_DIR/all-all-dev.qwen.jsonl"
TEST_JSONL="$DATASET_DIR/all-all-test.qwen.jsonl"

for f in "$TRAIN_JSONL" "$DEV_JSONL" "$TEST_JSONL"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing dataset file: $f"
    exit 1
  fi
done
echo "Using dataset dir: $DATASET_DIR"
wc -l "$TRAIN_JSONL" "$DEV_JSONL" "$TEST_JSONL" || true

SEQ_LEN="${QWEN32_MAX_SEQ_LEN:-1024}"
TRAIN_BS="${QWEN32_TRAIN_BS:-1}"
EVAL_BS="${QWEN32_EVAL_BS:-1}"
GRAD_ACC="${QWEN32_GRAD_ACC:-8}"

python3 finetune_nl2sql/train_lora.py \
  --model_id "$MODEL_ID" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --output_dir "$OUT_DIR" \
  --max_seq_len "$SEQ_LEN" \
  --pack \
  --per_device_train_batch_size "$TRAIN_BS" \
  --per_device_eval_batch_size "$EVAL_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate 1e-4 \
  --num_train_epochs "${EPOCHS_32B:-2}" \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --gradient_checkpointing \
  --dataloader_num_workers "${DL_WORKERS:-4}" \
  --tf32

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  EVAL_EXTRA_ARGS=()
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    EVAL_EXTRA_ARGS+=(--limit "$EVAL_LIMIT")
  fi

  if [[ "${EVAL_BASE:-0}" == "1" ]]; then
    python3 finetune_nl2sql/eval_exec.py \
      --base_model_id "$MODEL_ID" \
      --test_jsonl "$TEST_JSONL" \
      --out_dir "$OUT_DIR" \
      --max_new_tokens 256 \
      --gen_batch_size "${EVAL_GEN_BS_32B_BASE:-4}" \
      --validator_batch_size "${EVAL_VAL_BS:-50}" \
      --validator_parallelism "${EVAL_VAL_PAR:-4}" \
      "${EVAL_EXTRA_ARGS[@]}"
  fi

  python3 finetune_nl2sql/eval_exec.py \
    --base_model_id "$MODEL_ID" \
    --adapter_dir "$OUT_DIR" \
    --test_jsonl "$TEST_JSONL" \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 256 \
    --gen_batch_size "${EVAL_GEN_BS_32B:-4}" \
    --validator_batch_size "${EVAL_VAL_BS:-50}" \
    --validator_parallelism "${EVAL_VAL_PAR:-4}" \
    "${EVAL_EXTRA_ARGS[@]}"
fi

echo "Done: $OUT_DIR"

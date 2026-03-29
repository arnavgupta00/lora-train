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
  f.write('ADMIN_API_KEY = "' + key.replace('"', '\\"') + '"\\n')
PY
  else
    echo "Missing finetune_nl2sql/private_key.py (needed for execution-match eval)."
    echo "Set NL2SQL_ADMIN_API_KEY env var or create the file from private_key.py.example."
    exit 1
  fi
fi

MODEL_14B_ID="Qwen/Qwen2.5-14B-Instruct"
MODEL_32B_ID="Qwen/Qwen2.5-Coder-32B-Instruct"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_14B="$OUT_BASE/qwen2.5-14b-instruct-lora-$STAMP"
OUT_32B="$OUT_BASE/qwen2.5-coder-32b-instruct-lora-$STAMP"

DATASET_DIR="${DATASET_DIR:-}"
if [[ -z "$DATASET_DIR" ]]; then
  if [[ -f dataset/t3_test1000_rebalanced/all-all-train.qwen.jsonl ]]; then
    DATASET_DIR="dataset/t3_test1000_rebalanced"
  elif [[ -f dataset/t3/all-all-train.qwen.jsonl ]]; then
    DATASET_DIR="dataset/t3"
  else
    DATASET_DIR="dataset/t2"
  fi
fi
TRAIN_JSONL="$DATASET_DIR/all-all-train.qwen.jsonl"
DEV_JSONL="$DATASET_DIR/all-all-dev.qwen.jsonl"
TEST_JSONL="$DATASET_DIR/all-all-test.qwen.jsonl"

SEQ_14B="${QWEN14_MAX_SEQ_LEN:-1024}"
SEQ_32B="${QWEN32_MAX_SEQ_LEN:-1024}"

echo "Training 14B -> $OUT_14B"
python3 finetune_nl2sql/train_lora.py \
  --model_id "$MODEL_14B_ID" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --output_dir "$OUT_14B" \
  --max_seq_len "$SEQ_14B" \
  --pack \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs "${EPOCHS_14B:-3}" \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --gradient_checkpointing \
  --dataloader_num_workers "${DL_WORKERS:-4}" \
  --tf32

echo "Training 32B -> $OUT_32B"
python3 finetune_nl2sql/train_lora.py \
  --model_id "$MODEL_32B_ID" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --output_dir "$OUT_32B" \
  --max_seq_len "$SEQ_32B" \
  --pack \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
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

echo "Eval 14B -> $OUT_14B"
python3 finetune_nl2sql/eval_exec.py \
  --base_model_id "$MODEL_14B_ID" \
  --adapter_dir "$OUT_14B" \
  --test_jsonl "$TEST_JSONL" \
  --out_dir "$OUT_14B" \
  --max_new_tokens 256 \
  --gen_batch_size "${EVAL_GEN_BS_14B:-8}" \
  --validator_batch_size "${EVAL_VAL_BS:-50}" \
  --validator_parallelism "${EVAL_VAL_PAR:-4}"

echo "Eval 32B -> $OUT_32B"
python3 finetune_nl2sql/eval_exec.py \
  --base_model_id "$MODEL_32B_ID" \
  --adapter_dir "$OUT_32B" \
  --test_jsonl "$TEST_JSONL" \
  --out_dir "$OUT_32B" \
  --max_new_tokens 256 \
  --gen_batch_size "${EVAL_GEN_BS_32B:-4}" \
  --validator_batch_size "${EVAL_VAL_BS:-50}" \
  --validator_parallelism "${EVAL_VAL_PAR:-4}"

echo "Done:"
echo "  14B: $OUT_14B"
echo "  32B: $OUT_32B"

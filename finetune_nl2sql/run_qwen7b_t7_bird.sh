#!/usr/bin/env bash
###############################################################################
# RunPod Training Script for Qwen2.5-7B on t7 Dataset (BIRD Benchmark)
# 
# Usage: 
#   nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &
#
# Or with environment variables:
#   EPOCHS=3 LR=5e-5 nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &
#
# Required:
#   - NL2SQL_ADMIN_API_KEY environment variable (for execution-match eval)
#   - dataset/t7/ directory with train.jsonl, dev.jsonl, test.jsonl
#
# RunPod Configuration Recommendation:
#   - GPU: RTX A6000 (48GB) or A100 40GB 
#   - Storage: 50GB+ network volume at /runpod-volume
#   - Container: runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  Qwen2.5-7B Training on t7 Dataset"
echo "  Target: BIRD Benchmark (beat Claude Opus)"
echo "=============================================="
echo "Start time: $(date)"

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SIZE="7b"

# Training hyperparameters (optimized for BIRD benchmark)
EPOCHS="${EPOCHS:-1}"                    # 1 epoch (3 epochs = 26hrs, diminishing returns)
LR="${LR:-5e-5}"                         # 5e-5 (was 2e-4, lower = better for fine-tuning)
SEQ_LEN="${SEQ_LEN:-1024}"               # 1024 (most queries <1024 tokens, 2048 very slow)
TRAIN_BS="${TRAIN_BS:-2}"                # 2 for 24GB GPUs (4 caused OOM on RTX 4090)
EVAL_BS="${EVAL_BS:-2}"                  # 2 for 24GB GPUs
GRAD_ACC="${GRAD_ACC:-16}"               # Effective batch size = 2 * 16 = 32
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"      # 10% warmup

# LoRA configuration
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Evaluation
EVAL_BASE="${EVAL_BASE:-1}"              # Also evaluate base model for comparison
SKIP_EVAL="${SKIP_EVAL:-0}"

# =============================================================================
# Directory Setup
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d /runpod-volume ]]; then
  export HF_HOME="/runpod-volume/hf"
  export TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
  export HF_DATASETS_CACHE="/runpod-volume/hf/datasets"
  OUT_BASE="/runpod-volume/outputs"
  echo "Using RunPod volume for caching"
else
  export HF_HOME="/workspace/hf"
  export TRANSFORMERS_CACHE="/workspace/hf/transformers"
  export HF_DATASETS_CACHE="/workspace/hf/datasets"
  OUT_BASE="$ROOT_DIR/outputs"
  echo "Using workspace for caching"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$OUT_BASE"

# Run name with timestamp
RUN_NAME="qwen2.5-7b-t7-bird-$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$OUT_BASE/$RUN_NAME}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/run.log"
echo "Output directory: $OUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

echo ">>> Installing dependencies..."
python3 -m pip install -U pip setuptools wheel
python3 -m pip install accelerate datasets peft requests safetensors bitsandbytes
python3 -m pip install "transformers>=4.45.0,<5"
python3 -m pip install hf-transfer || echo "Warning: hf-transfer install failed (optional)"

# Check for API key
if [[ ! -f finetune_nl2sql/private_key.py ]]; then
  if [[ -n "${NL2SQL_ADMIN_API_KEY:-}" ]]; then
    echo "Creating private_key.py from environment..."
    python3 - <<'PY'
import os
key = os.environ.get("NL2SQL_ADMIN_API_KEY", "").strip()
if not key:
    raise SystemExit("NL2SQL_ADMIN_API_KEY is empty")
with open("finetune_nl2sql/private_key.py", "w") as f:
    f.write(f'ADMIN_API_KEY = "{key}"\n')
print("Created finetune_nl2sql/private_key.py")
PY
  else
    echo "WARNING: Missing NL2SQL_ADMIN_API_KEY - eval may fail"
  fi
fi

# =============================================================================
# Dataset Setup
# =============================================================================

echo ""
echo ">>> Setting up dataset..."

# Use t7 dataset
DATASET_DIR="${DATASET_DIR:-dataset/t7}"

# t7 uses different file names
if [[ -f "$DATASET_DIR/train.jsonl" ]]; then
  TRAIN_JSONL="$DATASET_DIR/train.jsonl"
  DEV_JSONL="$DATASET_DIR/dev.jsonl"
  TEST_JSONL="$DATASET_DIR/test.jsonl"
elif [[ -f "$DATASET_DIR/all-all-train.qwen.jsonl" ]]; then
  # Fallback to old naming
  TRAIN_JSONL="$DATASET_DIR/all-all-train.qwen.jsonl"
  DEV_JSONL="$DATASET_DIR/all-all-dev.qwen.jsonl"
  TEST_JSONL="$DATASET_DIR/all-all-test.qwen.jsonl"
else
  echo "ERROR: Dataset not found in $DATASET_DIR"
  echo "Expected files: train.jsonl, dev.jsonl, test.jsonl"
  echo ""
  echo "Available datasets:"
  ls -la dataset/ 2>/dev/null || echo "No dataset directory found"
  exit 1
fi

# Verify files exist
for f in "$TRAIN_JSONL" "$DEV_JSONL" "$TEST_JSONL"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing dataset file: $f"
    exit 1
  fi
done

echo "Using dataset: $DATASET_DIR"
echo "Dataset sizes:"
wc -l "$TRAIN_JSONL" "$DEV_JSONL" "$TEST_JSONL"
echo ""

# =============================================================================
# GPU Detection
# =============================================================================

echo ">>> Detecting GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
  VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
  echo "Total VRAM: ${VRAM_MIB} MiB"
else
  echo "WARNING: nvidia-smi not found"
  VRAM_MIB=0
fi

# 7B model fits in 24GB+ without 8-bit
LOAD8=0
LOAD8_ARGS=()
if [[ "${FORCE_8BIT:-0}" == "1" ]] || [[ -n "${VRAM_MIB:-}" && "${VRAM_MIB}" -lt 24000 ]]; then
  echo "Enabling 8-bit base load (low VRAM detected)"
  LOAD8=1
  LOAD8_ARGS+=(--load_in_8bit)
fi

echo ""

# =============================================================================
# Training
# =============================================================================

echo "=============================================="
echo ">>> Starting Training"
echo "=============================================="
echo "Model: $MODEL_ID"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Sequence Length: $SEQ_LEN"
echo "Batch Size: $TRAIN_BS x $GRAD_ACC = $((TRAIN_BS * GRAD_ACC)) effective"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo "Output: $OUT_DIR"
echo ""

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
  echo "Resuming from: $RESUME_FROM"
fi

python3 -u finetune_nl2sql/train_lora.py \
  --model_id "$MODEL_ID" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --output_dir "$OUT_DIR" \
  --max_seq_len "$SEQ_LEN" \
  --pack \
  --per_device_train_batch_size "$TRAIN_BS" \
  --per_device_eval_batch_size "$EVAL_BS" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --warmup_ratio "$WARMUP_RATIO" \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200 \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --gradient_checkpointing \
  --dataloader_num_workers 4 \
  --tf32 \
  "${LOAD8_ARGS[@]}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo ">>> Training complete!"
echo ""

# =============================================================================
# Evaluation
# =============================================================================

if [[ "$SKIP_EVAL" != "1" ]]; then
  EVAL_EXTRA_ARGS=()
  if [[ -n "${EVAL_LIMIT:-}" ]]; then
    EVAL_EXTRA_ARGS+=(--limit "$EVAL_LIMIT")
  fi

  # Evaluate BASE model (for comparison)
  if [[ "$EVAL_BASE" == "1" ]]; then
    echo "=============================================="
    echo ">>> Evaluating BASE model (no LoRA)"
    echo "=============================================="
    
    python3 finetune_nl2sql/eval_exec.py \
      --base_model_id "$MODEL_ID" \
      --test_jsonl "$TEST_JSONL" \
      --out_dir "$OUT_DIR" \
      --max_new_tokens 512 \
      --gen_batch_size 8 \
      --validator_batch_size 50 \
      --validator_parallelism 4 \
      "${LOAD8_ARGS[@]}" \
      "${EVAL_EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    
    echo ""
  fi

  # Evaluate FINE-TUNED model (with LoRA)
  echo "=============================================="
  echo ">>> Evaluating FINE-TUNED model (with LoRA)"
  echo "=============================================="
  
  python3 -u finetune_nl2sql/eval_exec.py \
    --base_model_id "$MODEL_ID" \
    --adapter_dir "$OUT_DIR" \
    --test_jsonl "$TEST_JSONL" \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 512 \
    --gen_batch_size 2 \
    --validator_batch_size 50 \
    --validator_parallelism 4 \
    "${LOAD8_ARGS[@]}" \
    "${EVAL_EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo ">>> COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Output directory: $OUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Results:"
ls -la "$OUT_DIR"/*.json 2>/dev/null || echo "No JSON results yet"
echo ""

# Show eval results if available
if [[ -f "$OUT_DIR/eval_report.lora.json" ]]; then
  echo "LoRA Model Results:"
  cat "$OUT_DIR/eval_report.lora.json"
  echo ""
fi

if [[ -f "$OUT_DIR/eval_report.base.json" ]]; then
  echo "Base Model Results:"
  cat "$OUT_DIR/eval_report.base.json"
  echo ""
fi

echo "Done: $OUT_DIR"

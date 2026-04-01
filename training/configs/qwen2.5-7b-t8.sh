#!/usr/bin/env bash
###############################################################################
# RunPod Training Script for Qwen2.5-7B on T8 Dataset (BIRD Benchmark)
# 
# T8 Improvements over T7:
#   - 100% DDL schema format (matching BIRD eval)
#   - CASE upsampled: 5% → 23%
#   - CTE upsampled: 0.5% → 5%
#   - Window functions upsampled: 0.5% → 5%
#   - Complex column name training (backticks, spaces)
#   - 22,782 training examples (vs 16,699 in t7)
#
# Usage: 
#   nohup bash training/configs/qwen2.5-7b-t8.sh > run.log 2>&1 &
#
# Or with environment variables:
#   EPOCHS=2 LR=5e-5 nohup bash training/configs/qwen2.5-7b-t8.sh > run.log 2>&1 &
#
# RunPod Configuration Recommendation:
#   - GPU: RTX A6000 (48GB) or A100 40GB 
#   - Storage: 50GB+ network volume at /runpod-volume
#   - Container: runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  Qwen2.5-7B Training on T8 Dataset"
echo "  Target: BIRD 55-60% (from 44.26%)"
echo "=============================================="
echo "Start time: $(date)"

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SIZE="7b"

# Training hyperparameters (optimized for T8)
EPOCHS="${EPOCHS:-2}"                    # 2 epochs (more data, less overfitting risk)
LR="${LR:-5e-5}"                         # 5e-5 (proven good from t7)
SEQ_LEN="${SEQ_LEN:-1024}"               # 1024 tokens
TRAIN_BS="${TRAIN_BS:-2}"                # 2 for 24GB GPUs
EVAL_BS="${EVAL_BS:-2}"                  # 2 for 24GB GPUs
GRAD_ACC="${GRAD_ACC:-16}"               # Effective batch size = 2 * 16 = 32
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"      # 10% warmup

# LoRA configuration (increased capacity)
LORA_R="${LORA_R:-32}"                   # 32 (was 16 in t7)
LORA_ALPHA="${LORA_ALPHA:-64}"           # 64 (was 32 in t7)
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Evaluation
EVAL_BASE="${EVAL_BASE:-0}"              # Skip base eval (we know the baseline)
SKIP_EVAL="${SKIP_EVAL:-0}"

# =============================================================================
# Directory Setup
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
RUN_NAME="qwen2.5-7b-t8-bird-$(date +%Y%m%d_%H%M%S)"
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

# =============================================================================
# Dataset Setup - T8 (clean structure)
# =============================================================================

echo ""
echo ">>> Setting up T8 dataset..."

DATASET_DIR="${DATASET_DIR:-data/training/t8/training}"

TRAIN_JSONL="$DATASET_DIR/train.jsonl"
DEV_JSONL="$DATASET_DIR/dev.jsonl"

# Verify files exist
for f in "$TRAIN_JSONL" "$DEV_JSONL"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing dataset file: $f"
    echo ""
    echo "T8 structure should be:"
    echo "  data/training/t8/"
    echo "  ├── training/"
    echo "  │   ├── train.jsonl"
    echo "  │   └── dev.jsonl"
    echo "  └── eval/"
    echo "      └── bird_dev.jsonl"
    exit 1
  fi
done

echo "Using dataset: $DATASET_DIR"
echo "Dataset sizes:"
wc -l "$TRAIN_JSONL" "$DEV_JSONL"
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
echo ">>> Starting Training on T8"
echo "=============================================="
echo "Model: $MODEL_ID"
echo "Dataset: T8 (DDL format, pattern-upsampled)"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Sequence Length: $SEQ_LEN"
echo "Batch Size: $TRAIN_BS x $GRAD_ACC = $((TRAIN_BS * GRAD_ACC)) effective"
echo "LoRA: r=$LORA_R, alpha=$LORA_ALPHA (increased from t7)"
echo "Output: $OUT_DIR"
echo ""

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
  echo "Resuming from: $RESUME_FROM"
fi

python3 -u training/train_lora.py \
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
echo "Next steps:"
echo "  1. Run BIRD evaluation: bash evaluation/run_bird_eval.sh"
echo "  2. Compare with t7 baseline (44.26%)"
echo "  3. Target: 55-60% execution accuracy"
echo ""
echo "Results:"
ls -la "$OUT_DIR"/*.json 2>/dev/null || echo "No JSON results yet"

echo ""
echo "Done: $OUT_DIR"

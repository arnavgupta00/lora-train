#!/usr/bin/env bash
###############################################################################
# Script 1: LoRA SFT Training for Qwen3-1.7B on T9 v4 Dataset
# 
# This script fine-tunes Qwen3-1.7B on the T9 v4 text-to-SQL dataset using
# LoRA (Low-Rank Adaptation). Optimized for RTX 3090/4090 24GB VRAM.
#
# Usage:
#   # Basic run (3 epochs, full dataset)
#   nohup bash training/configs/qwen3-1.7b-sft-t9.sh > sft.log 2>&1 &
#   tail -f sft.log
#
#   # Fast run (2 epochs, optimized batch size)
#   EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 nohup bash training/configs/qwen3-1.7b-sft-t9.sh > sft.log 2>&1 &
#
# Environment Variables:
#   MODEL_ID     - Base model (default: Qwen/Qwen3-1.7B)
#   EPOCHS       - Training epochs (default: 3)
#   LR           - Learning rate (default: 2e-4)
#   BATCH_SIZE   - Per-device batch size (default: 4)
#   GRAD_ACC     - Gradient accumulation steps (default: 8)
#   SEQ_LEN      - Max sequence length (default: 2048, use 1024 for speed)
#   LORA_R       - LoRA rank (default: 32)
#   LORA_ALPHA   - LoRA alpha (default: 64)
#   OUT_DIR      - Output directory (auto-generated if not set)
#   RESUME_FROM  - Checkpoint path to resume from
#
# Hardware Requirements:
#   - GPU: RTX 3090/4090 (24GB) or A100
#   - RAM: 32GB recommended
#   - Disk: 20GB+ for model cache and outputs
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  Qwen3-1.7B LoRA SFT Training on T9 v4"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"

# Training hyperparameters
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"

# LoRA configuration
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# =============================================================================
# Directory Setup
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
echo "Working directory: $ROOT_DIR"

# HuggingFace cache setup
if [[ -d /runpod-volume ]]; then
    export HF_HOME="/runpod-volume/hf"
    export TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
    export HF_DATASETS_CACHE="/runpod-volume/hf/datasets"
    OUT_BASE="/runpod-volume/outputs"
    echo "Using RunPod volume for caching"
elif [[ -d /workspace ]]; then
    export HF_HOME="/workspace/hf"
    export TRANSFORMERS_CACHE="/workspace/hf/transformers"
    export HF_DATASETS_CACHE="/workspace/hf/datasets"
    OUT_BASE="/workspace/outputs"
    echo "Using /workspace for caching"
else
    export HF_HOME="$HOME/.cache/huggingface"
    OUT_BASE="$ROOT_DIR/outputs"
    echo "Using local cache"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$OUT_BASE"

# Output directory
RUN_NAME="qwen3-1.7b-sft-t9-$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$OUT_BASE/$RUN_NAME}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/sft_train.log"
echo "Output directory: $OUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# =============================================================================
# Dataset Setup
# =============================================================================

# T9 v4 dataset
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/data/training/t9/train_v4.jsonl}"
DEV_JSONL="${DEV_JSONL:-$ROOT_DIR/data/training/t9/dev_v4.jsonl}"

# Verify files exist
for f in "$TRAIN_JSONL" "$DEV_JSONL"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Dataset file not found: $f"
        echo ""
        echo "Expected T9 v4 dataset at:"
        echo "  $ROOT_DIR/data/training/t9/train_v4.jsonl"
        echo "  $ROOT_DIR/data/training/t9/dev_v4.jsonl"
        exit 1
    fi
done

echo "Dataset:"
echo "  Train: $TRAIN_JSONL"
echo "  Dev:   $DEV_JSONL"
echo ""
echo "Dataset sizes:"
wc -l "$TRAIN_JSONL" "$DEV_JSONL"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

echo ">>> Checking/installing dependencies..."

# Check if pip packages are installed
python3 -c "import transformers; import peft; import torch" 2>/dev/null || {
    echo "Installing required packages..."
    python3 -m pip install -U pip setuptools wheel
    python3 -m pip install accelerate datasets peft requests safetensors
    python3 -m pip install "transformers>=4.45.0,<5"
    python3 -m pip install hf-transfer || echo "Warning: hf-transfer install failed (optional)"
}

# =============================================================================
# GPU Detection
# =============================================================================

echo ""
echo ">>> GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)"
    echo "Total VRAM: ${VRAM_MIB} MiB"
    
    # Auto-enable 8-bit for low VRAM
    LOAD8_ARGS=()
    if [[ "${FORCE_8BIT:-0}" == "1" ]] || [[ "$VRAM_MIB" -lt 20000 ]]; then
        echo "Enabling 8-bit base load (VRAM < 20GB or FORCE_8BIT=1)"
        LOAD8_ARGS+=(--load_in_8bit)
    fi
else
    echo "WARNING: nvidia-smi not found"
    LOAD8_ARGS=()
fi
echo ""

# =============================================================================
# Training
# =============================================================================

echo "=============================================="
echo ">>> Starting SFT Training"
echo "=============================================="
echo "Model:           $MODEL_ID"
echo "Epochs:          $EPOCHS"
echo "Learning Rate:   $LR"
echo "Sequence Length: $SEQ_LEN"
echo "Batch Size:      $BATCH_SIZE x $GRAD_ACC = $((BATCH_SIZE * GRAD_ACC)) effective"
echo "LoRA:            r=$LORA_R, alpha=$LORA_ALPHA"
echo "Output:          $OUT_DIR"
echo ""

# Resume checkpoint handling
RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    if [[ -d "$RESUME_FROM" ]]; then
        RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
        echo "Resuming from: $RESUME_FROM"
    else
        echo "WARNING: Resume checkpoint not found: $RESUME_FROM"
    fi
fi

# Run training
python3 -u training/train_lora.py \
    --model_id "$MODEL_ID" \
    --train_jsonl "$TRAIN_JSONL" \
    --dev_jsonl "$DEV_JSONL" \
    --output_dir "$OUT_DIR" \
    --max_seq_len "$SEQ_LEN" \
    --pack \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
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

# =============================================================================
# Completion
# =============================================================================

echo ""
echo "=============================================="
echo ">>> SFT Training Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Output directory: $OUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Contents:"
ls -la "$OUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run GRPO training: bash training/configs/qwen3-1.7b-grpo-t9.sh"
echo "  2. Or evaluate directly: bash evaluation/run_eval_basic.sh"
echo ""

# Export output dir for chaining scripts
export SFT_OUTPUT_DIR="$OUT_DIR"
echo "Exported SFT_OUTPUT_DIR=$OUT_DIR"

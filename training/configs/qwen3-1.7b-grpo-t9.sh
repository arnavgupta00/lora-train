#!/usr/bin/env bash
###############################################################################
# Script 2: GRPO Training for Qwen3-1.7B on T9/BIRD
# 
# This script runs GRPO (Group Relative Policy Optimization) training
# on top of an SFT-trained model. GRPO uses execution-based rewards
# to further improve SQL generation quality.
#
# Prerequisites:
#   - SFT model trained (Script 1)
#   - BIRD databases extracted
#
# Usage:
#   # Basic run (uses default SFT output)
#   nohup bash training/configs/qwen3-1.7b-grpo-t9.sh > grpo.log 2>&1 &
#   tail -f grpo.log
#
#   # Specify SFT adapter
#   SFT_ADAPTER="./outputs/qwen3-1.7b-sft/" nohup bash training/configs/qwen3-1.7b-grpo-t9.sh > grpo.log 2>&1 &
#
#   # Fast mode (fewer generations, smaller dataset)
#   NUM_GEN=4 GRPO_SAMPLES=2000 nohup bash training/configs/qwen3-1.7b-grpo-t9.sh > grpo.log 2>&1 &
#
# Environment Variables:
#   MODEL_ID       - Base model (default: Qwen/Qwen3-1.7B)
#   SFT_ADAPTER    - Path to SFT LoRA adapter
#   NUM_GEN        - SQL candidates per question (default: 8, use 4 for speed)
#   GRPO_SAMPLES   - Training examples (default: all, use 2000-3000 for speed)
#   LR             - Learning rate (default: 5e-7)
#   EPOCHS         - Training epochs (default: 1)
#
# Hardware Requirements:
#   - GPU: RTX 3090/4090 (24GB) or A100
#   - Time: 4-6 hours (full), 1.5-2 hours (fast mode)
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  Qwen3-1.7B GRPO Training"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"

# GRPO hyperparameters
NUM_GEN="${NUM_GEN:-8}"           # SQL candidates per question
EPOCHS="${EPOCHS:-1}"             # GRPO epochs
LR="${LR:-5e-7}"                  # Learning rate (very small for RL)
KL_COEF="${KL_COEF:-0.1}"         # KL divergence coefficient
TEMPERATURE="${TEMPERATURE:-0.7}" # Sampling temperature
GRPO_SAMPLES="${GRPO_SAMPLES:-0}" # 0 = use all

# LoRA configuration (for GRPO adapter)
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

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
elif [[ -d /workspace ]]; then
    export HF_HOME="/workspace/hf"
    export TRANSFORMERS_CACHE="/workspace/hf/transformers"
    export HF_DATASETS_CACHE="/workspace/hf/datasets"
    OUT_BASE="/workspace/outputs"
else
    export HF_HOME="$HOME/.cache/huggingface"
    OUT_BASE="$ROOT_DIR/outputs"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$OUT_BASE"

# Output directory
RUN_NAME="qwen3-1.7b-grpo-t9-$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$OUT_BASE/$RUN_NAME}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/grpo_train.log"
echo "Output directory: $OUT_DIR"
echo "Log file: $LOG_FILE"

# =============================================================================
# Find SFT Adapter
# =============================================================================

SFT_ADAPTER="${SFT_ADAPTER:-}"

# If not specified, find most recent SFT output
if [[ -z "$SFT_ADAPTER" ]]; then
    echo ""
    echo ">>> Looking for SFT adapter..."
    
    # Check for exported SFT_OUTPUT_DIR from previous script
    if [[ -n "${SFT_OUTPUT_DIR:-}" && -d "$SFT_OUTPUT_DIR" ]]; then
        SFT_ADAPTER="$SFT_OUTPUT_DIR"
        echo "Using SFT_OUTPUT_DIR: $SFT_ADAPTER"
    else
        # Find most recent SFT output
        SFT_ADAPTER=$(ls -dt "$OUT_BASE"/qwen3-1.7b-sft-* 2>/dev/null | head -n1 || echo "")
        if [[ -n "$SFT_ADAPTER" && -d "$SFT_ADAPTER" ]]; then
            echo "Found recent SFT adapter: $SFT_ADAPTER"
        fi
    fi
fi

# Validate SFT adapter
if [[ -z "$SFT_ADAPTER" || ! -d "$SFT_ADAPTER" ]]; then
    echo "ERROR: SFT adapter not found!"
    echo ""
    echo "Please either:"
    echo "  1. Run SFT training first: bash training/configs/qwen3-1.7b-sft-t9.sh"
    echo "  2. Set SFT_ADAPTER environment variable"
    echo ""
    echo "Example:"
    echo "  SFT_ADAPTER=/path/to/sft/adapter bash training/configs/qwen3-1.7b-grpo-t9.sh"
    exit 1
fi

# Check for adapter files
if [[ ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
    echo "WARNING: adapter_config.json not found in $SFT_ADAPTER"
    echo "This may not be a valid LoRA adapter directory"
fi

echo ""

# =============================================================================
# Dataset and Database Setup
# =============================================================================

# Training data (use BIRD train for GRPO)
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/data/training/t9/train_v4.jsonl}"

# Database directory (for execution rewards)
DB_DIR="${DB_DIR:-}"

# Try to find BIRD databases
if [[ -z "$DB_DIR" ]]; then
    for candidate in \
        "/workspace/bird_eval/dev_databases" \
        "/workspace/lora-train/bird_eval/dev_20240627/dev_databases" \
        "/runpod-volume/bird_eval/dev_databases" \
        "$ROOT_DIR/bird_eval/dev_databases" \
        "$ROOT_DIR/data/bird/databases"; do
        if [[ -d "$candidate" ]]; then
            DB_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$DB_DIR" || ! -d "$DB_DIR" ]]; then
    echo "ERROR: BIRD database directory not found!"
    echo ""
    echo "Please set DB_DIR to the path containing BIRD databases"
    echo ""
    echo "Example:"
    echo "  DB_DIR=/path/to/bird/databases bash training/configs/qwen3-1.7b-grpo-t9.sh"
    exit 1
fi

echo "Dataset:   $TRAIN_JSONL"
echo "Databases: $DB_DIR"
echo ""

# =============================================================================
# GPU Detection
# =============================================================================

echo ">>> GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi
echo ""

# =============================================================================
# Training
# =============================================================================

echo "=============================================="
echo ">>> Starting GRPO Training"
echo "=============================================="
echo "Model:           $MODEL_ID"
echo "SFT Adapter:     $SFT_ADAPTER"
echo "Num Generations: $NUM_GEN"
echo "Epochs:          $EPOCHS"
echo "Learning Rate:   $LR"
echo "KL Coefficient:  $KL_COEF"
echo "Temperature:     $TEMPERATURE"
echo "Output:          $OUT_DIR"
echo ""

# Build arguments
EXTRA_ARGS=()
if [[ "$GRPO_SAMPLES" -gt 0 ]]; then
    EXTRA_ARGS+=(--max_examples "$GRPO_SAMPLES")
    echo "Using $GRPO_SAMPLES examples (fast mode)"
fi

# Run GRPO training
python3 -u training/train_grpo.py \
    --base_model_id "$MODEL_ID" \
    --sft_adapter_dir "$SFT_ADAPTER" \
    --train_jsonl "$TRAIN_JSONL" \
    --db_dir "$DB_DIR" \
    --output_dir "$OUT_DIR" \
    --num_generations "$NUM_GEN" \
    --temperature "$TEMPERATURE" \
    --learning_rate "$LR" \
    --num_epochs "$EPOCHS" \
    --kl_coef "$KL_COEF" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --save_steps 100 \
    --logging_steps 10 \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# =============================================================================
# Completion
# =============================================================================

echo ""
echo "=============================================="
echo ">>> GRPO Training Complete!"
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
echo "  1. Run evaluation: bash evaluation/run_eval_basic.sh"
echo "  2. Run self-consistency eval: bash evaluation/run_eval_self_consistency.sh"
echo ""

# Export output dir for chaining scripts
export GRPO_OUTPUT_DIR="$OUT_DIR"
echo "Exported GRPO_OUTPUT_DIR=$OUT_DIR"

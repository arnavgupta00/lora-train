#!/usr/bin/env bash
###############################################################################
# Combined Pipeline: SFT + GRPO + Evaluation for Qwen3-1.7B
#
# Sleep-friendly script that runs the complete training and evaluation pipeline.
# Start it before bed, wake up to results.
#
# QUICK START (from fresh clone):
#   git clone https://github.com/arnavgupta00/lora-train.git
#   cd lora-train
#   # Download BIRD data (see BIRD_SETUP section below)
#   EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
#     nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
#   tail -f pipeline.log
#
# Pipeline:
#   0. Auto-install Python dependencies (if missing)
#   1. Pre-flight checks (deps, GPU, datasets)
#   2. SFT Training → outputs/qwen3-1.7b-sft-<timestamp>/
#   3. GRPO Training (optional) → outputs/qwen3-1.7b-grpo-<timestamp>/
#   4. Basic Evaluation (greedy)
#   5. Self-Consistency Evaluation (optional)
#   6. Summary report
#
# Usage:
#   # Full pipeline (6-8 hours)
#   nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
#   tail -f pipeline.log
#
#   # Fast pipeline (2-3 hours) ⭐ RECOMMENDED
#   EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
#     nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
#
#   # Ultra-fast pipeline (1-1.5 hours)
#   EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 USE_LITE=1 SKIP_SC=1 \
#     nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
#
# Environment Variables:
#   # SFT Configuration
#   EPOCHS=3          # SFT epochs (use 2 for speed)
#   BATCH_SIZE=4      # Per-device batch (use 8 on 24GB)
#   SEQ_LEN=2048      # Sequence length (use 1024 for speed)
#   LR=2e-4           # Learning rate
#
#   # GRPO Configuration
#   SKIP_GRPO=0       # Set to 1 to skip GRPO training
#   NUM_GEN=8         # SQL candidates per question
#   GRPO_SAMPLES=0    # 0=all, or limit for speed
#
#   # Evaluation
#   SKIP_EVAL=0       # Skip all evaluation
#   SKIP_SC=0         # Skip self-consistency (just do greedy)
#   N_SAMPLES=10      # SC sample count
#
#   # Dataset
#   USE_LITE=0        # Use 8K lite dataset (faster, lower quality)
#
#   # BIRD Data Paths (auto-detected or set manually)
#   BIRD_DEV_JSON=/path/to/bird/dev.json
#   DB_DIR=/path/to/bird/dev_databases
#
# Hardware: RTX 3090/4090 (24GB VRAM)
# Time: 2-8 hours depending on configuration
#
# BIRD_SETUP:
#   Download BIRD benchmark from: https://bird-bench.github.io/
#   Extract to: ./bird_eval/ or /workspace/bird_eval/
#   Required files:
#     - dev.json (evaluation questions)
#     - dev_databases/ (SQLite databases)
###############################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Model
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"

# SFT Training
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# GRPO Training
SKIP_GRPO="${SKIP_GRPO:-0}"
NUM_GEN="${NUM_GEN:-8}"
GRPO_SAMPLES="${GRPO_SAMPLES:-0}"
GRPO_LR="${GRPO_LR:-5e-7}"

# Evaluation
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SC="${SKIP_SC:-0}"
N_SAMPLES="${N_SAMPLES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

# Dataset
USE_LITE="${USE_LITE:-0}"

# =============================================================================
# Setup
# =============================================================================

echo "=============================================="
echo "  Qwen3-1.7B Full Training Pipeline"
echo "=============================================="
echo ""
echo "Start Time:  $(date)"
echo "Pipeline ID: $(date +%Y%m%d_%H%M%S)"
echo ""

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
echo "Working Directory: $ROOT_DIR"

# HuggingFace cache setup
if [[ -d /runpod-volume ]]; then
    export HF_HOME="/runpod-volume/hf"
    export TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
    export HF_DATASETS_CACHE="/runpod-volume/hf/datasets"
    OUT_BASE="/runpod-volume/outputs"
    PLATFORM="runpod"
elif [[ -d /workspace ]]; then
    export HF_HOME="/workspace/hf"
    export TRANSFORMERS_CACHE="/workspace/hf/transformers"
    export HF_DATASETS_CACHE="/workspace/hf/datasets"
    OUT_BASE="/workspace/outputs"
    PLATFORM="vast"
else
    export HF_HOME="$HOME/.cache/huggingface"
    OUT_BASE="$ROOT_DIR/outputs"
    PLATFORM="local"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$OUT_BASE"
mkdir -p "$ROOT_DIR/results"

# Pipeline timestamp
PIPELINE_TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_DIR="$ROOT_DIR/results/pipeline_$PIPELINE_TS"
mkdir -p "$PIPELINE_DIR"

MAIN_LOG="$PIPELINE_DIR/pipeline.log"

# Redirect all output to main log
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "Platform:    $PLATFORM"
echo "Output Base: $OUT_BASE"
echo "Results:     $PIPELINE_DIR"
echo ""

# =============================================================================
# Configuration Summary
# =============================================================================

echo "=============================================="
echo ">>> Configuration"
echo "=============================================="
echo ""
echo "Model: $MODEL_ID"
echo ""
echo "SFT Training:"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE (grad_acc: $GRAD_ACC)"
echo "  Seq Length:   $SEQ_LEN"
echo "  Learning Rate: $LR"
echo "  LoRA:         r=$LORA_R, alpha=$LORA_ALPHA"
echo ""

if [[ "$SKIP_GRPO" == "1" ]]; then
    echo "GRPO Training: SKIPPED"
else
    echo "GRPO Training:"
    echo "  Generations:  $NUM_GEN"
    echo "  Samples:      ${GRPO_SAMPLES:-all}"
    echo "  Learning Rate: $GRPO_LR"
fi
echo ""

if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "Evaluation: SKIPPED"
else
    echo "Evaluation:"
    echo "  Basic (Greedy): YES"
    if [[ "$SKIP_SC" == "1" ]]; then
        echo "  Self-Consistency: SKIPPED"
    else
        echo "  Self-Consistency: N=$N_SAMPLES"
    fi
fi
echo ""

if [[ "$USE_LITE" == "1" ]]; then
    echo "Dataset: LITE (8K subset)"
else
    echo "Dataset: FULL (14K)"
fi
echo ""

# Estimate time
est_sft_hrs=$(echo "scale=1; $EPOCHS * 1.0" | bc)
if [[ "$BATCH_SIZE" == "8" ]]; then est_sft_hrs=$(echo "scale=1; $est_sft_hrs * 0.8" | bc); fi
if [[ "$SEQ_LEN" == "1024" ]]; then est_sft_hrs=$(echo "scale=1; $est_sft_hrs * 0.7" | bc); fi
if [[ "$USE_LITE" == "1" ]]; then est_sft_hrs=$(echo "scale=1; $est_sft_hrs * 0.6" | bc); fi

est_grpo_hrs="0"
if [[ "$SKIP_GRPO" != "1" ]]; then
    est_grpo_hrs="4.0"
    if [[ "$GRPO_SAMPLES" -gt 0 && "$GRPO_SAMPLES" -lt 3000 ]]; then
        est_grpo_hrs="1.5"
    fi
fi

est_eval_hrs="0.5"
if [[ "$SKIP_SC" != "1" ]]; then
    est_eval_hrs="1.5"
fi

total_hrs=$(echo "scale=1; $est_sft_hrs + $est_grpo_hrs + $est_eval_hrs" | bc)
echo "Estimated Time: ~${total_hrs} hours"
echo ""

# =============================================================================
# Dependency Installation
# =============================================================================

echo "=============================================="
echo ">>> Checking and Installing Dependencies"
echo "=============================================="
echo ""

INSTALL_REQUIRED=0

# Check Python packages
MISSING_PACKAGES=()

echo -n "Checking PyTorch... "
if python3 -c "import torch" 2>/dev/null; then
    version=$(python3 -c "import torch; print(torch.__version__)")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("torch")
    INSTALL_REQUIRED=1
fi

echo -n "Checking transformers... "
if python3 -c "import transformers" 2>/dev/null; then
    version=$(python3 -c "import transformers; print(transformers.__version__)")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("transformers")
    INSTALL_REQUIRED=1
fi

echo -n "Checking accelerate... "
if python3 -c "import accelerate" 2>/dev/null; then
    version=$(python3 -c "import accelerate; print(accelerate.__version__)")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("accelerate")
    INSTALL_REQUIRED=1
fi

echo -n "Checking peft... "
if python3 -c "import peft" 2>/dev/null; then
    version=$(python3 -c "import peft; print(peft.__version__)")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("peft")
    INSTALL_REQUIRED=1
fi

echo -n "Checking trl... "
if python3 -c "import trl" 2>/dev/null; then
    version=$(python3 -c "import trl; print(trl.__version__)")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("trl")
    INSTALL_REQUIRED=1
fi

echo -n "Checking datasets... "
if python3 -c "import datasets" 2>/dev/null; then
    version=$(python3 -c "import datasets; print(datasets.__version__)")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("datasets")
    INSTALL_REQUIRED=1
fi

echo -n "Checking bitsandbytes... "
if python3 -c "import bitsandbytes" 2>/dev/null; then
    version=$(python3 -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo "unknown")
    echo "OK ($version)"
else
    echo "MISSING"
    MISSING_PACKAGES+=("bitsandbytes")
    INSTALL_REQUIRED=1
fi

echo ""

# Install missing packages
if [[ $INSTALL_REQUIRED -eq 1 ]]; then
    echo ">>> Installing missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    
    # Special handling for PyTorch (often slow via pip)
    if [[ " ${MISSING_PACKAGES[*]} " =~ " torch " ]]; then
        echo "Installing PyTorch with optimized method..."
        echo ""
        
        # Detect Python version for correct wheel
        PYTHON_VERSION=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
        
        # Try pip install with timeout first (fast if CDN is good)
        echo "Attempting pip install (will timeout if too slow)..."
        timeout 60 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null
        
        if [[ $? -ne 0 ]]; then
            echo "Pip install too slow, using direct download method..."
            
            # Download wheel directly (uses full bandwidth like wget)
            TORCH_WHEEL="torch-2.5.1+cu121-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl"
            TORCH_URL="https://download.pytorch.org/whl/cu121/${TORCH_WHEEL}"
            
            echo "Downloading PyTorch wheel directly..."
            if wget -q --show-progress "$TORCH_URL" 2>&1 | grep -q "ERROR"; then
                echo "Wget failed, falling back to regular pip (this may take time)..."
                pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            else
                echo "Installing from local wheel..."
                pip3 install "$TORCH_WHEEL" -q
                rm -f "$TORCH_WHEEL"
            fi
        fi
        echo ""
    fi
    
    # Check if requirements.txt exists for other packages
    if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
        echo "Installing other dependencies from requirements.txt..."
        # Install all except torch (already installed)
        grep -v "^torch" "$ROOT_DIR/requirements.txt" | pip3 install -r /dev/stdin -q
    else
        echo "Installing other packages individually..."
        # Other packages
        for pkg in "${MISSING_PACKAGES[@]}"; do
            if [[ "$pkg" != "torch" ]]; then
                pip3 install "$pkg" -q
            fi
        done
    fi
    
    echo ""
    echo ">>> Dependencies installed successfully!"
    echo ""
else
    echo ">>> All dependencies already installed!"
    echo ""
fi

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "=============================================="
echo ">>> Pre-flight Checks"
echo "=============================================="
echo ""

PREFLIGHT_PASSED=1

# Check GPU
echo -n "GPU: "
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo "$GPU_NAME (${GPU_MEM}MB)"
    
    if [[ $GPU_MEM -lt 20000 ]]; then
        echo "  WARNING: Less than 20GB VRAM detected. May need reduced batch size."
    fi
else
    echo "WARNING - nvidia-smi not found (training requires GPU)"
fi

# Verify core packages are now available
echo -n "PyTorch: "
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    :
else
    echo "FAILED - PyTorch still not available after install"
    PREFLIGHT_PASSED=0
fi

# CRITICAL: Check CUDA availability in PyTorch
echo -n "CUDA in PyTorch: "
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    CUDA_GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo "✓ ($CUDA_GPU_NAME)"
else
    echo "✗ FAILED - GPU NOT DETECTED BY PYTORCH"
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  CRITICAL: PyTorch cannot detect your GPU!                  ║"
    echo "  ║  Training will run on CPU and be EXTREMELY SLOW (days).     ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Common causes:"
    echo "    1. PyTorch CUDA version doesn't match NVIDIA driver"
    echo "    2. CUDA not installed or not in PATH"
    echo ""
    echo "  Check your CUDA version:"
    echo "    nvidia-smi  # Look for 'CUDA Version: X.Y'"
    echo ""
    echo "  Reinstall PyTorch with correct CUDA version:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        echo "    # Your NVIDIA driver: $DRIVER_VERSION"
        # Guess CUDA version from driver
        if [[ "$DRIVER_VERSION" > "535" ]]; then
            echo "    pip uninstall torch torchvision torchaudio -y"
            echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        else
            echo "    pip uninstall torch torchvision torchaudio -y"
            echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        fi
    else
        echo "    pip uninstall torch torchvision torchaudio -y"
        echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    fi
    echo ""
    PREFLIGHT_PASSED=0
fi

echo -n "Transformers: "
if python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null; then
    :
else
    echo "FAILED - transformers still not available after install"
    PREFLIGHT_PASSED=0
fi

echo -n "PEFT: "
if python3 -c "import peft; print(peft.__version__)" 2>/dev/null; then
    :
else
    echo "FAILED - peft not installed"
    PREFLIGHT_PASSED=0
fi

# Check dataset
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/data/training/t9/train_v4.jsonl}"
DEV_JSONL="${DEV_JSONL:-$ROOT_DIR/data/training/t9/dev_v4.jsonl}"

if [[ "$USE_LITE" == "1" ]]; then
    TRAIN_JSONL="$ROOT_DIR/data/training/t9/train_lite.jsonl"
    # Dev set doesn't have a lite version, use full
fi

echo -n "Training Data: "
if [[ -f "$TRAIN_JSONL" ]]; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_JSONL")
    echo "$TRAIN_JSONL ($TRAIN_COUNT examples)"
else
    echo "FAILED - $TRAIN_JSONL not found"
    PREFLIGHT_PASSED=0
fi

echo -n "Dev Data: "
if [[ -f "$DEV_JSONL" ]]; then
    DEV_COUNT=$(wc -l < "$DEV_JSONL")
    echo "$DEV_JSONL ($DEV_COUNT examples)"
else
    echo "FAILED - $DEV_JSONL not found"
    PREFLIGHT_PASSED=0
fi

# Check BIRD data (for GRPO and eval)
DB_DIR="${DB_DIR:-}"
BIRD_DEV_JSON="${BIRD_DEV_JSON:-}"

# Find BIRD files
for candidate in \
    "/workspace/bird_eval/dev.json" \
    "/workspace/lora-train/bird_eval/dev_20240627/dev.json" \
    "/runpod-volume/bird_eval/dev.json" \
    "$ROOT_DIR/bird_eval/dev.json" \
    "$ROOT_DIR/data/bird/dev.json"; do
    if [[ -f "$candidate" ]]; then
        BIRD_DEV_JSON="$candidate"
        break
    fi
done

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

echo -n "BIRD Data: "
if [[ -f "$BIRD_DEV_JSON" && -d "$DB_DIR" ]]; then
    echo "OK"
    echo "  dev.json: $BIRD_DEV_JSON"
    echo "  databases: $DB_DIR"
else
    if [[ "$SKIP_GRPO" == "1" && "$SKIP_EVAL" == "1" ]]; then
        echo "SKIPPED (not needed for SFT-only run)"
    else
        echo "NOT FOUND"
        echo ""
        echo "  BIRD benchmark data is required for GRPO training and evaluation."
        echo ""
        echo "  Please download BIRD benchmark:"
        echo "    1. Visit: https://bird-bench.github.io/"
        echo "    2. Download the dev set"
        echo "    3. Extract to one of these locations:"
        echo "       - $ROOT_DIR/bird_eval/"
        echo "       - /workspace/bird_eval/"
        echo ""
        echo "  Or set paths manually:"
        echo "    export BIRD_DEV_JSON=/path/to/dev.json"
        echo "    export DB_DIR=/path/to/dev_databases"
        echo ""
        echo "  To skip evaluation and GRPO, run with:"
        echo "    SKIP_GRPO=1 SKIP_EVAL=1 bash training/configs/qwen3-1.7b-full-pipeline.sh"
        echo ""
        PREFLIGHT_PASSED=0
    fi
fi

echo ""

if [[ $PREFLIGHT_PASSED -eq 0 ]]; then
    echo ">>> PRE-FLIGHT CHECKS FAILED"
    echo "Please fix the issues above and try again."
    exit 1
fi

echo ">>> All pre-flight checks passed!"
echo ""

# =============================================================================
# Phase 1: SFT Training
# =============================================================================

echo "=============================================="
echo ">>> Phase 1: SFT Training"
echo "=============================================="
echo "Started: $(date)"
echo ""

SFT_OUTPUT_DIR="$OUT_BASE/qwen3-1.7b-sft-$PIPELINE_TS"
SFT_LOG="$PIPELINE_DIR/sft_training.log"

# Export for downstream scripts
export SFT_OUTPUT_DIR
export DB_DIR
export BIRD_DEV_JSON

python3 -u training/train_lora.py \
    --model_id "$MODEL_ID" \
    --train_jsonl "$TRAIN_JSONL" \
    --dev_jsonl "$DEV_JSONL" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --max_seq_len "$SEQ_LEN" \
    --learning_rate "$LR" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --save_steps 500 \
    --logging_steps 50 \
    --gradient_checkpointing 2>&1 | tee "$SFT_LOG"

SFT_EXIT_CODE=${PIPESTATUS[0]}

if [[ $SFT_EXIT_CODE -ne 0 ]]; then
    echo ""
    echo ">>> SFT TRAINING FAILED (exit code: $SFT_EXIT_CODE)"
    echo "Check log: $SFT_LOG"
    exit 1
fi

echo ""
echo ">>> SFT Training Complete!"
echo "Output: $SFT_OUTPUT_DIR"
echo "Completed: $(date)"
echo ""

# =============================================================================
# Phase 2: GRPO Training (Optional)
# =============================================================================

GRPO_OUTPUT_DIR=""

if [[ "$SKIP_GRPO" == "1" ]]; then
    echo "=============================================="
    echo ">>> Phase 2: GRPO Training - SKIPPED"
    echo "=============================================="
    echo ""
else
    echo "=============================================="
    echo ">>> Phase 2: GRPO Training"
    echo "=============================================="
    echo "Started: $(date)"
    echo ""
    
    GRPO_OUTPUT_DIR="$OUT_BASE/qwen3-1.7b-grpo-$PIPELINE_TS"
    GRPO_LOG="$PIPELINE_DIR/grpo_training.log"
    
    export GRPO_OUTPUT_DIR
    
    GRPO_ARGS=()
    if [[ "$GRPO_SAMPLES" -gt 0 ]]; then
        GRPO_ARGS+=(--max_examples "$GRPO_SAMPLES")
    fi
    
    python3 -u training/train_grpo.py \
        --base_model_id "$MODEL_ID" \
        --sft_adapter_dir "$SFT_OUTPUT_DIR" \
        --train_jsonl "$TRAIN_JSONL" \
        --db_dir "$DB_DIR" \
        --output_dir "$GRPO_OUTPUT_DIR" \
        --num_generations "$NUM_GEN" \
        --learning_rate "$GRPO_LR" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        "${GRPO_ARGS[@]}" 2>&1 | tee "$GRPO_LOG"
    
    GRPO_EXIT_CODE=${PIPESTATUS[0]}
    
    if [[ $GRPO_EXIT_CODE -ne 0 ]]; then
        echo ""
        echo ">>> GRPO TRAINING FAILED (exit code: $GRPO_EXIT_CODE)"
        echo "Continuing to evaluation with SFT model only..."
        GRPO_OUTPUT_DIR=""
    else
        echo ""
        echo ">>> GRPO Training Complete!"
        echo "Output: $GRPO_OUTPUT_DIR"
    fi
    echo "Completed: $(date)"
    echo ""
fi

# =============================================================================
# Phase 3: Evaluation
# =============================================================================

if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "=============================================="
    echo ">>> Phase 3: Evaluation - SKIPPED"
    echo "=============================================="
    echo ""
else
    echo "=============================================="
    echo ">>> Phase 3: Evaluation"
    echo "=============================================="
    echo "Started: $(date)"
    echo ""
    
    EVAL_RESULTS_DIR="$PIPELINE_DIR/eval_results"
    mkdir -p "$EVAL_RESULTS_DIR"
    
    # Determine best model to evaluate
    if [[ -n "$GRPO_OUTPUT_DIR" && -d "$GRPO_OUTPUT_DIR" ]]; then
        BEST_MODEL="$GRPO_OUTPUT_DIR"
        BEST_MODEL_NAME="GRPO"
    else
        BEST_MODEL="$SFT_OUTPUT_DIR"
        BEST_MODEL_NAME="SFT"
    fi
    
    # --- 3a: Baseline Evaluation ---
    echo ">>> 3a: Baseline Evaluation (greedy)"
    BASELINE_LOG="$PIPELINE_DIR/eval_baseline.log"
    
    python3 -u evaluation/eval_bird.py \
        --model_id "$MODEL_ID" \
        --bird_dev_json "$BIRD_DEV_JSON" \
        --db_dir "$DB_DIR" \
        --output_dir "$EVAL_RESULTS_DIR/baseline" \
        --batch_size "$EVAL_BATCH_SIZE" 2>&1 | tee "$BASELINE_LOG"
    
    echo ""
    
    # --- 3b: SFT Model Evaluation ---
    echo ">>> 3b: SFT Model Evaluation (greedy)"
    SFT_EVAL_LOG="$PIPELINE_DIR/eval_sft.log"
    
    python3 -u evaluation/eval_bird.py \
        --model_id "$MODEL_ID" \
        --adapter_dir "$SFT_OUTPUT_DIR" \
        --bird_dev_json "$BIRD_DEV_JSON" \
        --db_dir "$DB_DIR" \
        --output_dir "$EVAL_RESULTS_DIR/sft" \
        --batch_size "$EVAL_BATCH_SIZE" 2>&1 | tee "$SFT_EVAL_LOG"
    
    echo ""
    
    # --- 3c: GRPO Model Evaluation (if available) ---
    if [[ -n "$GRPO_OUTPUT_DIR" && -d "$GRPO_OUTPUT_DIR" ]]; then
        echo ">>> 3c: GRPO Model Evaluation (greedy)"
        GRPO_EVAL_LOG="$PIPELINE_DIR/eval_grpo.log"
        
        python3 -u evaluation/eval_bird.py \
            --model_id "$MODEL_ID" \
            --adapter_dir "$GRPO_OUTPUT_DIR" \
            --bird_dev_json "$BIRD_DEV_JSON" \
            --db_dir "$DB_DIR" \
            --output_dir "$EVAL_RESULTS_DIR/grpo" \
            --batch_size "$EVAL_BATCH_SIZE" 2>&1 | tee "$GRPO_EVAL_LOG"
        
        echo ""
    fi
    
    # --- 3d: Self-Consistency Evaluation ---
    if [[ "$SKIP_SC" != "1" ]]; then
        echo ">>> 3d: Self-Consistency Evaluation (N=$N_SAMPLES)"
        SC_EVAL_LOG="$PIPELINE_DIR/eval_sc.log"
        
        python3 -u evaluation/eval_self_consistency.py \
            --model_id "$MODEL_ID" \
            --adapter_dir "$BEST_MODEL" \
            --bird_dev_json "$BIRD_DEV_JSON" \
            --db_dir "$DB_DIR" \
            --output_dir "$EVAL_RESULTS_DIR/sc_${BEST_MODEL_NAME,,}" \
            --n_samples "$N_SAMPLES" \
            --temperature 0.7 \
            --top_p 0.95 2>&1 | tee "$SC_EVAL_LOG"
        
        echo ""
    fi
    
    echo ">>> Evaluation Complete!"
    echo "Results: $EVAL_RESULTS_DIR"
    echo "Completed: $(date)"
    echo ""
fi

# =============================================================================
# Summary Report
# =============================================================================

echo "=============================================="
echo ">>> Pipeline Summary"
echo "=============================================="
echo ""
echo "End Time: $(date)"
echo ""

# Create summary file
SUMMARY_FILE="$PIPELINE_DIR/SUMMARY.txt"

{
    echo "========================================="
    echo "  QWEN3-1.7B TRAINING PIPELINE SUMMARY"
    echo "========================================="
    echo ""
    echo "Pipeline ID: $PIPELINE_TS"
    echo "Platform:    $PLATFORM"
    echo "Model:       $MODEL_ID"
    echo ""
    echo "Configuration:"
    echo "  SFT Epochs:    $EPOCHS"
    echo "  Batch Size:    $BATCH_SIZE x $GRAD_ACC"
    echo "  Seq Length:    $SEQ_LEN"
    echo "  GRPO:          $(if [[ "$SKIP_GRPO" == "1" ]]; then echo "SKIPPED"; else echo "YES (gen=$NUM_GEN)"; fi)"
    echo "  Self-Consist:  $(if [[ "$SKIP_SC" == "1" ]]; then echo "SKIPPED"; else echo "YES (N=$N_SAMPLES)"; fi)"
    echo ""
    echo "Outputs:"
    echo "  SFT Model:  $SFT_OUTPUT_DIR"
    if [[ -n "$GRPO_OUTPUT_DIR" ]]; then
        echo "  GRPO Model: $GRPO_OUTPUT_DIR"
    fi
    echo "  Results:    $PIPELINE_DIR"
    echo ""
    echo "========================================="
    echo "  RESULTS"
    echo "========================================="
    echo ""
    
    if [[ "$SKIP_EVAL" != "1" ]]; then
        # Parse results
        for result_file in "$EVAL_RESULTS_DIR"/*.json; do
            if [[ -f "$result_file" ]]; then
                name=$(basename "$result_file" .json)
                accuracy=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
                echo "  $name: $accuracy"
            fi
        done
    else
        echo "  Evaluation was skipped"
    fi
    
    echo ""
    echo "========================================="
} | tee "$SUMMARY_FILE"

echo ""
echo "Full log: $MAIN_LOG"
echo "Summary:  $SUMMARY_FILE"
echo ""
echo "=============================================="
echo ">>> PIPELINE COMPLETE!"
echo "=============================================="

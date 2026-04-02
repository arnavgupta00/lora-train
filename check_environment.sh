#!/usr/bin/env bash
###############################################################################
# Environment Check Script
#
# Verifies that your environment is ready for training Qwen3-1.7B.
# Run this after cloning the repo to ensure everything is set up correctly.
#
# Usage:
#   bash check_environment.sh
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  Qwen3-1.7B Training Environment Check"
echo "=============================================="
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# =============================================================================
# System Checks
# =============================================================================

echo ">>> System Information"
echo ""

echo -n "Operating System: "
uname -s
echo ""

echo -n "Python Version: "
if command -v python3 >/dev/null 2>&1; then
    python3 --version
else
    echo "FAILED - python3 not found"
    ((CHECKS_FAILED++))
fi
echo ""

echo -n "CUDA/GPU: "
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    echo "$GPU_NAME (${GPU_MEM}MB)"
    ((CHECKS_PASSED++))
    
    if [[ $GPU_MEM -lt 20000 ]]; then
        echo "  ⚠️  WARNING: Less than 20GB VRAM. Training may require reduced batch sizes."
        ((WARNINGS++))
    fi
else
    echo "FAILED - nvidia-smi not found (GPU required for training)"
    ((CHECKS_FAILED++))
fi
echo ""

# =============================================================================
# Python Dependencies
# =============================================================================

echo ">>> Python Dependencies"
echo ""

check_package() {
    local pkg=$1
    local import_name=${2:-$1}
    
    echo -n "  $pkg: "
    if python3 -c "import $import_name" 2>/dev/null; then
        version=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "✓ ($version)"
        ((CHECKS_PASSED++))
    else
        echo "✗ MISSING"
        ((CHECKS_FAILED++))
    fi
}

check_package "torch" "torch"
check_package "transformers" "transformers"
check_package "accelerate" "accelerate"
check_package "peft" "peft"
check_package "trl" "trl"
check_package "datasets" "datasets"
check_package "bitsandbytes" "bitsandbytes"
echo ""

# =============================================================================
# Dataset Checks
# =============================================================================

echo ">>> Dataset Availability"
echo ""

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -n "  T9 Training Data: "
if [[ -f "$ROOT_DIR/data/training/t9/train_v4.jsonl" ]]; then
    count=$(wc -l < "$ROOT_DIR/data/training/t9/train_v4.jsonl" 2>/dev/null || echo "0")
    echo "✓ ($count examples)"
    ((CHECKS_PASSED++))
else
    echo "✗ NOT FOUND"
    echo "    Expected: $ROOT_DIR/data/training/t9/train_v4.jsonl"
    ((CHECKS_FAILED++))
fi

echo -n "  T9 Dev Data: "
if [[ -f "$ROOT_DIR/data/training/t9/dev_v4.jsonl" ]]; then
    count=$(wc -l < "$ROOT_DIR/data/training/t9/dev_v4.jsonl" 2>/dev/null || echo "0")
    echo "✓ ($count examples)"
    ((CHECKS_PASSED++))
else
    echo "⚠️  NOT FOUND (optional)"
    ((WARNINGS++))
fi
echo ""

# =============================================================================
# BIRD Benchmark Data
# =============================================================================

echo ">>> BIRD Benchmark Data (required for evaluation)"
echo ""

BIRD_FOUND=0

# Check common locations
for candidate in \
    "$ROOT_DIR/bird_eval/dev.json" \
    "/workspace/bird_eval/dev.json" \
    "/runpod-volume/bird_eval/dev.json"; do
    if [[ -f "$candidate" ]]; then
        echo "  dev.json: ✓ $candidate"
        BIRD_FOUND=1
        break
    fi
done

if [[ $BIRD_FOUND -eq 0 ]]; then
    echo "  dev.json: ✗ NOT FOUND"
fi

DB_FOUND=0
for candidate in \
    "$ROOT_DIR/bird_eval/dev_databases" \
    "/workspace/bird_eval/dev_databases" \
    "/runpod-volume/bird_eval/dev_databases"; do
    if [[ -d "$candidate" ]]; then
        echo "  dev_databases/: ✓ $candidate"
        DB_FOUND=1
        break
    fi
done

if [[ $DB_FOUND -eq 0 ]]; then
    echo "  dev_databases/: ✗ NOT FOUND"
fi

if [[ $BIRD_FOUND -eq 1 && $DB_FOUND -eq 1 ]]; then
    echo ""
    echo "  BIRD data is ready for evaluation! ✓"
    ((CHECKS_PASSED++))
else
    echo ""
    echo "  ⚠️  BIRD data not found (required for GRPO training and evaluation)"
    echo ""
    echo "  To download BIRD benchmark:"
    echo "    1. Visit https://bird-bench.github.io/"
    echo "    2. Download the dev set"
    echo "    3. Extract to: $ROOT_DIR/bird_eval/"
    echo "       - bird_eval/dev.json"
    echo "       - bird_eval/dev_databases/"
    echo ""
    echo "  Or skip evaluation with: SKIP_GRPO=1 SKIP_EVAL=1"
    ((WARNINGS++))
fi
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=============================================="
echo "  Summary"
echo "=============================================="
echo ""
echo "Checks passed: $CHECKS_PASSED"
echo "Checks failed: $CHECKS_FAILED"
echo "Warnings:      $WARNINGS"
echo ""

if [[ $CHECKS_FAILED -eq 0 ]]; then
    echo "✅ Environment is ready for training!"
    echo ""
    echo "Next steps:"
    echo "  1. Download BIRD data (if not done): https://bird-bench.github.io/"
    echo "  2. Run training pipeline:"
    echo "     bash training/configs/qwen3-1.7b-full-pipeline.sh"
    echo ""
    echo "See docs/TRAINING_GUIDE.md for detailed instructions."
    exit 0
else
    echo "❌ Environment setup incomplete"
    echo ""
    if [[ $CHECKS_FAILED -gt 0 ]]; then
        echo "Missing dependencies can be installed with:"
        echo "  pip3 install -r requirements.txt"
        echo ""
        echo "Or run the full pipeline script, which auto-installs:"
        echo "  bash training/configs/qwen3-1.7b-full-pipeline.sh"
    fi
    echo ""
    exit 1
fi

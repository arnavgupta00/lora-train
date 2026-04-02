#!/bin/bash
# Fix PyTorch CUDA version mismatch
# Problem: PyTorch CUDA version > GPU driver CUDA version
# Solution: Install PyTorch matching GPU driver's max supported CUDA

set -e

echo "=============================================="
echo "PyTorch CUDA Version Fixer"
echo "=============================================="
echo ""

# Check current versions
echo "Current state:"
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs echo "  GPU driver version:"

# Detect GPU driver CUDA capability
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo ""
echo "GPU driver $DRIVER_VERSION supports:"

if (( $(echo "$DRIVER_VERSION >= 550" | bc -l) )); then
    echo "  ✓ CUDA 12.4 (driver >= 550)"
    CUDA_VERSION="cu124"
    PYTORCH_URL="https://download.pytorch.org/whl/cu124"
elif (( $(echo "$DRIVER_VERSION >= 525" | bc -l) )); then
    echo "  ✓ CUDA 12.1 (driver >= 525)"
    CUDA_VERSION="cu121"
    PYTORCH_URL="https://download.pytorch.org/whl/cu121"
elif (( $(echo "$DRIVER_VERSION >= 450" | bc -l) )); then
    echo "  ✓ CUDA 11.8 (driver >= 450)"
    CUDA_VERSION="cu118"
    PYTORCH_URL="https://download.pytorch.org/whl/cu118"
else
    echo "  ✗ Driver too old (< 450)"
    echo "Please update your GPU driver first."
    exit 1
fi

echo ""
echo "Installing PyTorch for $CUDA_VERSION..."
pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url "$PYTORCH_URL"

echo ""
echo "=============================================="
echo "Verification:"
python3 << 'PYTHON'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("✓ Ready for GPU training/inference!")
else:
    print("✗ CUDA still not available - check driver installation")
    exit(1)
PYTHON

echo ""
echo "Done! Run your evaluation/training now:"
echo "  python3 evaluation/eval_bird.py ..."

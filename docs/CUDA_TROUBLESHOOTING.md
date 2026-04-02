# CUDA Compatibility Issues - Troubleshooting Guide

## Symptom: Training Stuck at 0% for Hours

If you see:
```
0%|          | 0/1317 [00:00<?, ?it/s]
UserWarning: no accelerator is found
CUDA initialization: The NVIDIA driver on your system is too old
```

**Problem:** PyTorch cannot detect your GPU → Training runs on CPU (extremely slow!)

---

## Quick Fix

**Easiest: Run the automated fixer (auto-detects your driver version):**
```bash
bash tools/fix_pytorch_cuda.sh
```

This script:
- Detects your GPU driver version
- Installs the correct PyTorch for your driver
- Verifies CUDA is working
- Ready to run!

---

**Manual fix (if you prefer):**

```bash
# 1. Kill stuck process
pkill -f train_lora.py

# 2. Check your CUDA version
nvidia-smi
# Look for "CUDA Version: X.Y" in the output

# 3. Reinstall PyTorch with matching CUDA version
pip uninstall torch torchvision torchaudio -y

# OPTION A: Ultra-fast with aria2c (16 parallel connections - FASTEST!)
apt-get update && apt-get install -y aria2
PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
aria2c -x 16 -s 16 https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl
pip install torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl --no-deps
pip install filelock typing-extensions sympy networkx jinja2 fsspec
rm torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl

# OPTION B: Download with wget (if aria2c not available)
PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
wget https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl
pip install torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl --no-deps
pip install filelock typing-extensions sympy networkx jinja2 fsspec
rm torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl

# OPTION C: Install only torch (smaller, skips torchvision/torchaudio)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# OPTION D: Regular pip (may be slow ~100 KB/s from some regions)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OPTION E: Use mirror (WARNING: usually installs wrong CUDA version!)
# Only use if you verify CUDA version matches afterward
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. Verify GPU is detected
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Should show:
#   CUDA available: True
#   GPU: NVIDIA GeForce RTX 3090 (or your GPU model)

# 5. Restart training
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

---

## Understanding CUDA Versions

### ✅ CUDA Backward Compatibility

**KEY CONCEPT:** CUDA is backward compatible - newer drivers can run older CUDA applications!

```
Example: Your driver supports CUDA 12.8
✅ Can run: PyTorch CUDA 12.1 (older → works!)
✅ Can run: PyTorch CUDA 12.0 (older → works!)
✅ Can run: PyTorch CUDA 11.8 (older → works!)
❌ Cannot run: PyTorch CUDA 13.0 (newer → fails!)

Rule: PyTorch CUDA version MUST be ≤ Driver CUDA version
```

### Driver to PyTorch CUDA Mapping

| NVIDIA Driver Version | Max CUDA Support | Recommended PyTorch |
|----------------------|------------------|---------------------|
| 535.x or newer | CUDA 12.1+ | `torch==2.5.1+cu121` |
| 520.x - 534.x | CUDA 11.8 | `torch==2.5.1+cu118` |
| Older | CUDA 11.7 or 11.6 | See PyTorch website |

**Check your driver CUDA version:**
```bash
nvidia-smi | grep "CUDA Version"
# Example output: CUDA Version: 12.8
```

**Then install PyTorch with EQUAL OR LOWER CUDA version:**
```bash
# Driver supports CUDA 12.8 → Install PyTorch CUDA 12.1 ✅
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Driver supports CUDA 11.8 → Install PyTorch CUDA 11.8 ✅
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

---

## Slow PyTorch Download Issue

**Symptom:** `pip install torch` or `wget` downloading at ~100-500 KB/s despite fast internet (e.g., 1 Gbps connection)

**Cause:** PyTorch's CDN (`download.pytorch.org`) can be extremely slow from certain geographic regions, especially Asia

**FASTEST Solution - aria2c with Parallel Connections:**
```bash
# Install aria2c (multi-connection download tool)
apt-get update && apt-get install -y aria2

# Download with 16 parallel connections (5-10x faster!)
PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
aria2c -x 16 -s 16 https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl

# Install from local file with --no-deps (prevents slow dependency downloads)
pip install torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl --no-deps

# Install small Python dependencies (filelock, sympy, etc.)
pip install filelock typing-extensions sympy networkx jinja2 fsspec

# Clean up
rm torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl
```

**Alternative - Smaller Download (torch only):**
```bash
# Skip torchvision/torchaudio (not needed for training) - only 780 MB instead of 2+ GB
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Alternative - wget (if aria2c unavailable):**
```bash
PYTHON_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
wget https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl
pip install torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl --no-deps
pip install filelock typing-extensions sympy networkx jinja2 fsspec
rm torch-2.5.1+cu121-${PYTHON_VER}-${PYTHON_VER}-linux_x86_64.whl
```

**⚠️ WARNING - PyPI Mirrors Install Wrong CUDA Version:**
```bash
# ❌ DON'T USE THESE - They install CUDA 13.0 (too new for most drivers!)
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple  # ❌ CUDA 13.0
pip install torch -i https://mirrors.aliyun.com/pypi/simple/   # ❌ CUDA 13.0

# Problem: CUDA 13.0 > Your driver (12.8) → GPU won't work!
# Always use PyTorch's official repository with specific CUDA version.
```

**Why mirrors fail:**
1. Mirrors only have the latest PyTorch (CUDA 13.0)
2. Your driver supports CUDA 12.8 (or lower)
3. CUDA 13.0 > 12.8 → Backward compatibility breaks
4. Result: `CUDA available: False` even with working GPU

**Correct approach:**
```bash
# Always specify CUDA version explicitly from PyTorch repo
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# This ensures CUDA 12.1 which works with drivers 12.1+
```

**Speed Comparison:**
- Default pip: ~100-200 KB/s (1-2 hours for 780 MB)
- wget: ~500 KB/s - 5 MB/s (3-25 minutes)
- aria2c (16 connections): ~20-50 MB/s (15-40 seconds) ✅ **FASTEST**
- torch only (no vision/audio): ~50% smaller download ✅ **RECOMMENDED**

**Note:** The training pipeline script automatically tries aria2c → wget → pip in order of speed.

---

## Common Error Messages

### Error 1: "CUDA initialization: The NVIDIA driver on your system is too old"

**Cause:** PyTorch was compiled for CUDA 13.0 but your driver supports CUDA 12.x

**Fix:** Reinstall PyTorch with CUDA 12.1
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Error 2: "no accelerator is found"

**Cause:** PyTorch cannot access the GPU

**Check:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If `False`:
1. Wrong CUDA version (see fix above)
2. CUDA not installed
3. Wrong GPU drivers

### Error 3: "UserWarning: pin_memory is set as true but no accelerator is found"

**Cause:** Same as above - GPU not detected

**Fix:** Reinstall PyTorch with correct CUDA version

---

## Platform-Specific Notes

### RunPod
```bash
# Check what CUDA is available
ls /usr/local/cuda*

# Usually CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Vast.ai
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Google Colab
```bash
# Colab usually has CUDA 12.x pre-installed
# If issues:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Verification Checklist

After fixing, verify these all pass:

```bash
# 1. CUDA version
nvidia-smi

# 2. PyTorch can import
python3 -c "import torch; print(torch.__version__)"

# 3. GPU detected by PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# 4. Can allocate GPU memory
python3 -c "import torch; x = torch.zeros(100, 100).cuda(); print('GPU memory allocated!')"

# 5. GPU name matches
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

All should complete without errors.

---

## Still Not Working?

### Check GPU visibility
```bash
echo $CUDA_VISIBLE_DEVICES
# Should be empty or "0" (not "-1")
```

### Check CUDA installation
```bash
nvcc --version
# Should show CUDA version
```

### Check LD_LIBRARY_PATH
```bash
echo $LD_LIBRARY_PATH
# Should include /usr/local/cuda/lib64
```

### Reinstall CUDA toolkit (last resort)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# Then reinstall PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## How to Prevent This

When setting up a new environment:

```bash
# 1. Check CUDA first
nvidia-smi

# 2. Install PyTorch with matching CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cu118

# 3. Verify immediately
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not detected!'"

# 4. Then install other packages
pip install transformers accelerate peft trl
```

---

## Performance After Fix

**Before (CPU):**
- 0 it/s (stuck)
- nvidia-smi shows 0% GPU usage
- Training would take days/weeks

**After (GPU):**
- 2-5 it/s
- nvidia-smi shows 90-100% GPU usage
- Training completes in 2-3 hours

If you still see 0% GPU usage after the fix, something else is wrong - check the verification checklist above.

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

```bash
# 1. Kill stuck process
pkill -f train_lora.py

# 2. Check your CUDA version
nvidia-smi
# Look for "CUDA Version: X.Y" in the output

# 3. Reinstall PyTorch with matching CUDA version
pip uninstall torch torchvision torchaudio -y

# For CUDA 12.1 (most common on cloud GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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

| NVIDIA Driver Version | Compatible CUDA Toolkit | PyTorch Index URL |
|----------------------|------------------------|-------------------|
| 535.x or newer | CUDA 12.1 | `--index-url https://download.pytorch.org/whl/cu121` |
| 520.x - 534.x | CUDA 11.8 | `--index-url https://download.pytorch.org/whl/cu118` |
| Older | CUDA 11.7 or 11.6 | See PyTorch website |

**Check your driver:**
```bash
nvidia-smi | grep "CUDA Version"
```

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

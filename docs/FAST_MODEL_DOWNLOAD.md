# Fast Model Download Guide

When HuggingFace downloads are slow (100-500 KB/s), use these methods to download 10-50x faster.

## Problem

HuggingFace throttles single-connection downloads, resulting in:
- **Slow speed**: 100-500 KB/s
- **Long wait**: 4+ hours for a 3GB model
- **Timeouts**: Downloads may fail on unstable connections

## Solution 1: aria2c Parallel Download (RECOMMENDED)

aria2c opens multiple parallel connections, bypassing per-connection throttling.

```bash
# Install aria2c
apt-get update && apt-get install -y aria2

# Create model directory
mkdir -p models/qwen2.5-1.5b-instruct

# Download model weights with 16 parallel connections (20-50 MB/s!)
aria2c -x 16 -s 16 -k 1M \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors" \
  -d models/qwen2.5-1.5b-instruct \
  -o model.safetensors

# Download config files (small, instant)
for f in config.json tokenizer.json tokenizer_config.json generation_config.json vocab.json merges.txt; do
  wget -q "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/$f" \
    -O models/qwen2.5-1.5b-instruct/$f 2>/dev/null || true
done

# Use local model for evaluation
python3 evaluation/eval_bird.py \
  --model_id ./models/qwen2.5-1.5b-instruct \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_results
```

**Expected speed**: 20-50 MB/s (1-3 minutes for 3GB model)

## Solution 2: ModelScope Mirror (Best for Asia)

ModelScope hosts the same models with faster servers in Asia.

```bash
# Install modelscope
pip install modelscope -q

# Download from ModelScope
python3 -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')
print(f'Downloaded to: {model_dir}')
"

# Use downloaded model
python3 evaluation/eval_bird.py \
  --model_id ./models/qwen/Qwen2.5-1.5B-Instruct \
  ...
```

## Solution 3: huggingface-cli with Resume

If connection is unstable, use HF CLI which supports resume:

```bash
# Download with resume support
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ./models/qwen2.5-1.5b-instruct \
  --local-dir-use-symlinks False

# Use local model
python3 evaluation/eval_bird.py \
  --model_id ./models/qwen2.5-1.5b-instruct \
  ...
```

## Speed Comparison

| Method | Speed | Time for 3GB |
|--------|-------|--------------|
| HuggingFace (single) | 200 KB/s | ~4 hours |
| aria2c (16 connections) | 20-50 MB/s | 1-3 minutes |
| ModelScope (Asia) | 10-30 MB/s | 2-5 minutes |
| HF CLI | 200 KB/s | ~4 hours (but resumable) |

## Why aria2c Works

HuggingFace rate-limits each TCP connection to ~200 KB/s. By opening 16 parallel connections:

```
16 connections × 200 KB/s = 3.2 MB/s minimum
                          = 20-50 MB/s typical
```

The `-x 16 -s 16` flags tell aria2c to:
- `-x 16`: Use up to 16 connections per server
- `-s 16`: Split the file into 16 segments
- `-k 1M`: Minimum segment size of 1MB

## Download Different Models

### Qwen2.5-1.5B-Instruct (3.1GB)
```bash
aria2c -x 16 -s 16 -k 1M \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors" \
  -d models/qwen2.5-1.5b-instruct -o model.safetensors
```

### Qwen3-1.7B (3.4GB)
```bash
aria2c -x 16 -s 16 -k 1M \
  "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model.safetensors" \
  -d models/qwen3-1.7b -o model.safetensors
```

### Qwen2.5-7B-Instruct (15GB, sharded)
```bash
# Download all shards
for i in 1 2 3 4; do
  aria2c -x 16 -s 16 -k 1M \
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-0000${i}-of-00004.safetensors" \
    -d models/qwen2.5-7b-instruct -o model-0000${i}-of-00004.safetensors
done
```

## Troubleshooting

### "aria2c: command not found"
```bash
apt-get update && apt-get install -y aria2
```

### Download still slow with aria2c
Try a different mirror or VPN:
```bash
# Use HuggingFace mirror (if available)
aria2c -x 16 -s 16 \
  "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors" \
  -d models/qwen2.5-1.5b-instruct -o model.safetensors
```

### Config files missing after download
```bash
# Download all config files
cd models/qwen2.5-1.5b-instruct
for f in config.json tokenizer.json tokenizer_config.json generation_config.json vocab.json merges.txt special_tokens_map.json; do
  wget -q "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/$f" -O $f 2>/dev/null || true
done
```

### Model loading fails with "missing key"
Ensure you have all required files:
```bash
ls -la models/qwen2.5-1.5b-instruct/
# Should contain:
# - model.safetensors (3.1GB)
# - config.json
# - tokenizer.json
# - tokenizer_config.json
# - generation_config.json
```

## See Also

- [CUDA Troubleshooting](CUDA_TROUBLESHOOTING.md) - GPU detection issues
- [Evaluation Progress Monitoring](EVAL_PROGRESS_MONITORING.md) - Track eval progress
- [Training Guide](TRAINING_GUIDE.md) - Complete training documentation

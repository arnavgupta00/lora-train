# RunPod Training Commands

## Quick Start

### 1. Clone the repo on RunPod
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/lm.git
cd lm
```

### 2. Set your API key (for evaluation)
```bash
export NL2SQL_ADMIN_API_KEY="your-api-key-here"
```

### 3. Run training (background mode)
```bash
nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &
```

### 4. Monitor progress
```bash
tail -f run.log
# or check GPU usage
watch -n 5 nvidia-smi
```

### 5. Kill running process (if needed)
```bash
# Find the process ID
ps aux | grep run_qwen7b_t7_bird.sh

# Kill by PID (replace 12345 with actual PID)
kill 12345

# Or kill all training processes
pkill -f train_lora.py

# Force kill if needed
pkill -9 -f train_lora.py
```

---

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 1 | Number of training epochs (1 epoch takes ~2-5 hours) |
| `LR` | 5e-5 | Learning rate |
| `SEQ_LEN` | 1024 | Max sequence length (1024 is 2x faster than 2048) |
| `TRAIN_BS` | 2 | Per-device batch size (2 for 24GB GPUs to avoid OOM) |
| `GRAD_ACC` | 16 | Gradient accumulation (effective BS = TRAIN_BS × GRAD_ACC = 32) |
| `EVAL_BASE` | 1 | Also evaluate base model |
| `SKIP_EVAL` | 0 | Skip evaluation phase |
| `EVAL_LIMIT` | - | Limit eval examples |

### Example with custom settings:
```bash
# Default settings work for all GPUs (24GB+)
nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &

# Longer sequences (slower but handles very long queries)
SEQ_LEN=2048 nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &

# More epochs (diminishing returns after 1-2 epochs)
EPOCHS=2 nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &
```

---

## RunPod Configuration

### Recommended GPU Options (in order of preference)

1. **RTX A6000 (48GB)** - Best value for 7B
   - $0.79/hr, plenty of VRAM
   
2. **A100 40GB** - Faster but more expensive
   - $1.64/hr, enterprise grade
   
3. **RTX 4090 (24GB)** - Budget option
   - $0.44/hr, might need 8-bit loading

### Container Template
```
runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04
```

### Storage
- Network volume: 50GB minimum (for model cache)
- Mount at: `/runpod-volume`

---

## Training Time Estimates

| GPU | Est. Time (1 epoch) | Cost | Notes |
|-----|---------------------|------|-------|
| **RTX 4090** | **2-3 hours** | **~$0.88-1.32** | ⭐ Fastest option |
| A100 40GB | 2-3 hours | ~$3.28-4.92 | Enterprise, expensive |
| **RTX 3090 Ti** | **5-6 hours** | **~$1.35-1.62** | ⭐ Best value |
| RTX A5000 | 5-6 hours | ~$1.40-1.68 | Similar to 3090 Ti |
| ⚠️ A40 | ⚠️ 6-8 hours | ⚠️ ~$2.40-3.20 | Avoid - slow & expensive |

**Performance tip**: Gaming GPUs (3090 Ti, 4090) are faster than workstation GPUs (A40) for training due to higher clock speeds.

Note: Times are for SEQ_LEN=1024 (default). SEQ_LEN=2048 is 2-3x slower.

---

## After Training

### Check results
```bash
cat outputs/qwen2.5-7b-t7-bird-*/eval_report.lora.json
cat outputs/qwen2.5-7b-t7-bird-*/eval_report.base.json
```

### Download results
From RunPod web UI, download:
- `outputs/qwen2.5-7b-t7-bird-*/` - Full output
- `run.log` - Training log

---

## What's in t7 Dataset

| Source | Count |
|--------|-------|
| Original t3 | 8,938 |
| BIRD train | 9,311 |
| Complex patterns | 306 |
| **Total unique** | **16,699** |

SQL complexity:
- JOINs: 46%
- Aggregations: 37%
- CASE: 5%
- Subqueries: 5%
- CTEs: 0.5%
- Window functions: 0.5%

---

## Target Benchmark

BIRD Benchmark scores to beat:

| Model | Score |
|-------|-------|
| Claude Opus 4.6 | 70.15% |
| GPT-4 | 54.89% |
| ChatGPT | 39.30% |
| Qwen2.5-7B (fine-tuned) | **71.72%** (target) |

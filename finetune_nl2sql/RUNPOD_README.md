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

---

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 3 | Number of training epochs |
| `LR` | 5e-5 | Learning rate |
| `SEQ_LEN` | 2048 | Max sequence length |
| `TRAIN_BS` | 8 | Per-device batch size (A40 48GB: 8, RTX 3090 Ti 24GB: 4) |
| `GRAD_ACC` | 4 | Gradient accumulation steps (effective BS = TRAIN_BS × GRAD_ACC = 32) |
| `EVAL_BASE` | 1 | Also evaluate base model |
| `SKIP_EVAL` | 0 | Skip evaluation phase |
| `EVAL_LIMIT` | - | Limit eval examples |

### Example with custom settings:
```bash
# For RTX 3090 Ti (24GB) - use smaller batch size
TRAIN_BS=4 EPOCHS=5 LR=3e-5 nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &

# For A40 (48GB) - default settings work (TRAIN_BS=8)
EPOCHS=5 LR=3e-5 nohup bash finetune_nl2sql/run_qwen7b_t7_bird.sh > run.log 2>&1 &
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

| GPU | Est. Time | Cost | Batch Size |
|-----|-----------|------|------------|
| A100 40GB | 2-3 hours | ~$4-5 | 8 (32 effective) |
| A40 48GB | 3-4 hours | ~$1.2-1.6 | 8 (32 effective) |
| RTX 3090 Ti | 5-6 hours | ~$1.35-1.62 | 4 (16 effective) |
| RTX 4090 | 5-6 hours | ~$2.2-2.64 | 4 (16 effective) |

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

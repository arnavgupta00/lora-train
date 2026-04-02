# Qwen3-1.7B Training & Evaluation Guide

Complete guide for training Qwen3-1.7B on the T9 dataset for BIRD text-to-SQL benchmark.

## Quick Start

### Fastest Path (2-3 hours, ~55-60% accuracy)

```bash
# Clone and setup
git clone <repo-url> && cd lm
pip install -r requirements.txt

# Run fast pipeline
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

### Full Path (6-8 hours, ~60-67% accuracy)

```bash
# Run complete pipeline (SFT + GRPO + Eval)
nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

---

## Prerequisites

### Hardware
- **GPU**: RTX 3090/4090 (24GB VRAM) or A100
- **RAM**: 32GB+ recommended
- **Disk**: 50GB+ free space

### Software
```bash
# Required packages
pip install torch transformers accelerate peft trl bitsandbytes
pip install datasets wandb tqdm

# Optional (faster loading)
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### Data
- **T9 Dataset**: `data/training/t9/train_v4.jsonl` (14,034 examples)
- **BIRD Benchmark**: Download from [BIRD website](https://bird-bench.github.io/)

---

## Scripts Overview

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `qwen3-1.7b-sft-t9.sh` | SFT training only | 2-3 hrs | LoRA adapter |
| `qwen3-1.7b-grpo-t9.sh` | GRPO training (after SFT) | 4-6 hrs | LoRA adapter |
| `qwen3-1.7b-full-pipeline.sh` | Combined SFT+GRPO+Eval | 6-8 hrs | Full results |
| `run_eval_basic.sh` | Basic evaluation (greedy) | 30 min | Accuracy |
| `run_eval_self_consistency.sh` | Self-consistency eval (N=10) | 1-2 hrs | Accuracy |

---

## Detailed Commands

### Script 1: SFT Training

```bash
# Basic run
nohup bash training/configs/qwen3-1.7b-sft-t9.sh > sft.log 2>&1 &
tail -f sft.log

# Fast mode (2 epochs, larger batch)
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 \
  nohup bash training/configs/qwen3-1.7b-sft-t9.sh > sft.log 2>&1 &

# Check progress
tail -100 sft.log | grep -E "loss|epoch|step"
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 4 | Per-device batch size |
| `GRAD_ACC` | 8 | Gradient accumulation |
| `SEQ_LEN` | 2048 | Max sequence length |
| `LR` | 2e-4 | Learning rate |
| `LORA_R` | 32 | LoRA rank |
| `LORA_ALPHA` | 64 | LoRA alpha |

### Script 2: GRPO Training

```bash
# After SFT completes
nohup bash training/configs/qwen3-1.7b-grpo-t9.sh > grpo.log 2>&1 &
tail -f grpo.log

# Fast mode (fewer generations)
NUM_GEN=4 GRPO_SAMPLES=2000 \
  nohup bash training/configs/qwen3-1.7b-grpo-t9.sh > grpo.log 2>&1 &

# Specify SFT adapter explicitly
SFT_ADAPTER=./outputs/qwen3-1.7b-sft-20240101_120000 \
  bash training/configs/qwen3-1.7b-grpo-t9.sh
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SFT_ADAPTER` | auto | Path to SFT adapter |
| `NUM_GEN` | 8 | SQL candidates per question |
| `GRPO_SAMPLES` | all | Limit training examples |
| `LR` | 5e-7 | Learning rate (very small) |
| `KL_COEF` | 0.1 | KL divergence coefficient |

### Script 3a: Basic Evaluation

```bash
# Evaluate all available models
nohup bash evaluation/run_eval_basic.sh > eval.log 2>&1 &
tail -f eval.log

# Evaluate specific model
MODEL_PATH=./outputs/qwen3-1.7b-sft-20240101_120000 \
  bash evaluation/run_eval_basic.sh

# Baseline only
EVAL_BASELINE_ONLY=1 bash evaluation/run_eval_basic.sh
```

### Script 3b: Self-Consistency Evaluation

```bash
# Default (N=10 samples)
nohup bash evaluation/run_eval_self_consistency.sh > eval_sc.log 2>&1 &
tail -f eval_sc.log

# Faster with fewer samples
N_SAMPLES=5 bash evaluation/run_eval_self_consistency.sh

# Specific model
MODEL_PATH=./outputs/qwen3-1.7b-grpo-20240101_180000 N_SAMPLES=10 \
  bash evaluation/run_eval_self_consistency.sh
```

### Combined Pipeline (Sleep-Friendly)

```bash
# Full pipeline - start before sleep, wake up to results
nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log

# Fast pipeline (skip GRPO)
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &

# Ultra-fast (skip GRPO and SC eval)
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 SKIP_SC=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &

# Maximum accuracy (more GRPO)
EPOCHS=3 NUM_GEN=8 N_SAMPLES=10 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

**Pipeline Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_GRPO` | 0 | Skip GRPO training |
| `SKIP_EVAL` | 0 | Skip all evaluation |
| `SKIP_SC` | 0 | Skip self-consistency eval |
| `USE_LITE` | 0 | Use 8K lite dataset |

---

## Speed Optimization Paths

### Path A: Full Pipeline (6-8 hours)
**Expected accuracy: 60-67%**
```bash
nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

### Path B: Fast Pipeline (2-3 hours) ⭐ RECOMMENDED
**Expected accuracy: 55-60%**
```bash
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

### Path C: Ultra-Fast (1-1.5 hours)
**Expected accuracy: 50-55%**
```bash
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 USE_LITE=1 SKIP_SC=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

### Path D: Maximum Accuracy (8-10 hours)
**Expected accuracy: 62-67%**
```bash
EPOCHS=3 NUM_GEN=8 GRPO_SAMPLES=3000 N_SAMPLES=15 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

---

## Monitoring & Troubleshooting

### Check Progress
```bash
# Live log
tail -f pipeline.log

# Training metrics
tail -200 pipeline.log | grep -E "loss|accuracy|step|epoch"

# GPU utilization
watch -n 1 nvidia-smi

# Check if still running
ps aux | grep python
```

### Common Issues

**Out of Memory (OOM)**
```bash
# Reduce batch size
BATCH_SIZE=2 GRAD_ACC=16 bash training/configs/qwen3-1.7b-sft-t9.sh

# Reduce sequence length
SEQ_LEN=1024 bash training/configs/qwen3-1.7b-sft-t9.sh
```

**BIRD Data Not Found**
```bash
# Set paths explicitly
export BIRD_DEV_JSON=/path/to/bird/dev.json
export DB_DIR=/path/to/bird/dev_databases
```

**Training Stuck**
```bash
# Check CUDA errors
nvidia-smi
tail -50 pipeline.log | grep -i "error\|cuda\|memory"

# Kill and restart
pkill -f train_lora.py
```

---

## Output Structure

After running the full pipeline:

```
outputs/
├── qwen3-1.7b-sft-20240101_120000/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.json
└── qwen3-1.7b-grpo-20240101_180000/
    ├── adapter_config.json
    └── adapter_model.safetensors

results/
└── pipeline_20240101_120000/
    ├── pipeline.log
    ├── sft_training.log
    ├── grpo_training.log
    ├── eval_results/
    │   ├── baseline_results.json
    │   ├── sft_results.json
    │   ├── grpo_results.json
    │   └── grpo_sc_results.json
    └── SUMMARY.txt
```

---

## Expected Results

| Configuration | Method | Expected Accuracy |
|---------------|--------|-------------------|
| Baseline | Greedy | ~45-48% |
| SFT (2 epochs) | Greedy | ~52-55% |
| SFT (3 epochs) | Greedy | ~55-58% |
| SFT + GRPO | Greedy | ~58-62% |
| SFT + SC (N=10) | Voting | ~58-62% |
| SFT + GRPO + SC | Voting | ~62-67% |

---

## Cloud Cost Estimates

### RunPod (RTX 3090)
| Path | Time | Cost |
|------|------|------|
| Fast (B) | 2.5 hrs | ~$1.10 |
| Full (A) | 7 hrs | ~$3.10 |
| Maximum (D) | 10 hrs | ~$4.40 |

### Vast.ai (RTX 3090)
| Path | Time | Cost |
|------|------|------|
| Fast (B) | 2.5 hrs | ~$0.75 |
| Full (A) | 7 hrs | ~$2.10 |

---

## Next Steps

After training completes:

1. **Check Results**: `cat results/pipeline_*/SUMMARY.txt`
2. **Best Model**: Use GRPO model if available, else SFT model
3. **Deploy**: Load adapter with `PeftModel.from_pretrained()`
4. **Error Corrector**: To be added in future iteration

---

## Files Reference

| File | Description |
|------|-------------|
| `training/train_lora.py` | Core SFT training script |
| `training/train_grpo.py` | GRPO training with execution rewards |
| `training/configs/qwen3-1.7b-sft-t9.sh` | SFT wrapper script |
| `training/configs/qwen3-1.7b-grpo-t9.sh` | GRPO wrapper script |
| `training/configs/qwen3-1.7b-full-pipeline.sh` | Combined pipeline |
| `evaluation/eval_bird.py` | Basic BIRD evaluation |
| `evaluation/eval_self_consistency.py` | Self-consistency evaluation |
| `evaluation/run_eval_basic.sh` | Basic eval wrapper |
| `evaluation/run_eval_self_consistency.sh` | SC eval wrapper |
| `data/training/t9/train_v4.jsonl` | T9 v4 training dataset |
| `data/training/t9/dev_v4.jsonl` | T9 v4 dev dataset |

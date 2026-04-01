# lora-train

Fine-tuning small models (7B-14B) to beat mainstream models on NL2SQL benchmarks.

## 🎯 Goal
Beat GPT-4 (54.89%) and Claude Opus (70.15%) on BIRD benchmark using a 7B parameter model with LoRA.

## 📁 Repository Structure

```
lm/
├── data/                    # All datasets
│   ├── raw/                 # BIRD/Spider original data
│   ├── processed/           # ChatML formatted
│   └── training/            # Training sets (t2, t7, etc.)
│
├── training/                # Training code
│   ├── train_lora.py        # Main training script
│   ├── eval_exec.py         # Execution-match evaluation
│   └── configs/             # Per-model training configs
│
├── evaluation/              # BIRD benchmark evaluation
│   ├── run_bird_eval.sh     # Current (v3) - DDL schema
│   └── versions/            # History showing learning journey
│
├── results/                 # Training outputs
│   ├── qwen2.5-7b/          # 7B model runs
│   ├── qwen2.5-14b/         # 14B model runs
│   └── reports/             # Analysis documents
│
├── tools/                   # Data generation scripts
└── experiments/             # Archived old experiments
```

## 📊 Results Timeline

| Date | Dataset | Model | BIRD Score | Notes |
|------|---------|-------|------------|-------|
| 2026-03-28 | t2 | 14B | 24.4%* | Custom schemas only |
| 2026-03-31 | t7 | 7B | **TBD** | BIRD + custom (running) |

*Custom eval, not official BIRD

See `CHANGELOG.md` for the full journey.

## 🚀 Quick Start

### Training (RunPod)
```bash
cd /workspace/lora-train
bash training/configs/qwen2.5-7b.sh
```

### BIRD Evaluation
```bash
bash evaluation/run_bird_eval.sh
```

## 📚 Documentation

- `training/docs/README.md` - Training guide
- `training/docs/RUNPOD_README.md` - RunPod setup
- `evaluation/versions/` - History of eval attempts (learning journey)
- `CHANGELOG.md` - Full project history

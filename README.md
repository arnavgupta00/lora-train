# NL2SQL Fine-tuning Project

Fine-tuning small language models (1.7B-7B) for text-to-SQL on the BIRD benchmark.

## 🎯 Latest: Qwen3-1.7B Training Pipeline (T9 Dataset)

**New in v5:** Complete training pipeline for Qwen3-1.7B with LoRA SFT + GRPO + Self-Consistency evaluation on T9 dataset (14K examples).

### Quick Start (Cloud GPU)

```bash
# Clone the repo
git clone https://github.com/arnavgupta00/lora-train.git
cd lora-train

# ⚠️ IMPORTANT: Install PyTorch with CUDA support first!
# Check your CUDA version
nvidia-smi  # Look for "CUDA Version: X.Y"

# Install PyTorch matching your CUDA version
# For CUDA 12.1 (most cloud GPUs):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# Download BIRD benchmark data (required for evaluation)
# Visit https://bird-bench.github.io/ and extract to ./bird_eval/

# Run fast training pipeline (2-3 hours on RTX 3090)
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log
```

**⚠️ Troubleshooting:** If training is stuck at 0% for >5 minutes, see [`docs/CUDA_TROUBLESHOOTING.md`](docs/CUDA_TROUBLESHOOTING.md)

**⚠️ Slow Downloads?** If model downloads are <500 KB/s, use aria2c for 10-50x faster downloads. See [`docs/FAST_MODEL_DOWNLOAD.md`](docs/FAST_MODEL_DOWNLOAD.md)

**Expected Results:**
- **Fast mode (2-3 hrs):** ~55-60% BIRD accuracy
- **Full mode (6-8 hrs):** ~60-67% BIRD accuracy with GRPO + Self-Consistency

See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for detailed commands and options.

### Monitoring Evaluation Progress

After training completes, evaluate the model:

```bash
# Basic BIRD evaluation (no LoRA)
python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_results_base

# Expected: ~8 minutes (2-5 min download + 2-3 min generation + 30s evaluation)
# Expected accuracy: ~35-42%
```

**⚠️ Progress Tracking:** See [`docs/QUICK_EVAL_GUIDE.md`](docs/QUICK_EVAL_GUIDE.md) for how to monitor download speed, generation progress, and evaluation metrics in real-time.

---

## 🏆 Previous Best: Qwen2.5-7B (v1-3)

**Qwen2.5-7B + LoRA: 44.26%** on BIRD benchmark
- Gap to GPT-4: -10.63% (closing in!)
- 37.61% improvement from initial attempts (6.65% → 44.26%)
- Training time: 50 minutes on RTX 5090
- Dataset: t7 (16,699 BIRD + custom examples)

See [`results/qwen2.5-7b/v1-3/RUN_LOG.md`](results/qwen2.5-7b/v1-3/RUN_LOG.md) for full analysis.

---

## 📁 Repository Structure

```
lora-train/
├── data/                           # All datasets
│   ├── raw/                        # BIRD/Spider original data
│   ├── processed/                  # ChatML formatted data
│   └── training/                   # Training sets
│       ├── t2/                     # 562 examples (custom schemas)
│       ├── t7/                     # 16,699 examples (BIRD + custom) - Qwen2.5-7B
│       └── t9/                     # 14,034 examples (T9 v4) - Qwen3-1.7B ✅ NEW
│           ├── train_v4.jsonl      # Training data (14K examples)
│           └── dev_v4.jsonl        # Dev data (665 examples)
│
├── training/                       # Training code
│   ├── train_lora.py               # SFT training script
│   ├── train_grpo.py               # GRPO training (execution rewards) ✅ NEW
│   └── configs/                    # Training wrapper scripts
│       ├── qwen2.5-7b.sh          # 7B config (v1-3)
│       ├── qwen3-1.7b-sft-t9.sh   # Qwen3 SFT (Script 1) ✅ NEW
│       ├── qwen3-1.7b-grpo-t9.sh  # Qwen3 GRPO (Script 2) ✅ NEW
│       └── qwen3-1.7b-full-pipeline.sh  # Combined pipeline ✅ NEW
│
├── evaluation/                     # BIRD benchmark evaluation
│   ├── run_bird_eval.sh           # v3 evaluation (DDL schema)
│   ├── eval_bird.py               # Unified BIRD eval ✅ NEW
│   ├── eval_self_consistency.py   # Self-consistency voting ✅ NEW
│   ├── run_eval_basic.sh          # Basic eval wrapper (Script 3a) ✅ NEW
│   └── run_eval_self_consistency.sh  # SC eval wrapper (Script 3b) ✅ NEW
│
├── results/                        # Training outputs by model
│   ├── qwen2.5-7b/
│   │   └── v1-3/                  # 44.26% BIRD score
│   └── qwen2.5-7b/v5/
│       └── research/              # T9 dataset analysis & planning ✅ NEW
│           ├── T9_SPECIFICATION.md
│           ├── T9_V2_EVALUATION_REPORT.md
│           └── SOTA_METHODS_ANALYSIS.md
│
├── docs/                           # Documentation ✅ NEW
│   ├── TRAINING_GUIDE.md          # Complete training & eval guide
│   ├── CUDA_TROUBLESHOOTING.md    # GPU/CUDA setup issues
│   ├── EVALUATION_STRATEGY.md     # Parallel eval strategies
│   ├── QUICK_EVAL_GUIDE.md        # Quick reference for progress monitoring
│   ├── EVAL_PROGRESS_MONITORING.md # Detailed progress tracking guide
│   └── FAST_MODEL_DOWNLOAD.md     # Speed up slow HF downloads with aria2c
│
└── tools/                          # Dataset generation scripts
    └── create_t7_dataset.py       # Creates t7 dataset
```

## 📊 Results Timeline

| Date | Run | Dataset | Model | BIRD Score | Status |
|------|-----|---------|-------|------------|--------|
| Mar 28 | Initial | t2 (562) | 14B | 24.4%* | Custom eval only |
| Mar 30 | t3 | t3 (23.5K) | 14B | - | No BIRD eval |
| **Mar 31** | **v1-3** | **t7 (16.7K)** | **7B** | **44.26%** | ✅ **Best** |

*Custom eval, not official BIRD benchmark

### Evaluation Journey (v1-3)
- **v1:** 6.65% - No schema in prompt
- **v2:** 39.11% - Added simple schema (format mismatch)
- **v3:** 44.26% - Fixed DDL schema format (+5.15%)

## 🎯 Leaderboard Context

| Model | BIRD Execution Accuracy | Gap |
|-------|-------------------------|-----|
| CSC-SQL + Qwen 7B | 71.72% | Target |
| Claude Opus 4.6 | 70.15% | Far |
| DAIL-SQL + GPT-4 | 57.41% | Reachable |
| GPT-4 baseline | 54.89% | **-10.63%** |
| **Our Qwen 7B (v1-3)** | **44.26%** | Current |
| GPT-3.5 | 40.08% | Beat ✅ |

## 🚀 Quick Start

### Option 1: Qwen3-1.7B Pipeline (Recommended for Cloud GPU)

**Faster training, advanced techniques (GRPO + Self-Consistency)**

```bash
# Clone repo
git clone https://github.com/arnavgupta00/lora-train.git
cd lora-train

# Optional: Check environment setup
bash check_environment.sh

# Download BIRD benchmark (required for eval)
# Visit: https://bird-bench.github.io/
# Extract dev.json and dev_databases/ to ./bird_eval/

# Fast training (2-3 hours on RTX 3090)
# Dependencies auto-install on first run
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log

# Full training (6-8 hours, includes GRPO)
nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
```

**Expected Results:**
- Fast mode: ~55-60% BIRD accuracy
- Full mode: ~60-67% BIRD accuracy

See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for all commands and options.

---

### Option 2: Qwen2.5-7B (Proven 44.26% Result)

**Proven baseline, simpler setup**

```bash
cd lora-train

# Training (RunPod)
bash training/configs/qwen2.5-7b.sh
# ~50 minutes on RTX 5090

# BIRD Evaluation
bash evaluation/run_bird_eval.sh \
  --model-path results/qwen2.5-7b/qwen2.5-7b-t7-bird-*/
# ~45 minutes
```

## 📚 Key Documentation

**Results & Analysis:**
- [`results/qwen2.5-7b/v1-3/RUN_LOG.md`](results/qwen2.5-7b/v1-3/RUN_LOG.md) - Current best: 44.26% analysis
- [`CHANGELOG.md`](CHANGELOG.md) - Full project journey with dates
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) - Quick results comparison

**Training & Evaluation:**
- [`training/docs/README.md`](training/docs/README.md) - Training guide
- [`training/docs/RUNPOD_README.md`](training/docs/RUNPOD_README.md) - RunPod setup
- [`evaluation/versions/`](evaluation/versions/) - Eval debugging history (v1→v2→v3)

**Data:**
- [`data/training/t7/metadata.json`](data/training/t7/metadata.json) - t7 dataset details
- [`tools/create_t7_dataset.py`](tools/create_t7_dataset.py) - Dataset generation script

## 💡 Next Steps to Beat GPT-4

Based on v1-3 analysis, here are paths to close the 10.63% gap:

1. **Train for 3 epochs** (vs current 1 epoch)
   - Expected: +8-10% → **52-54%** (may beat GPT-4!)
   - Cost: 2.5 hours more training time
   - **Recommended first try**

2. **BIRD-only training data**
   - Remove custom schemas, focus on official BIRD
   - Expected: +10-15% → **54-59%**
   - Better benchmark alignment

3. **Increase LoRA rank** (16 → 32)
   - More parameters = better learning
   - Expected: +3-5%
   - Minimal cost increase

4. **Longer sequences** (1024 → 2048 tokens)
   - Currently truncating 0.1% of examples
   - Expected: +1-2%

See [`results/qwen2.5-7b/v1-3/RUN_LOG.md`](results/qwen2.5-7b/v1-3/RUN_LOG.md) for detailed recommendations.

## 🔍 Key Learnings

1. **Prompt format is critical:** Schema format mismatch caused 32% performance drop
2. **Training loss doesn't predict eval:** Loss 0.227 looked great but eval showed gaps
3. **DDL schema > simple format:** 39.11% → 44.26% just from proper schema extraction
4. **Domain matters:** Best on student_club (61.4%), worst on financial (30.2%)

## 📝 Dataset Information

- **t2:** 562 examples (custom schemas) - Used for initial 14B training
- **t7:** 16,699 examples (BIRD 9.4K + custom) - **Current best, use this** ✅
- **t3:** 23,577 examples (rebalanced) - Used for 14B, but missing locally (see `RECOVER_T3_DATASET.md`)

All datasets use ChatML format with response-only loss masking.

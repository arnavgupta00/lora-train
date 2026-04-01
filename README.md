# NL2SQL Fine-tuning Project

Fine-tuning small models (7B-14B) to compete with mainstream models on NL2SQL benchmarks.

## 🎯 Goal
Beat GPT-4 (54.89%) on BIRD benchmark using a 7B parameter model with LoRA fine-tuning.

## 🏆 Current Best Result

**Qwen2.5-7B + LoRA (v1-3): 44.26%** on BIRD benchmark
- Gap to GPT-4: -10.63% (closing in!)
- 37.61% improvement from initial attempts (6.65% → 44.26%)
- Training time: 50 minutes on RTX 5090
- Dataset: t7 (16,699 BIRD + custom examples)

See [`results/qwen2.5-7b/v1-3/RUN_LOG.md`](results/qwen2.5-7b/v1-3/RUN_LOG.md) for full analysis.

## 📁 Repository Structure

```
lm/
├── data/                           # All datasets
│   ├── raw/                        # BIRD/Spider original data
│   ├── processed/                  # ChatML formatted data
│   ├── training/                   # Training sets
│   │   ├── t2/                     # 562 examples (custom schemas)
│   │   └── t7/                     # 16,699 examples (BIRD + custom) ✅
│   └── t3_test1000_rebalanced/     # 23,577 examples (missing locally)
│
├── training/                       # Training code
│   ├── train_lora.py               # Main training script
│   ├── eval_exec.py                # Execution-match evaluation
│   ├── configs/                    # Per-model training configs
│   │   ├── qwen2.5-7b.sh          # 7B config (used for v1-3)
│   │   └── qwen2.5-14b.sh         # 14B config
│   └── docs/                       # Training documentation
│
├── evaluation/                     # BIRD benchmark evaluation
│   ├── run_bird_eval.sh           # Current (v3) - DDL schema extraction
│   └── versions/                   # v1 (6.65%) → v2 (39.11%) → v3 (44.26%)
│
├── results/                        # Training outputs by model
│   ├── qwen2.5-7b/
│   │   └── v1-3/                  # 44.26% BIRD score ✅
│   │       ├── RUN_LOG.md         # Full analysis & recommendations
│   │       ├── dataset_used/      # → data/training/t7/
│   │       ├── training_config.sh # → training/configs/qwen2.5-7b.sh
│   │       ├── evaluation_script.sh # → evaluation/run_bird_eval.sh
│   │       └── bird_evaluation/   # All BIRD results
│   └── qwen2.5-14b/
│       └── qwen2.5-14b-*/         # 24.4% on custom eval
│           ├── RUN_LOG.md
│           └── dataset_used_t3/   # → data/t3_test1000_rebalanced/
│
├── tools/                          # Dataset generation scripts
│   └── create_t7_dataset.py       # Creates t7 from BIRD + custom
│
└── experiments/                    # Archived experiments (preserved)
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

### Training (RunPod)
```bash
cd /workspace/lora-train
bash training/configs/qwen2.5-7b.sh  # Trains with t7 dataset
# Training time: ~50 minutes on RTX 5090
# Output: results/qwen2.5-7b/qwen2.5-7b-t7-bird-YYYYMMDD_HHMMSS/
```

### BIRD Evaluation
```bash
bash evaluation/run_bird_eval.sh \
  --model-path results/qwen2.5-7b/qwen2.5-7b-t7-bird-*/
# Evaluation time: ~45 minutes
# Output: bird_eval_report.json with execution accuracy
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

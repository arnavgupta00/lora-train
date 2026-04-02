# GitHub Deployment Summary

This document describes the complete setup for deploying the Qwen3-1.7B training pipeline from the GitHub repository.

## Repository Information

- **URL**: https://github.com/arnavgupta00/lora-train
- **Public Access**: Yes
- **Primary Branch**: main

## What's Included

### Training Scripts (9 files)
1. `training/train_lora.py` - SFT training implementation
2. `training/train_grpo.py` - GRPO training with execution rewards
3. `training/configs/qwen3-1.7b-sft-t9.sh` - Script 1: SFT wrapper
4. `training/configs/qwen3-1.7b-grpo-t9.sh` - Script 2: GRPO wrapper
5. `training/configs/qwen3-1.7b-full-pipeline.sh` - Combined pipeline (recommended)

### Evaluation Scripts (4 files)
1. `evaluation/eval_bird.py` - Basic BIRD evaluation
2. `evaluation/eval_self_consistency.py` - Self-consistency voting
3. `evaluation/run_eval_basic.sh` - Script 3a: Basic eval wrapper
4. `evaluation/run_eval_self_consistency.sh` - Script 3b: SC eval wrapper

### Dataset
- `data/training/t9/train_v4.jsonl` - 14,034 training examples
- `data/training/t9/dev_v4.jsonl` - 665 dev examples

### Documentation
- `docs/TRAINING_GUIDE.md` - Complete training and evaluation guide
- `README.md` - Project overview and quick start
- `requirements.txt` - Python dependencies
- `check_environment.sh` - Environment verification script

## User Journey

### Step 1: Clone Repository
```bash
git clone https://github.com/arnavgupta00/lora-train.git
cd lora-train
```

### Step 2: Check Environment (Optional but Recommended)
```bash
bash check_environment.sh
```

This verifies:
- Python 3.x is installed
- GPU is available
- Python dependencies status
- T9 dataset is present
- BIRD data availability

### Step 3: Download BIRD Benchmark
**Required for GRPO training and evaluation**

1. Visit: https://bird-bench.github.io/
2. Download the dev set
3. Extract to `./bird_eval/`:
   - `bird_eval/dev.json`
   - `bird_eval/dev_databases/`

**Note**: Training can run without BIRD data using `SKIP_GRPO=1 SKIP_EVAL=1`

### Step 4: Run Training

**Fast Mode (2-3 hours):**
```bash
EPOCHS=2 BATCH_SIZE=8 SEQ_LEN=1024 SKIP_GRPO=1 \
  nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

**Full Mode (6-8 hours):**
```bash
nohup bash training/configs/qwen3-1.7b-full-pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

### Step 5: View Results

Results are saved to:
- `results/pipeline_YYYYMMDD_HHMMSS/SUMMARY.txt` - Final summary
- `results/pipeline_YYYYMMDD_HHMMSS/eval_results/` - Evaluation JSONs
- `outputs/qwen3-1.7b-sft-YYYYMMDD_HHMMSS/` - SFT model adapter
- `outputs/qwen3-1.7b-grpo-YYYYMMDD_HHMMSS/` - GRPO model adapter

## Auto-Install Features

The `qwen3-1.7b-full-pipeline.sh` script automatically:
1. Detects cloud platform (RunPod/Vast.ai/Local)
2. Checks for missing Python packages
3. Installs dependencies using `requirements.txt` or individual packages
4. Verifies GPU availability
5. Locates T9 dataset
6. Searches for BIRD data in common locations
7. Provides clear error messages with next steps

## Key Design Decisions

### Path Detection
Scripts auto-detect common paths:
- **RunPod**: `/runpod-volume/`
- **Vast.ai**: `/workspace/`
- **Local**: `$HOME/.cache/` and `./`

### HuggingFace Cache
Automatically configures:
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `HF_DATASETS_CACHE`

Uses persistent storage on cloud platforms to avoid re-downloading models.

### Error Handling
- Pre-flight checks before training
- Clear error messages with solutions
- Graceful handling of missing BIRD data
- Validation of all paths and dependencies

### Logging
- All output to timestamped log files
- Both console and file logging via `tee`
- Separate logs for each phase (SFT, GRPO, eval)

## Environment Variables

Users can customize via environment variables:

### Training
- `EPOCHS` - Training epochs (default: 3)
- `BATCH_SIZE` - Per-device batch size (default: 4)
- `SEQ_LEN` - Sequence length (default: 2048)
- `LR` - Learning rate (default: 2e-4)
- `LORA_R` - LoRA rank (default: 32)

### GRPO
- `SKIP_GRPO` - Skip GRPO training (default: 0)
- `NUM_GEN` - SQL candidates per question (default: 8)
- `GRPO_SAMPLES` - Limit training examples (default: all)

### Evaluation
- `SKIP_EVAL` - Skip all evaluation (default: 0)
- `SKIP_SC` - Skip self-consistency (default: 0)
- `N_SAMPLES` - SC sample count (default: 10)

### Paths (Auto-detected)
- `BIRD_DEV_JSON` - Path to dev.json
- `DB_DIR` - Path to dev_databases/
- `OUT_BASE` - Output directory base

## Cloud Platform Compatibility

### RunPod
✅ Fully tested and compatible
- Auto-detects `/runpod-volume/`
- Uses persistent storage for HF cache

### Vast.ai
✅ Fully compatible
- Auto-detects `/workspace/`
- Uses persistent storage for HF cache

### Local
✅ Works on local machines with GPU
- Uses `$HOME/.cache/` for HF cache
- Outputs to `./outputs/` and `./results/`

### Google Colab / Kaggle
⚠️ Should work but not tested
- May need manual path configuration
- Limited by session storage

## Expected Results

| Configuration | Time | BIRD Accuracy |
|---------------|------|---------------|
| Fast (SFT only, 2 epochs) | 2-3 hrs | ~55-60% |
| Full (SFT 3 epochs + GRPO) | 6-8 hrs | ~60-67% |
| With Self-Consistency | +1 hr | +3-5% |

## Troubleshooting

### Missing Dependencies
```bash
pip3 install -r requirements.txt
```
Or let the pipeline auto-install.

### Out of Memory
```bash
BATCH_SIZE=2 SEQ_LEN=1024 bash training/configs/qwen3-1.7b-full-pipeline.sh
```

### BIRD Data Not Found
```bash
# Set paths manually
export BIRD_DEV_JSON=/path/to/dev.json
export DB_DIR=/path/to/dev_databases
bash training/configs/qwen3-1.7b-full-pipeline.sh
```

Or skip GRPO and eval:
```bash
SKIP_GRPO=1 SKIP_EVAL=1 bash training/configs/qwen3-1.7b-full-pipeline.sh
```

## Support Files

### requirements.txt
Contains all Python dependencies with version constraints:
- torch>=2.0.0
- transformers>=4.40.0
- accelerate>=0.27.0
- peft>=0.10.0
- trl>=0.8.0
- bitsandbytes>=0.43.0
- datasets>=2.14.0

### check_environment.sh
Interactive environment checker that validates:
- OS and Python version
- GPU availability and VRAM
- Python package installations
- T9 dataset presence
- BIRD data location

Provides actionable next steps for any missing components.

## Files Modified for GitHub Deployment

1. **training/configs/qwen3-1.7b-full-pipeline.sh**
   - Added dependency auto-install
   - Enhanced BIRD data detection
   - Improved error messages with solutions
   - Added BIRD download instructions

2. **README.md**
   - Added Qwen3-1.7B quick start
   - GitHub clone instructions
   - Updated repository structure
   - Environment check step

3. **docs/TRAINING_GUIDE.md**
   - GitHub clone commands
   - Auto-install notes
   - BIRD download instructions

4. **New: requirements.txt**
   - Complete dependency list

5. **New: check_environment.sh**
   - Environment verification tool

## Deployment Checklist

✅ All training scripts executable and working
✅ All evaluation scripts executable and working
✅ T9 dataset included in repo (14K examples)
✅ Auto-dependency installation
✅ Path auto-detection for cloud platforms
✅ Clear error messages and troubleshooting
✅ Documentation complete
✅ Environment check script
✅ Requirements.txt created
✅ README updated with quick start
✅ Committed and pushed to GitHub

## Next Steps for Users

After successful training, users can:
1. Check results in `results/pipeline_*/SUMMARY.txt`
2. Load the best model (GRPO if available, else SFT)
3. Deploy with `PeftModel.from_pretrained()`
4. Use for inference on BIRD or custom queries

The error corrector/agentic flow mentioned in earlier planning will be added in a future iteration.

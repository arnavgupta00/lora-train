# RunPod Training Instructions - T8 Dataset

**Target:** Train Qwen2.5-7B on T8 dataset and achieve 55-60% on BIRD benchmark (from current 44.26%)

---

## Prerequisites

- RunPod pod with:
  - GPU: RTX A6000 (48GB) or A100 40GB
  - Storage: 50GB+ network volume at `/runpod-volume`
  - Container: `runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04`

---

## Step 1: Clone Repository

```bash
cd /workspace
git clone https://github.com/arnavgupta00/lora-train.git
cd lora-train
```

---

## Step 2: Generate T8 Dataset

The t8 dataset is gitignored (too large). Generate it from source data:

```bash
# Verify source data exists
ls -la data/raw/bird_train.jsonl
ls -la data/raw/bird_dev.jsonl
ls -la data/training/t7/

# Generate t8 dataset (takes ~30 seconds)
python3 tools/dataset_creation/create_t8_dataset.py

# Verify t8 was created
ls -la data/training/t8/training/
ls -la data/training/t8/eval/

# Check example counts
wc -l data/training/t8/training/train.jsonl
wc -l data/training/t8/training/dev.jsonl
wc -l data/training/t8/eval/bird_dev.jsonl

# Expected output:
#   22782 train.jsonl
#     928 dev.jsonl
#    1534 bird_dev.jsonl
```

---

## Step 3: Download BIRD Benchmark (for evaluation)

```bash
cd /workspace/lora-train
mkdir -p bird_eval
cd bird_eval

# Download BIRD dev set (330 MB)
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip

# Extract - creates dev_20240627/ directory
unzip dev.zip

# The zip contains another zip for databases - extract it
cd dev_20240627
unzip dev_databases.zip

# Clean up the nested zip
rm dev_databases.zip

cd /workspace/lora-train
```

**Verify installation:**

```bash
# Check databases exist (should show 11)
find bird_eval/dev_20240627/dev_databases -name "*.sqlite" | wc -l

# List all databases
ls bird_eval/dev_20240627/dev_databases/
# Expected: california_schools, card_games, codebase_community, 
#           debit_card_specializing, european_football_2, financial,
#           formula_1, student_club, superhero, thrombosis_prediction, toxicology

# Check files
ls bird_eval/dev_20240627/
# Should see: dev.json, dev.sql, dev_tables.json, dev_databases/
```

---

## Step 4: Start Training

**Note:** The t8 training script does NOT require the NL2SQL_ADMIN_API_KEY. This was only needed for old custom evaluation (eval_exec.py) which we're not using. BIRD evaluation uses actual SQLite databases instead.

```bash
# Start training in background (no API key needed)
nohup bash training/configs/qwen2.5-7b-t8.sh > t8_train.log 2>&1 &

# Save the process ID
echo $! > t8_train.pid

# Monitor training
tail -f t8_train.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Training Config (qwen2.5-7b-t8.sh)
- **Dataset:** T8 (22,782 examples, 100% DDL format)
- **Epochs:** 2 (vs 1 in t7)
- **LoRA:** r=32, α=64 (vs r=16, α=32 in t7)
- **Learning Rate:** 5e-5
- **Batch Size:** 2 × 16 grad_acc = 32 effective
- **Sequence Length:** 1024 tokens
- **Expected Time:** ~2-3 hours on RTX 5090

---

## Step 5: Run BIRD Evaluation

After training completes:

```bash
# Update eval script with your adapter path
# Edit evaluation/run_bird_eval.sh line 44:
# ADAPTER_DIR="/workspace/lora-train/outputs/qwen2.5-7b-t8-bird-YYYYMMDD_HHMMSS"

# Find your adapter directory
ls -la /runpod-volume/outputs/
# or
ls -la outputs/

# Run evaluation
nohup bash evaluation/run_bird_eval.sh > bird_eval_t8.log 2>&1 &

# Monitor progress
tail -f bird_eval_t8.log

# Check results
cat outputs/bird_evaluation/bird_eval_report_v3.json
```

---

## Expected Results

### T7 Baseline (v1-3)
- **Execution Accuracy:** 44.26%
- **Dataset:** 16,699 examples, 50% simple format, 50% DDL
- **LoRA:** r=16, α=32
- **Epochs:** 1

### T8 Target
- **Target Accuracy:** 55-60%
- **Dataset:** 22,782 examples, 100% DDL format
- **Improvements:**
  - CASE: 5.5% → 23.4%
  - CTE: 0.5% → 5.1%
  - Window: 0.5% → 4.8%
  - Complex column training (backticks)
- **LoRA:** r=32, α=64
- **Epochs:** 2

### Comparison with Baselines
- **GPT-4 (zero-shot):** 54.89%
- **GPT-3.5:** 40.08%
- **Our T7:** 44.26%
- **Our T8 Goal:** 55-60%

---

## Troubleshooting

### OOM Error
```bash
# Reduce batch size
export TRAIN_BS=1
export GRAD_ACC=32
bash training/configs/qwen2.5-7b-t8.sh
```

### Flash Attention Error
```bash
# Already handled in script - uses SDPA instead
# No action needed
```

### Missing Dataset
```bash
# Regenerate t8
python3 tools/dataset_creation/create_t8_dataset.py
```

### Training Hangs at Checkpoint
```bash
# Resume from checkpoint
export RESUME_FROM="/runpod-volume/outputs/qwen2.5-7b-t8-bird-YYYYMMDD_HHMMSS/checkpoint-XXX"
bash training/configs/qwen2.5-7b-t8.sh
```

---

## File Locations

### Input Files
- **T8 Training Data:** `data/training/t8/training/train.jsonl` (22,782 examples)
- **T8 Dev Data:** `data/training/t8/training/dev.jsonl` (928 examples)
- **T8 Eval Data:** `data/training/t8/eval/bird_dev.jsonl` (1,534 examples)
- **Training Config:** `training/configs/qwen2.5-7b-t8.sh`
- **BIRD Databases:** `bird_eval/dev_20240627/dev_databases/`

### Output Files
- **Model Adapters:** `/runpod-volume/outputs/qwen2.5-7b-t8-bird-YYYYMMDD_HHMMSS/`
- **Training Logs:** `t8_train.log`
- **Eval Report:** `outputs/bird_evaluation/bird_eval_report_v3.json`
- **Predictions:** `outputs/bird_evaluation/predictions_v3.json`

---

## Quick Commands Reference

```bash
# Generate t8 dataset
python3 tools/dataset_creation/create_t8_dataset.py

# Start training
nohup bash training/configs/qwen2.5-7b-t8.sh > t8_train.log 2>&1 &

# Monitor training
tail -f t8_train.log

# Run evaluation
bash evaluation/run_bird_eval.sh

# Check results
cat outputs/bird_evaluation/bird_eval_report_v3.json | jq '.execution_accuracy'
```

---

## Success Criteria

✅ **Training:**
- Loss decreases smoothly to ~0.2-0.3
- No OOM errors
- Checkpoint saved successfully
- Training completes in 2-3 hours

✅ **Evaluation:**
- Execution accuracy: **55-60%** (target)
- Minimum acceptable: **50%** (10% improvement over t7)
- Stretch goal: **60%+** (close to GPT-4's 54.89%)

---

## Post-Training

1. **Save results locally:**
   ```bash
   # Create results directory
   mkdir -p results/qwen2.5-7b/t8-v1/
   
   # Copy important files
   cp /runpod-volume/outputs/qwen2.5-7b-t8-bird-*/adapter_*.safetensors results/qwen2.5-7b/t8-v1/
   cp outputs/bird_evaluation/bird_eval_report_v3.json results/qwen2.5-7b/t8-v1/
   cp t8_train.log results/qwen2.5-7b/t8-v1/
   ```

2. **Create RUN_LOG.md** documenting:
   - Accuracy achieved
   - Comparison with t7
   - Best/worst databases
   - Next steps if target not met

3. **Update main README.md** with new results

---

## Next Steps If Target Not Met

If accuracy is **below 55%**:

1. **Increase epochs:** 2 → 3
   ```bash
   export EPOCHS=3
   bash training/configs/qwen2.5-7b-t8.sh
   ```

2. **Try 14B model:**
   ```bash
   bash training/configs/qwen2.5-14b.sh  # Need to create this
   ```

3. **Add more complex column examples:**
   - Edit `tools/dataset_creation/create_t8_dataset.py`
   - Increase complex_examples from 80 to 500
   - Regenerate and retrain

If accuracy is **55-60%**: ✅ Success! LinkedIn post time!

If accuracy is **60%+**: 🎉 Exceeds expectations! Beat GPT-4!

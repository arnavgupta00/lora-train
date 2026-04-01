# Run v1-3: BIRD Benchmark Evaluation Journey

**Model:** Qwen2.5-7B-Instruct + LoRA  
**Dataset:** t7 (16,699 training examples)  
**Date:** March 31 - April 1, 2026  
**Status:** ✅ Completed

> **Note:** The t3 dataset is not included in this directory as it's excluded by `.gitignore` (large size). The t7 dataset used here is located at `data/training/t7/`.

---

## 📊 Final Results

### Evaluation Progression

| Version | Schema Format | Execution Accuracy | Change |
|---------|--------------|-------------------|--------|
| **v1** | No schema (db_id only) | **6.65%** | Baseline |
| **v2** | Simple `table(col1,col2)` | **39.11%** | +32.46% |
| **v3** | Full DDL `CREATE TABLE...` | **44.26%** | +5.15% |

**Total improvement: 37.61 percentage points** (6.65% → 44.26%)

### Benchmark Comparison

| Model | BIRD Execution Accuracy |
|-------|------------------------|
| **Our Model (v3)** | **44.26%** ✅ |
| GPT-4 baseline | 54.89% |
| DAIL-SQL + GPT-4 | 57.41% |
| Claude Opus 4.6 | 70.15% |
| CSC-SQL + Qwen 7B | 71.72% |

**Gap to GPT-4:** -10.63%  
**Gap to Claude Opus:** -25.89%

---

## 🔍 What Went Wrong (And Right)

### v1: Catastrophic Failure (6.65%)
**Root cause:** Model received only `"Database: california_schools"` without schema
- **Result:** Model refused to generate SQL ("I can't help with that")
- **Execution errors:** 1390/1534 (90.6%)
- **Learning:** Schema is absolutely critical

### v2: Major Improvement (39.11%)
**Change:** Added simplified schema: `users(id,name,email)`
- **Result:** Model started generating SQL
- **Execution errors:** 342/1534 (22.3%)
- **Problem:** Schema format didn't match training data
  - Training used: `CREATE TABLE users (id INTEGER, name TEXT...)`
  - Eval used: `users(id,name,...)`
- **Learning:** Schema format must match training exactly

### v3: Final Optimization (44.26%)
**Change:** Extracted full DDL CREATE TABLE statements from SQLite
- **Result:** 5.15% improvement
- **Execution errors:** 245/1534 (16.0%)
- **Success:** Schema format now matches training data
- **Learning:** Still gaps in model capability

---

## 📈 Performance by Database

### Best Performers (>50%)
1. **student_club**: 61.4% (97/158) ✅
2. **superhero**: 61.2% (79/129) ✅
3. **european_football_2**: 59.7% (77/129) ✅
4. **codebase_community**: 55.4% (103/186) ✅

### Worst Performers (<40%)
1. **california_schools**: 20.2% (18/89, 10 errors) ❌
2. **financial**: 30.2% (32/106, 14 errors) ❌
3. **card_games**: 36.1% (69/191, 43 errors) ❌
4. **thrombosis_prediction**: 36.2% (59/163, 45 errors) ❌
5. **toxicology**: 36.6% (53/145, 10 errors) ❌

---

## 🎯 Why We're Still Behind

### 1. Training Data Quality (Primary Issue)
- **t7 dataset**: 16,699 examples
  - 9,428 BIRD training examples
  - ~7,271 custom schema examples
- **Problem:** Custom examples may introduce noise
- **Evidence:** Poor performance on financial/California schools suggests domain-specific weakness

### 2. Model Size Limitation
- **7B parameters** vs competitors using larger models
- GPT-4: ~1.7T parameters (estimated)
- Claude: Unknown but likely >100B

### 3. Schema Complexity
- Worst performers have complex schemas:
  - `california_schools`: 21 columns, multiple foreign keys
  - `financial`: Nested transactions, complex aggregations
  - `thrombosis_prediction`: Medical domain terminology

### 4. Training Epochs
- **Only 1 epoch** (50 minutes training)
- Models with 3 epochs typically score 2-5% higher
- Trade-off: Speed vs accuracy

---

## 💡 What To Do Next

### Option 1: Quick Wins (Estimated +5-10%)
1. **Train for 3 epochs instead of 1**
   - Cost: 150 minutes training time
   - Expected: 49-54% accuracy
   - May reach GPT-4 baseline!

2. **Increase LoRA rank from 16 to 32**
   - More parameters = better learning
   - Cost: Slightly larger adapters (300MB vs 150MB)

3. **Longer sequences: 1024 → 2048 tokens**
   - Currently truncating 0.1% of examples
   - May help with complex queries

### Option 2: Better Data (Estimated +10-15%)
1. **BIRD-only training**
   - Remove custom schemas
   - Focus on official BIRD data
   - Expected: 54-59% (may beat GPT-4!)

2. **Add Spider benchmark data**
   - Another 7,000 examples
   - Different domain coverage
   - May improve generalization

### Option 3: Larger Model (Estimated +5-10%)
1. **Qwen2.5-14B instead of 7B**
   - Requires more VRAM (40GB+)
   - Cost: Longer training (90 min vs 50 min)
   - Expected: 49-54% accuracy

---

## 📝 Training Configuration

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "dataset": "t7 (16,699 examples)",
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05
  },
  "training": {
    "batch_size": 2,
    "gradient_accumulation": 16,
    "effective_batch": 32,
    "learning_rate": 5e-5,
    "epochs": 1,
    "time": "50 minutes",
    "loss": "1.136 → 0.227"
  }
}
```

---

## 📁 Files in This Run

- `qwen2.5-7b-t7-bird-20260331_193013/` - Training outputs
  - `adapter_model.safetensors` (154MB) - LoRA weights
  - `run.log` - Full training log
  - `run_meta.json` - Config used
  - `checkpoint-200/`, `checkpoint-309/` - Intermediate checkpoints

- `bird_evaluation/` - BIRD benchmark results
  - `bird_eval_report_v3.json` - Final results
  - `predictions_v3.json` - All 1,534 predictions
  - Earlier versions (v1, v2) for comparison

---

## 🎓 Key Learnings

1. **Prompt format is CRITICAL**
   - 6.65% → 44.26% just from schema format fixes
   - Must match training data exactly

2. **Schema matters more than model size**
   - Wrong schema: 6.65%
   - Right schema: 44.26%
   - Difference: 37.61% (vs ~5-10% from 7B→14B)

3. **Training loss doesn't tell full story**
   - Loss: 0.227 (excellent)
   - Accuracy: 44.26% (mediocre)
   - Need proper evaluation to know true performance

4. **Domain-specific weaknesses**
   - Medical (thrombosis): 36.2%
   - Financial: 30.2%
   - May need domain-specific training data

---

## 🚀 Recommended Next Step

**Train for 3 epochs with BIRD-only data**

Why:
- Highest ROI (may reach GPT-4 baseline)
- Low cost (just 2.5 hours more training)
- Clean data = better learning
- Can compare directly to other BIRD results

Command:
```bash
cd /workspace/lora-train
# Edit configs/qwen2.5-7b.sh: EPOCHS=3, use bird_train_chatml.jsonl only
bash training/configs/qwen2.5-7b.sh
```

Expected result: **52-58% accuracy** (close to or beating GPT-4!)

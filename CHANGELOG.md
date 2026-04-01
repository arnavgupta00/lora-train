# Changelog - NL2SQL Fine-tuning Journey

## Project Goal
Fine-tune a small model (7B-14B parameters) to beat mainstream models on NL2SQL benchmarks.

---

## 2026-03-28: First Training Run (t2 dataset)

### What we did
- Trained Qwen2.5-14B with LoRA on custom NL2SQL dataset
- 562 training examples from custom schemas (b2b_saas, hospitality_chain, etc.)

### Results
| Model | Execution Match |
|-------|-----------------|
| Base 14B | 12.2% |
| LoRA 14B | 24.4% |

**2x improvement!** But still far from BIRD benchmark targets.

### Learnings
- Need more training data
- Custom schemas alone aren't enough for benchmark

---

## 2026-03-30: Larger Dataset (t3 rebalanced)

### What we did
- Expanded to 23,577 training examples
- Rebalanced dataset with 1000 test examples
- Trained Qwen2.5-14B for 1.5 hours

### Results
- Training loss: 0.04 (very good convergence)
- But couldn't evaluate on BIRD benchmark yet

### Learnings
- Good training convergence doesn't guarantee good eval
- Need BIRD benchmark evaluation

---

## 2026-03-31: BIRD Benchmark Training (t7 dataset)

### What we did
- Created t7 dataset: 16,699 examples
  - 9,428 BIRD training examples
  - Custom schemas
  - Complex SQL patterns
- Trained Qwen2.5-7B with LoRA
- Training completed in 50 minutes on RTX 5090

### Training Metrics
- Initial loss: 1.136
- Final loss: 0.227
- Excellent convergence!

---

## 2026-04-01: BIRD Evaluation Journey

### v1: First Attempt - 0.33% 😱
**What went wrong:** Forgot to include schema in prompt!
- Model just saw "Database: california_schools"
- Responded with "I can't help with that"

### v2: Added Schema - 36.77%
**What went wrong:** Schema format mismatch!
- Training used: `CREATE TABLE users (id INTEGER, name TEXT...)`
- Evaluation used: `users(id,name,...)`

### v3: DDL Schema Format - ???%
**Fix:** Extract actual CREATE TABLE statements from SQLite
- Matches training data format exactly
- Running now...

---

## Key Learnings

1. **Prompt format must match training exactly**
   - System prompt, schema format, question format

2. **Schema format matters enormously**
   - DDL (CREATE TABLE) vs simplified (table(col)) = huge difference

3. **Training loss isn't everything**
   - 0.227 loss looked great, but eval revealed issues

4. **Version control your experiments**
   - Each eval version taught something new

---

## Benchmark Targets

| Model | BIRD Execution Accuracy |
|-------|------------------------|
| GPT-4 baseline | 54.89% |
| DAIL-SQL + GPT-4 | 57.41% |
| Claude Opus 4.6 | 70.15% |
| CSC-SQL + Qwen 7B | 71.72% |
| **Our Model** | **TBD** |

---

## Repository Structure (Updated 2026-04-01)

```
lm/
├── data/           # Datasets (raw, processed, training)
├── training/       # Training code and configs
├── evaluation/     # BIRD benchmark evaluation
│   └── versions/   # History of eval attempts
├── results/        # Model outputs and reports
├── tools/          # Data generation scripts
├── services/       # Validator service
└── experiments/    # Old experiments (archived)
```

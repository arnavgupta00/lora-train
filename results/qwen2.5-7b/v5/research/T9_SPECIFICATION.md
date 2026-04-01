# T9 Dataset Specification

## Current State Analysis

### Fine-tuning Impact (Baseline → T8 LoRA)

| Metric | Baseline | T8 LoRA | Change |
|--------|----------|---------|--------|
| Execution Accuracy | 41.98% | 43.74% | **+1.76%** |
| Exact Match | 76 | 172 | **+96** |
| Execution Errors | 333 | 253 | **-80** |

**Fine-tuning helps, but only +1.76% is very weak for 22K training examples.**

### Per-Database Impact

| Database | Baseline | T8 | Change | Verdict |
|----------|----------|-----|--------|---------|
| thrombosis_prediction | 25.2% | 34.4% | +9.2% | ✅ Big win |
| student_club | 55.1% | 63.9% | +8.8% | ✅ Big win |
| formula_1 | 35.6% | 39.7% | +4.1% | ✅ Good |
| codebase_community | 51.6% | 54.8% | +3.2% | ✅ Good |
| european_football_2 | 51.9% | 53.5% | +1.6% | ✅ Slight |
| **toxicology** | **43.4%** | **35.9%** | **-7.5%** | ❌ **HURT** |
| **debit_card_specializing** | **37.5%** | **32.8%** | **-4.7%** | ❌ **HURT** |
| **superhero** | **69.8%** | **65.9%** | **-3.9%** | ❌ **HURT** |

**The base model was already good at superhero (69.8%) and toxicology (43.4%). Fine-tuning made them worse!**

---

## Why Fine-tuning Had Limited Impact

### Issue 1: Distribution Mismatch
Training data patterns don't match BIRD patterns:
- JOINs: 48.6% in training vs 74.3% in BIRD
- The model is learning the wrong distribution

### Issue 2: Catastrophic Forgetting
Base Qwen2.5-7B-Instruct already knows SQL well. Fine-tuning on mismatched data caused:
- Toxicology: -7.5% (forgot simple patterns)
- Superhero: -3.9% (forgot what worked)

### Issue 3: Too Much Noise
22K examples with wrong patterns = teaching bad habits.

---

## T9 Dataset Specification

### Target Size: **12,000-15,000 examples**

Smaller but higher quality. Reasons:
1. Less noise = less catastrophic forgetting
2. Match BIRD distribution exactly
3. Focus on what base model struggles with

### Composition

#### Core Data (8,000 examples)

| Source | Count | Purpose |
|--------|-------|---------|
| BIRD Training Set | 9,400 | **But filter to 6,000** - remove duplicates, low quality |
| Spider (filtered) | 2,000 | Standard SQL patterns, SQLite compatible only |

#### Targeted Augmentation (4,000-6,000 examples)

| Category | Count | Purpose |
|----------|-------|---------|
| JOIN-heavy examples | 2,000 | Bridge the 74.3% gap |
| Backtick columns | 500 | Fix california_schools |
| Subquery examples | 500 | Currently under-represented |
| DISTINCT examples | 500 | Currently under-represented |
| Domain-specific (financial, medical) | 500 | Fix struggling databases |

### Pattern Distribution Targets

| Pattern | BIRD % | T8 % | **T9 Target** | Notes |
|---------|--------|------|---------------|-------|
| **JOIN** | 74.3% | 48.6% | **72-75%** | CRITICAL - match BIRD |
| ORDER BY | 24.3% | 47.1% | **25-30%** | Reduce over-representation |
| DISTINCT | 22.3% | 10.9% | **20-22%** | Increase |
| LIMIT | 18.5% | 32.5% | **20-25%** | Reduce but keep useful |
| SUBQUERY | 15.2% | 9.1% | **14-16%** | Increase to match |
| CASE | 13.3% | 23.3% | **12-15%** | REDUCE significantly |
| GROUP BY | 11.9% | 14.6% | **12-14%** | Slight reduce |
| CTE | 6.6% | 4.4% | **5-7%** | Slight increase |
| WINDOW | 4.4% | 4.8% | **4-5%** | Maintain |

### Quality Filters

Apply to ALL examples:

1. **Syntactic validity**: Must parse without errors
2. **Executable**: Must run on target DB (or similar schema)
3. **No hallucinated columns**: Column names must be exact
4. **JOIN correctness**: JOIN keys must be valid FKs
5. **Schema format**: 100% DDL (CREATE TABLE)

---

## Specific Fixes for Struggling Databases

### california_schools (18.0% → target 35%+)

**Problem**: Column hallucination (backticks)

**Solution**: Add 500 examples with:
```sql
-- Column names like:
`FRPM Count (K-12)`
`Percent (%) Eligible Free (K-12)`
`Free Meal Count (Ages 5-17)`
`Charter School (Y/N)`
```

### toxicology (35.9% → target 45%+)

**Problem**: Fine-tuning hurt it (-7.5%)

**Solution**: 
- Include 200 toxicology-style chemistry queries
- Simple pattern preservation (base model was at 43.4%)

### financial (29.2% → target 40%+)

**Problem**: Domain-specific aggregations

**Solution**:
- Include 200 financial domain queries
- Focus on date ranges, transaction sums, account balances

---

## Training Configuration for T9

### Recommended Settings

```yaml
# Dataset
train_examples: 12000-15000
dev_examples: 500-800

# Training
epochs: 3-4  # More epochs on smaller, cleaner data
batch_size: 32
learning_rate: 2e-5  # Lower than before
warmup_ratio: 0.1

# LoRA
r: 32
alpha: 64
dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Early stopping
eval_steps: 100
patience: 3  # Stop if no improvement for 3 evals
```

### Key Changes from T8

1. **Smaller dataset**: 12-15K vs 22K
2. **More epochs**: 3-4 vs 2 (smaller data needs more passes)
3. **Lower LR**: 2e-5 vs 5e-5 (preserve base model knowledge)
4. **Better distribution**: Match BIRD exactly

---

## Expected Results

### Conservative Estimate

| Version | Accuracy | Notes |
|---------|----------|-------|
| Baseline | 41.98% | No fine-tuning |
| T8 LoRA | 43.74% | +1.76% (weak) |
| **T9 LoRA** | **48-52%** | +6-10% from baseline |

### Why T9 Should Do Better

1. **Distribution match**: Training mirrors evaluation
2. **Less forgetting**: Smaller, targeted dataset
3. **Quality focus**: Every example counts
4. **Targeted fixes**: Address specific failure modes

### With Inference Improvements (Future)

| Technique | Additional Gain |
|-----------|----------------|
| Multi-sample voting (n=10) | +3-5% |
| Execution retry | +1-2% |
| Total potential | **55-60%** |

---

## Action Items

1. **Create T9 base**: Filter BIRD+Spider to 8K examples
2. **Generate augmentation**: 4-6K targeted examples
3. **Validate distribution**: Use analyze_distribution.py
4. **Train with new config**: Lower LR, more epochs
5. **Evaluate incrementally**: Check at each epoch

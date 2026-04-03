# Improvements to Boost NL2SQL Accuracy

## Current Status

- **Macro-average:** 66.8%
- **Best schemas:** 100% (logistics_fleet, real_estate_brokerage)
- **Worst schemas:** 1.9% (b2b_saas), 44.4% (cybersecurity_siem)

---

## 🚀 High-Impact Improvements

### 1. Fix Training Distribution (Expected: +15-20pp)

**Problem:** b2b_saas is 80% of test but only 17% of training.

**Fix:** Rebalance training data to match test distribution OR ensure equal representation.

```bash
# Create balanced training set
python3 scripts/nl2sql_eval/rebalance_splits.py \
  --input dataset/t3_test1000_rebalanced/all-all-train.qwen.jsonl \
  --output dataset/t3_balanced/all-all-train.qwen.jsonl \
  --target_per_schema 2000
```

**Action items:**
- [ ] Ensure each schema has 1500-2500 training examples
- [ ] If b2b_saas needs more examples, generate them
- [ ] Use stratified sampling for dev/test

---

### 2. Increase Training Epochs (Expected: +5-10pp)

**Current:** 1 epoch on 23K examples  
**Recommended:** 2-3 epochs

Your eval_loss was still decreasing (0.139 → 0.109), indicating underfitting.

```bash
# In your training command
--num_train_epochs 3
```

---

### 3. Lower Learning Rate (Expected: +3-5pp stability)

**Current:** 2e-4 (aggressive)  
**Recommended:** 5e-5 to 1e-4

Higher LR can overshoot optimal weights. Lower LR with more epochs is more stable.

```bash
--learning_rate 5e-5
--num_train_epochs 3
```

---

### 4. Increase Sequence Length (Expected: +2-5pp)

**Current:** 1024 tokens  
**Recommended:** 2048 tokens

Some complex queries (CTEs, subqueries) may be truncated.

```bash
--max_seq_len 2048
```

---

### 5. Add SQL Complexity to Training (Expected: +10-15pp on hard queries)

Your training data has very low complexity:
- Only 0.6% subqueries
- Only 0.1% CTEs

**Action items:**
- [ ] Generate training examples with CTEs
- [ ] Generate training examples with subqueries
- [ ] Generate training examples with window functions
- [ ] Add examples with UNION, EXCEPT, INTERSECT

**Example additions needed:**
```sql
-- CTE example
WITH monthly_revenue AS (
  SELECT organization_id, SUM(amount) as revenue
  FROM invoices
  GROUP BY organization_id
)
SELECT * FROM monthly_revenue WHERE revenue > 1000;

-- Subquery example
SELECT * FROM organizations
WHERE id IN (SELECT organization_id FROM invoices WHERE status = 'paid');

-- Window function example
SELECT name, revenue,
       RANK() OVER (ORDER BY revenue DESC) as rank
FROM organizations;
```

---

### 6. Enforce Output Contract (Expected: +5-10pp)

Many failures are due to column alias mismatches. Add explicit instructions:

**Current system prompt:**
```
You are a sqlite SQL generator. Return only SQL.
```

**Improved system prompt:**
```
You are a sqlite SQL generator. Return only SQL.
Rules:
- Use exact column aliases as specified in the question
- Always include ORDER BY for deterministic results
- Use ASC/DESC explicitly
```

---

### 7. Add Business Rules Context (Expected: +3-5pp)

Your training includes business rules but they could be more explicit:

```
Business rules:
- Revenue = invoices.total_amount_usd (NOT line items unless specified)
- Active = status = 'active'
- Paid invoices = status = 'paid'
```

---

## 📋 Quick Wins (Do These First)

| Improvement | Effort | Expected Gain |
|-------------|--------|---------------|
| Increase epochs to 3 | 5 min | +5-10pp |
| Lower LR to 5e-5 | 5 min | +3-5pp |
| Rebalance training | 30 min | +15-20pp |
| Increase seq_len to 2048 | 5 min | +2-5pp |

**Combined expected improvement:** 66.8% → **80-85%**

---

## 📊 Schema-Specific Fixes

### b2b_saas (1.9% → target 60%+)

**Problem:** Severely undertrained  
**Fix:** Generate 3000+ b2b_saas training examples

```bash
# Generate more b2b_saas examples via your dataset service
curl -X POST https://nl2sql-dataset-service.../v1/generate \
  -d '{"schema_id": "b2b_saas", "count": 3000}'
```

### cybersecurity_siem (44.4% → target 70%+)

**Problem:** Complex domain-specific terminology  
**Fix:** Add more diverse query patterns

### hr_ats_enterprise (50.0% → target 75%+)

**Problem:** Multiple table joins with application status tracking  
**Fix:** Add more examples with complex hiring funnel queries

---

## 🔧 Training Script Updates

Create an improved training script:

```bash
#!/bin/bash
# run_improved_training.sh

python3 finetune_nl2sql/train_lora.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --train_jsonl dataset/t3_balanced/all-all-train.qwen.jsonl \
  --dev_jsonl dataset/t3_balanced/all-all-dev.qwen.jsonl \
  --output_dir outputs/qwen14b-improved \
  --max_seq_len 2048 \           # Increased from 1024
  --pack \
  --per_device_train_batch_size 2 \  # Reduced for longer seqs
  --gradient_accumulation_steps 8 \   # Increased to compensate
  --learning_rate 5e-5 \         # Reduced from 2e-4
  --num_train_epochs 3 \         # Increased from 1
  --warmup_ratio 0.05 \          # Slightly more warmup
  --lora_r 32 \                  # Increased from 16
  --lora_alpha 64 \              # Keep 2:1 ratio
  --lora_dropout 0.05 \
  --gradient_checkpointing \
  --tf32
```

---

## 📈 Expected Results After Improvements

| Metric | Current | After Fixes |
|--------|---------|-------------|
| Macro Average | 66.8% | **80-85%** |
| b2b_saas | 1.9% | **55-65%** |
| cybersecurity_siem | 44.4% | **65-75%** |
| Best schemas | 100% | **100%** |

---

## Priority Order

1. **TODAY:** Increase epochs + lower LR + increase seq_len
2. **THIS WEEK:** Rebalance training distribution
3. **NEXT WEEK:** Add complex SQL patterns (CTEs, subqueries)
4. **ONGOING:** Generate more schema-specific examples for weak domains

---

*Last updated: 2026-03-31*

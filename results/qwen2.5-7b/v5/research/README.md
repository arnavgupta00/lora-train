# V5 Research Summary

## Objective
Understand why T8 training (44.26% → 43.74%) resulted in a score DROP and how to improve.

---

## Critical Finding: Fine-tuning Had Minimal Impact

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Base Qwen2.5-7B (no LoRA)** | **41.98%** | Zero fine-tuning |
| T8 Fine-tuned (22K examples) | 43.74% | 2 epochs, LoRA r=32 |
| **Improvement** | **+1.76%** | 😱 Terrible ROI |

**22,000 training examples only improved accuracy by 1.76%.**

Worse: Fine-tuning **hurt** databases where base model was already good:
- Toxicology: 43.4% → 35.9% (-7.5%)
- Superhero: 69.8% → 65.9% (-3.9%)

---

## Key Findings

### 1. Root Cause of Score Drop

**The upsampling strategy backfired.**

T8 over-indexed on CASE statements (5.1% → 23.3%) while diluting critical patterns:
- ORDER BY: 54.8% → 47.1% (needed for "top N" queries)
- LIMIT: 43.7% → 32.5% (needed for result limiting)
- DISTINCT: 12.2% → 10.9% (needed for deduplication)

Meanwhile, JOIN (the most important pattern at 74.3% in BIRD) only went from 45.9% → 48.6%.

### 2. Per-Database Analysis

Databases that improved with T8:
- superhero (+4.7%) - benefited from CTE/Window upsampling
- student_club (+2.5%) - simple queries matched better

Databases that degraded with T8:
- european_football_2 (-6.2%) - JOIN-heavy, hurt by CASE dilution
- debit_card_specializing (-4.7%) - pattern mismatch
- california_schools (-2.2%) - column hallucination persists

### 3. SOTA Gap Analysis

Our 7B model at 43.74% vs CSC-SQL's 7B at 71.72%:
- **Gap: 27.98 percentage points**
- Main difference: inference techniques (voting, self-correction, RL training)

---

## Recommendations

### Immediate Actions (T9 Dataset)

1. **Rebalance pattern distribution**
   - JOIN: Target 70% (up from 48.6%)
   - CASE: Target 15% (down from 23.3%)
   - ORDER/LIMIT: Maintain T7 levels

2. **Add backtick column examples**
   - Current: 80 examples
   - Target: 500+ examples
   - Focus on california_schools patterns

### Medium-Term Actions

3. **Implement multi-sample voting**
   - Generate 10 candidates per query
   - Execute each, vote on results
   - Expected gain: +3-5%

4. **Add execution retry loop**
   - If SQL errors, add error to prompt and retry
   - Expected gain: +1-2%

### Long-Term Actions

5. **Implement GRPO training**
   - Reward based on execution match
   - Train revision model for error correction
   - Expected gain: +5-10%

---

## Research Artifacts

| File | Purpose |
|------|---------|
| `BIRD_BENCHMARK_ANALYSIS.md` | Deep dive on BIRD evaluation methodology |
| `SCORE_DROP_INVESTIGATION.md` | Why T8 scored lower than T7 |
| `SOTA_METHODS_ANALYSIS.md` | How top methods achieve 70%+ |
| `ERROR_PATTERN_ANALYSIS.md` | Error patterns by database |
| `T9_SPECIFICATION.md` | **T9 dataset size, ratios, composition** |
| `analyze_distribution.py` | Tool to analyze dataset distributions |
| `t9_creation_guidelines.py` | Pattern distribution targets |
| `multi_sample_voting.py` | Self-consistency voting implementation |

---

## T9 Dataset Specification

### Size: 12,000-15,000 examples (SMALLER than T8's 22K)

**Rationale**: Quality > Quantity. Base model already knows SQL well.

### Pattern Distribution Targets

| Pattern | BIRD % | T8 % | T9 Target |
|---------|--------|------|-----------|
| **JOIN** | 74.3% | 48.6% | **72-75%** |
| ORDER BY | 24.3% | 47.1% | 25-30% |
| DISTINCT | 22.3% | 10.9% | 20-22% |
| LIMIT | 18.5% | 32.5% | 20-25% |
| SUBQUERY | 15.2% | 9.1% | 14-16% |
| **CASE** | 13.3% | 23.3% | **12-15%** |
| CTE | 6.6% | 4.4% | 5-7% |
| WINDOW | 4.4% | 4.8% | 4-5% |

### Composition

```
BIRD Training (filtered): 6,000 examples
Spider (SQLite only):     2,000 examples  
JOIN augmentation:        2,000 examples
Backtick columns:           500 examples
Subquery/DISTINCT:        1,000 examples
Domain-specific:            500 examples
───────────────────────────────────────
Total:                   12,000 examples
```

### Training Config Changes

- **Epochs**: 3-4 (more passes on smaller, cleaner data)
- **Learning Rate**: 2e-5 (lower to preserve base knowledge)
- **Early stopping**: After 3 evals without improvement

---

## Projected Improvements

| Change | Expected Gain | Cumulative |
|--------|---------------|------------|
| Baseline (no LoRA) | 41.98% | 41.98% |
| + T9 (better distribution) | +6-10% | ~48-52% |
| + Multi-sample voting | +3-5% | ~53-55% |
| + Execution retry | +1-2% | ~55-57% |
| + GRPO training (future) | +5-7% | ~60-65% |

**Realistic target with T9 + voting: 55-58%** (beating GPT-4's 54.89%)

---

## Next Steps

1. Create T9 dataset with corrected distribution (see T9_SPECIFICATION.md)
2. Train model on T9 (3-4 epochs, LR=2e-5)
3. Implement multi-sample voting inference
4. Evaluate and iterate

---

## 🚀 ALTERNATIVE PATH: Small Models (Recommended)

### The "Wow Factor" Approach

**Stop trying to make 7B work. Use smaller models that already work better.**

| Model | Size | BIRD Dev | Comparison |
|-------|------|----------|------------|
| Our Qwen2.5-7B LoRA | 7B | 43.74% | ❌ Poor |
| SLM-SQL-0.5B | **0.5B** | 56.87% | ✅ Beats GPT-4! |
| SLM-SQL-1.5B | **1.5B** | 67.08% | ✅ Near SOTA |

**Key Insight**: A 0.5B model (14x smaller) scores 13% higher than our 7B model.

### Why Small Models Win

1. **Better accuracy**: 67% vs our 44%
2. **Runs on laptops**: 8GB VRAM is enough
3. **100x cheaper inference**: No server needed
4. **Privacy**: Data never leaves the device
5. **"Wow factor"**: "Our 1.5B model beats GPT-4 on SQL"

### Small Model Resources

See `path-small/` folder for detailed research:

| File | Description |
|------|-------------|
| `SMALL_MODEL_STRATEGY.md` | Overall strategy and architecture |
| `DECISION_MATRIX.md` | Why small models are the better path |
| `IMPLEMENTATION_ROADMAP.md` | Week-by-week implementation plan |
| `tinysql_pipeline.py` | Code outline for pipeline |
| `quickstart.py` | Quick start script to test SLM-SQL |

### Quick Start

```bash
# Check dependencies
python path-small/quickstart.py --check

# Download and test SLM-SQL-1.5B
python path-small/quickstart.py --test
```

### Published Models (Ready to Use)

- `cycloneboy/SLM-SQL-0.5B` - 56.87% BIRD Dev
- `cycloneboy/SLM-SQL-1.5B` - 67.08% BIRD Dev
- `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct` - Merge/revision model

### Decision

**Two parallel paths forward:**

| Path | Approach | Expected Result | Time |
|------|----------|-----------------|------|
| **Path A: T9** | Fix training data, iterate | 50-55% | 2-3 weeks |
| **Path B: Small** | Use SLM-SQL models | 67%+ | 1 week |

**Recommendation**: Pursue Path B (Small Models) as primary, Path A (T9) as secondary research.

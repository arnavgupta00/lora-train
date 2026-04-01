# V5 Research Summary

## Objective
Understand why T8 training (44.26% → 43.74%) resulted in a score DROP and how to improve.

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
| `BIRD_BENCHMARK_ANALYSIS.md` | Deep dive on BIRD benchmark and evaluation |
| `SCORE_DROP_INVESTIGATION.md` | Why T8 scored lower than T7 |
| `SOTA_METHODS_ANALYSIS.md` | How top methods achieve 70%+ |
| `analyze_distribution.py` | Tool to analyze dataset distributions |
| `t9_creation_guidelines.py` | Guidelines for creating T9 dataset |
| `multi_sample_voting.py` | Implementation of voting inference |

---

## Projected Improvements

| Change | Expected Gain | Cumulative |
|--------|---------------|------------|
| Baseline (T8) | 43.74% | 43.74% |
| + T9 distribution fix | +3-4% | ~47% |
| + Multi-sample voting | +4-5% | ~51% |
| + Execution retry | +1-2% | ~53% |
| + GRPO training | +5-7% | ~58-60% |

**Realistic target: 55-60%** (beating GPT-4 baseline of 54.89%)

---

## Next Steps

1. Create T9 dataset with corrected distribution
2. Train model on T9 (3 epochs)
3. Implement multi-sample voting inference
4. Evaluate and iterate

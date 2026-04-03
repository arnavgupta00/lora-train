# Score Drop Investigation: T7 (44.26%) → T8 (43.74%)

## Executive Summary

**Score dropped by 0.52%** despite adding more "complex" queries to the training data.

### Primary Cause: Distribution Shift
The upsampling strategy overcorrected, creating a training distribution that diverges from BIRD's actual query patterns.

---

## Evidence

### 1. Pattern Frequency Analysis

```
Pattern         BIRD Actual   T7 Training   T8 Training   Problem
────────────────────────────────────────────────────────────────
JOIN            74.3%         45.9%         48.6%         Still 26% under BIRD
ORDER BY        24.3%         54.8%         47.1%         30% over BIRD
LIMIT           18.5%         43.7%         32.5%         Still 14% over BIRD  
CASE            13.3%         5.1%          23.3%         NOW 10% OVER BIRD ⚠️
CTE             6.6%          0.4%          4.4%          Better but still under
WINDOW          4.4%          0.5%          4.8%          Good match ✓
SUBQUERY        15.2%         4.7%          9.1%          Still 6% under
DISTINCT        22.3%         12.2%         10.9%         Got worse!
```

### 2. Per-Database Performance Changes

```
Database                    T7 Acc    T8 Acc    Change    Analysis
─────────────────────────────────────────────────────────────────
superhero                   61.2%     65.9%     +4.7%     ✅ Benefited from CTE/Window
student_club                61.4%     63.9%     +2.5%     ✅ Simple queries benefited
formula_1                   39.1%     39.7%     +0.6%     ≈ Marginal
card_games                  36.1%     36.1%     +0.0%     ≈ No change
codebase_community          55.4%     54.8%     -0.6%     ⚠️ Minor regression
toxicology                  36.6%     35.9%     -0.7%     ⚠️ Minor regression
financial                   30.2%     29.2%     -1.0%     ⚠️ Complex aggregations hurt
thrombosis_prediction       36.2%     34.4%     -1.8%     ❌ Medical domain harder
california_schools          20.2%     18.0%     -2.2%     ❌ Column hallucination worse
debit_card_specializing     37.5%     32.8%     -4.7%     ❌ Pattern mismatch
european_football_2         59.7%     53.5%     -6.2%     ❌ JOIN-heavy DB hurt
```

### 3. Error Analysis

**Execution Errors Increased**: T7: 245 (15.97%) → T8: 253 (16.49%)

Most common error type: **Column Hallucination**
- Model writes `FrpmCount` instead of `` `FRPM Count (K-12)` ``
- Model writes `T1.MailStreet` when column is in T2
- Model invents columns that don't exist

### 4. Exact Match Dropped

T7: 190 exact matches → T8: 172 exact matches (-9.5% relative drop)

The model is generating more "creative" SQL variations that don't match gold SQL structure, suggesting overfitting to CASE/CTE patterns.

---

## Root Cause Analysis

### Theory: Catastrophic Forgetting from Over-Upsampling

When T8 upsampled CASE statements from 5.1% → 23.3% (nearly 5x increase), the model:

1. **Learned to overuse CASE**: Even when simpler solutions exist
2. **Forgot simpler patterns**: LIMIT/ORDER familiarity decreased
3. **Added unnecessary complexity**: Leading to more execution errors

### Evidence for This Theory

1. **ORDER BY decreased**: 54.8% → 47.1% (-7.7%)
2. **LIMIT decreased**: 43.7% → 32.5% (-11.2%)
3. **DISTINCT decreased**: 12.2% → 10.9% (-1.3%)

These are HIGH-FREQUENCY patterns in BIRD (especially ORDER + LIMIT for "top N" queries).

---

## Why european_football_2 Dropped -6.2%

This database has:
- **Complex JOINs**: Player → Player_Attributes → Match
- **Multiple subqueries**: For statistical aggregations
- **Low CASE usage**: Queries are JOIN-heavy, not CASE-heavy

T8's CASE upsampling actively hurt this database because:
1. Model now tries to use CASE where JOIN is appropriate
2. JOIN training signal was diluted by CASE examples
3. Pattern recognition for football queries degraded

---

## Why california_schools Dropped -2.2% (Despite Added Examples)

Even though T8 added 80 complex column examples for california_schools:

1. **Still only 0.35% of training data** (80/22,782)
2. **Diluted by CASE upsampling**: Model saw 5,329 CASE examples
3. **Ratio problem**: 5,329 CASE vs 80 backtick = 67:1 ratio

The model learned CASE patterns 67x more than california_schools patterns.

---

## Recommendations

### For T9 Dataset

1. **Do NOT over-upsample any single pattern**
   - Target: Match BIRD distribution ± 5%
   - CASE should be ~15%, not 23%

2. **Increase JOIN examples to 70%+**
   - This is the most critical pattern
   - Currently massively underrepresented

3. **Maintain LIMIT/ORDER levels from T7**
   - These are essential for "top N" queries
   - T8's reduction hurt performance

4. **Increase backtick examples to 500+**
   - Current 80 is insufficient
   - Target: 2-3% of dataset

### Pattern Distribution Targets for T9

| Pattern | BIRD % | T7 % | T8 % | T9 Target |
|---------|--------|------|------|-----------|
| JOIN | 74.3% | 45.9% | 48.6% | **70%** |
| ORDER BY | 24.3% | 54.8% | 47.1% | **30%** |
| DISTINCT | 22.3% | 12.2% | 10.9% | **20%** |
| LIMIT | 18.5% | 43.7% | 32.5% | **25%** |
| SUBQUERY | 15.2% | 4.7% | 9.1% | **15%** |
| CASE | 13.3% | 5.1% | 23.3% | **15%** |
| GROUP BY | 11.9% | 15.5% | 14.6% | **12%** |
| CTE | 6.6% | 0.4% | 4.4% | **6%** |
| WINDOW | 4.4% | 0.5% | 4.8% | **5%** |

---

## Key Insight

**More data ≠ Better performance**

Quality and distribution match matter more than quantity. T8 had:
- 22,782 examples (vs T7's 16,699)
- 36% more data
- 0.5% worse performance

The extra 6,083 examples were mostly CASE/CTE upsampling that hurt overall distribution.

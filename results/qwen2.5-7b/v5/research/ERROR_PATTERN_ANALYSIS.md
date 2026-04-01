# Error Pattern Deep Dive

## Executive Summary

**T8 had 8 more execution errors than T7** (253 vs 245), but the distribution changed significantly.

---

## Error Count by Database

| Database | T7 Errors | T8 Errors | Change | Analysis |
|----------|-----------|-----------|--------|----------|
| california_schools | 10 (11.2%) | 23 (25.8%) | **+13** | 🔴 Major regression |
| financial | 14 (13.2%) | 25 (23.6%) | **+11** | 🔴 Major regression |
| formula_1 | 20 (11.5%) | 24 (13.8%) | +4 | ⚠️ Minor regression |
| student_club | 21 (13.3%) | 25 (15.8%) | +4 | ⚠️ Minor regression |
| card_games | 43 (22.5%) | 44 (23.0%) | +1 | ≈ No significant change |
| toxicology | 10 (6.9%) | 10 (6.9%) | 0 | ✓ Stable |
| european_football_2 | 15 (11.6%) | 14 (10.9%) | -1 | ✓ Slight improvement |
| codebase_community | 30 (16.1%) | 27 (14.5%) | -3 | ✓ Improved |
| debit_card_specializing | 10 (15.6%) | 5 (7.8%) | -5 | ✅ Significant improvement |
| superhero | 27 (20.9%) | 21 (16.3%) | -6 | ✅ Significant improvement |
| thrombosis_prediction | 45 (27.6%) | 35 (21.5%) | **-10** | ✅ Major improvement |

---

## Analysis

### Databases with Error Regression

#### 1. california_schools (+13 errors, +130%)

**Root Cause**: Column name hallucination

Example errors:
```sql
-- Model writes:
SELECT T1.FrpmCount FROM frpm...
-- Should be:
SELECT T1.`FRPM Count (K-12)` FROM frpm...

-- Model writes:
SELECT T1.MailStreet FROM frpm AS T1...
-- Should be:
SELECT T2.MailStreet FROM schools AS T2...
```

**Why T8 made it worse**:
- Only 80 backtick column examples in T8 (0.35% of dataset)
- Diluted by 5,329 CASE examples (23% of dataset)
- Model learned CASE patterns 67x more than california_schools patterns

**Fix**: Add 500+ california_schools-style examples

#### 2. financial (+11 errors, +78%)

**Root Cause**: Complex aggregation queries failing

This database requires:
- Nested aggregations (SUM of COUNTs)
- Date range calculations
- Multi-table JOINs with financial logic

**Why T8 made it worse**:
- T8 reduced LIMIT/ORDER patterns (common in financial queries)
- Added CASE patterns that don't match financial query style
- Domain-specific vocabulary not in training data

**Fix**: Add financial domain-specific examples

### Databases with Error Improvement

#### 1. thrombosis_prediction (-10 errors, -22%)

**Improved because**:
- CTE upsampling helped with medical data aggregations
- Window functions useful for patient timeline analysis

#### 2. superhero (-6 errors, -22%)

**Improved because**:
- Simpler schema benefited from cleaner DDL format
- Window functions useful for ranking queries

#### 3. debit_card_specializing (-5 errors, -50%)

**Improved because**:
- More consistent schema handling
- T8's DDL format matched evaluation better

---

## Net Impact

**Error trade-off**:
- Lost: +24 errors (california_schools, financial)
- Gained: -24 errors (thrombosis, superhero, debit_card, codebase)
- Net: +8 errors overall

**Accuracy trade-off**:
- Lost: california_schools (-2.2%), financial (-1.0%), european_football_2 (-6.2%)
- Gained: superhero (+4.7%), student_club (+2.5%)
- Net: -0.52% overall

---

## Key Insight

**The model improved on some databases but regressed on others.**

This suggests the training changes helped for certain query types but hurt others. The net effect was slightly negative because the databases that regressed (california_schools, european_football_2, financial) have more examples in the dev set.

### Database Weight Analysis

| Database | Examples | T8 Change | Weighted Impact |
|----------|----------|-----------|-----------------|
| european_football_2 | 129 | -6.2% | -8.0 queries |
| california_schools | 89 | -2.2% | -2.0 queries |
| financial | 106 | -1.0% | -1.1 queries |
| superhero | 129 | +4.7% | +6.1 queries |
| student_club | 158 | +2.5% | +4.0 queries |

**Net: -1 query correctly answered** (which aligns with the 0.52% drop)

---

## Recommendations

1. **Don't sacrifice high-example databases**
   - european_football_2 (129) and financial (106) have significant weight
   - Their regression offset gains elsewhere

2. **Target california_schools specifically**
   - Only 89 examples but disproportionate impact
   - Column hallucination is a clear, fixable problem

3. **Maintain balance**
   - T8's gains on thrombosis_prediction were good
   - But shouldn't come at cost of other databases

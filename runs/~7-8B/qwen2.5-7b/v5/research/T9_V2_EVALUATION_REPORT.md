# T9 v2 Dataset Evaluation Report

**Generated:** 2026-04-02  
**Files Analyzed:**
- `train_v2.jsonl` (10,434 examples)
- `dev_v2.jsonl` (665 examples)

---

## Executive Summary

| Metric | v1 | v2 | Change | Status |
|--------|----|----|--------|--------|
| **Total Examples** | 10,995 | 11,099 | +104 | ✅ |
| **CTE Examples** | 36 (0.3%) | 303 (2.9%) | +267 | ⚠️ Improved but not enough |
| **WINDOW Examples** | 12 (0.1%) | 290 (2.8%) | +278 | ⚠️ Improved but not enough |
| **Unknown db_id** | 1,574 (15.2%) | 1,692 (16.2%) | +118 | ❌ WORSE |
| **BIRD DB Coverage** | 0/6 | 0/6 | No change | ❌ CRITICAL |
| **Patterns in Target** | 2/9 | 1/9 | -1 | ❌ WORSE |

### Verdict: **v2 MADE SOME THINGS BETTER, SOME WORSE**

The CTE and WINDOW additions are positive, but new issues were introduced.

---

## Pattern Distribution Comparison

| Pattern | v1 % | v2 % | Target % | v1 → v2 | Status |
|---------|------|------|----------|---------|--------|
| **JOIN** | 78.5% | 69.0% | 72-75% | -9.5% | ❌ Now LOW (was HIGH) |
| **ORDER_BY** | 26.7% | 19.3% | 25-30% | -7.4% | ❌ Now LOW (was OK) |
| **DISTINCT** | 12.4% | 13.1% | 20-22% | +0.7% | ❌ Still LOW |
| **LIMIT** | 21.3% | 16.6% | 20-25% | -4.7% | ❌ Now LOW (was OK) |
| **SUBQUERY** | 8.6% | 11.7% | 14-16% | +3.1% | ⚠️ Improved but LOW |
| **CASE** | 7.1% | 8.1% | 12-15% | +1.0% | ⚠️ Improved but LOW |
| **GROUP_BY** | 16.4% | 12.6% | 12-14% | -3.8% | ✅ Now OK! |
| **CTE** | 0.3% | 2.9% | 5-7% | +2.6% | ⚠️ Much better but LOW |
| **WINDOW** | 0.1% | 2.8% | 4-5% | +2.7% | ⚠️ Much better but LOW |

### Patterns That Got Worse
- JOIN: 78.5% → 69.0% (now below target)
- ORDER_BY: 26.7% → 19.3% (now below target)
- LIMIT: 21.3% → 16.6% (now below target)

### Patterns That Improved
- CTE: 0.3% → 2.9% (major improvement, need +2% more)
- WINDOW: 0.1% → 2.8% (major improvement, need +1.5% more)
- GROUP_BY: 16.4% → 12.6% (now in target range)
- SUBQUERY: 8.6% → 11.7% (getting closer)

---

## Critical Issue: Unknown db_id

### Problem: New CTE/WINDOW Examples Lack db_id

| Pattern | Total | Unknown db_id | % Unknown |
|---------|-------|---------------|-----------|
| CTE | 303 | 286 | **93.8%** |
| WINDOW | 290 | 285 | **98.3%** |

**Impact:** Without db_id, these examples:
1. Cannot be validated against actual database schemas
2. May have column/table name mismatches
3. Reduce training data quality
4. May confuse the model during inference

### Sample Unknown db_id Entry
```
Schema:
CREATE TABLE schools (school_id INTEGER PRIMARY KEY, name TEXT, district TEXT, type TEXT);
CREATE TABLE teachers (teacher_id INTEGER PRIMARY KEY...

SQL: SELECT name, subject, ROW_NUMBER() OVER (PARTITION BY subject ORDER BY name) 
     as subject_alpha_rank FROM teachers
```
These appear to be synthetic examples - need db_id assignment or removal.

---

## BIRD Struggling Databases

**Status: UNCHANGED - Still Zero Coverage**

| Database | v1 | v2 | Required | Gap |
|----------|----|----|----------|-----|
| california_schools | 0 | 0 | 500 | -500 |
| toxicology | 0 | 0 | 200 | -200 |
| financial | 0 | 0 | 200 | -200 |
| debit_card_specializing | 0 | 0 | 100 | -100 |
| superhero | 0 | 0 | 100 | -100 |
| thrombosis_prediction | 0 | 0 | 100 | -100 |

**Total Missing:** ~1,200 examples for struggling databases

---

## Quality Analysis

### Random Sample Quality
- **Issue-free:** 18/20 (90%) ✅
- Main issues: Missing db_id on 2 examples

### Advanced Query Quality (CTE/WINDOW)
- **Issue-free:** 0/20 (0%) ❌
- **ALL 20 sampled CTE/WINDOW examples have unknown db_id**
- 3/20 use SELECT * anti-pattern
- 1/20 has malformed CTE structure

### Subquery Quality
- **Issue-free:** 10/20 (50%) ⚠️
- Main issues: Missing db_id, some SELECT * usage

### Anti-patterns Found
- SELECT * usage: 183 examples (1.8%)
- LIMIT without ORDER BY: Multiple instances
- Missing db_id on new examples: 93-98%

---

## Recommendations for v3

### Priority 1: Fix Unknown db_id (CRITICAL)

Option A: Assign realistic db_ids to synthetic examples
```python
# Map synthetic schemas to similar BIRD databases
schema_mapping = {
    'schools/teachers': 'student_club',
    'salespeople/sales': 'retail_world',
    'restaurants/orders': 'restaurant',
}
```

Option B: Remove synthetic examples and regenerate from real BIRD schemas
- More work but higher quality
- Ensures schema alignment

### Priority 2: Rebalance Patterns

Current gaps to fix:
```
JOIN:     +3% needed (add ~300 JOIN examples)
ORDER_BY: +6% needed (add ~600 ORDER BY examples)
DISTINCT: +7% needed (add ~700 DISTINCT examples)
LIMIT:    +4% needed (add ~400 LIMIT examples)
CTE:      +2% needed (add ~200 CTE examples)
WINDOW:   +2% needed (add ~200 WINDOW examples)
```

### Priority 3: Add BIRD Database Examples

Generate examples for each struggling database:
```
california_schools:     500 (focus on backtick columns)
toxicology:             200 (chemistry/molecular)
financial:              200 (date ranges, aggregations)
debit_card_specializing: 100 (transaction patterns)
superhero:              100 (preserve base model knowledge)
thrombosis_prediction:  100 (medical domain)
```

### Priority 4: Remove Anti-patterns

- Remove/fix 183 SELECT * examples
- Fix LIMIT without ORDER BY cases
- Ensure all CTEs have proper structure

---

## Expected Impact

### If v2 Is Used As-Is
- **Predicted Accuracy:** 42-46%
- Pattern regression may hurt JOIN and ORDER BY performance
- Unknown db_id examples may add noise
- Still no coverage of struggling databases

### If v3 Fixes Are Applied
- **Predicted Accuracy:** 52-58%
- Proper pattern distribution
- Quality CTE/WINDOW training
- BIRD database coverage prevents regression

---

## Action Checklist for v3

```
□ Fix db_id for 1,692 unknown entries (or remove/regenerate)
□ Add ~300 JOIN examples to reach 72%
□ Add ~600 ORDER BY examples to reach 25%  
□ Add ~700 DISTINCT examples to reach 20%
□ Add ~400 LIMIT examples to reach 20%
□ Add ~200 more CTE examples to reach 5%
□ Add ~200 more WINDOW examples to reach 4%
□ Generate 1,200 BIRD struggling database examples
□ Remove 183 SELECT * examples
□ Validate all SQLite syntax
□ Re-run quality analysis before training
```

---

## Summary

| Aspect | v1 Score | v2 Score | Change |
|--------|----------|----------|--------|
| Pattern Distribution | 2/9 | 1/9 | ❌ Worse |
| CTE/WINDOW Coverage | 1/10 | 5/10 | ✅ Better |
| Quality (random) | 6.5/10 | 7/10 | ✅ Better |
| Quality (advanced) | 5/10 | 3/10 | ❌ Worse (db_id issue) |
| BIRD DB Coverage | 0/10 | 0/10 | ➖ No change |
| **Overall** | **4.5/10** | **4/10** | **❌ Slightly Worse** |

**Bottom Line:** v2 is NOT an improvement over v1 due to:
1. Pattern regression (JOIN, ORDER_BY, LIMIT dropped below target)
2. Massive unknown db_id problem (93-98% of new CTE/WINDOW examples)
3. Still zero BIRD struggling database coverage

**Recommendation:** Create v3 with the fixes outlined above before training.

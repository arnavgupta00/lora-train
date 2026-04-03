# T9 Dataset Evaluation Report

**Generated:** 2026-04-02  
**Target:** BIRD Benchmark 60-67% accuracy with Qwen3-1.7B  
**Current Baseline:** 43.74% (T8 with Qwen2.5-7B)

---

## Executive Summary

| Metric | Status | Action Required |
|--------|--------|-----------------|
| **Dataset Size** | ✅ Good | 10,995 total (10,336 train + 659 dev) within 12-15K target |
| **Pattern Distribution** | ❌ Issues | 5 of 9 patterns outside target ranges |
| **Quality (Random Sample)** | ⚠️ 6.5/10 | ~15% samples have issues |
| **Quality (Complex Queries)** | ❌ 5/10 | 55% have anti-patterns or issues |
| **Quality (Simple Queries)** | ⚠️ 6.5/10 | 2 critical errors, 4 trivial examples |
| **BIRD DB Coverage** | ❌ Missing | 0 examples from 6 struggling databases |

### Verdict: **NEEDS MODIFICATION BEFORE TRAINING**

The T9 dataset has a solid foundation but requires significant corrections before it can achieve the 60-67% accuracy target.

---

## 1. Pattern Distribution Analysis

### Current vs Target Comparison

| Pattern | BIRD % | T9 % | Target % | Status | Gap |
|---------|--------|------|----------|--------|-----|
| **JOIN** | 74.3% | **78.5%** | 72-75% | ❌ HIGH | +3.5% |
| **ORDER BY** | 24.3% | 26.7% | 25-30% | ✅ OK | - |
| **DISTINCT** | 22.3% | **12.4%** | 20-22% | ❌ LOW | -7.6% |
| **LIMIT** | 18.5% | 21.3% | 20-25% | ✅ OK | - |
| **SUBQUERY** | 15.2% | **8.6%** | 14-16% | ❌ LOW | -5.4% |
| **CASE** | 13.3% | **7.1%** | 12-15% | ❌ LOW | -4.9% |
| **GROUP BY** | 11.9% | **16.4%** | 12-14% | ❌ HIGH | +2.4% |
| **CTE** | 6.6% | **0.3%** | 5-7% | ❌ CRITICAL | -6.3% |
| **WINDOW** | 4.4% | **0.1%** | 4-5% | ❌ CRITICAL | -4.3% |

### Critical Gaps

1. **CTEs (WITH clauses):** Only 36 examples (0.3%) vs target 5-7%. The dataset claims complex patterns but has almost none.
2. **Window Functions:** Only 12 examples (0.1%) vs target 4-5%. Critical for ranking queries.
3. **DISTINCT:** 12.4% vs target 20-22%. Under-represented.
4. **SUBQUERY:** 8.6% vs target 14-16%. Needs 500+ more examples.

---

## 2. Database Coverage

### BIRD Struggling Databases - ZERO Coverage

| Database | Issue | T9 Count | Required |
|----------|-------|----------|----------|
| california_schools | Backtick columns | **0** | 500 |
| toxicology | Fine-tuning hurt (-7.5%) | **0** | 200 |
| financial | Domain-specific | **0** | 200 |
| debit_card_specializing | Fine-tuning hurt (-4.7%) | **0** | 100 |
| superhero | Fine-tuning hurt (-3.9%) | **0** | 100 |
| thrombosis_prediction | Medical domain | **0** | 100 |

**Impact:** Without these, the model cannot improve on databases where T8 regressed.

### Top Databases in T9 (70 unique)

```
unknown:              1,574 (15.2%) ← Needs db_id assignment
works_cycles:           448 (4.3%)
public_review_platform: 358 (3.5%)
retail_world:           336 (3.3%)
movie_3:                303 (2.9%)
```

---

## 3. Quality Analysis (Subagent Reports)

### 3.1 Random Sample Quality (20 examples)

**Score: 6.5/10**

| Category | Count | Impact |
|----------|-------|--------|
| Issue-free | 15 (75%) | ✅ Good |
| Schema misalignment | 2 (10%) | ❌ Critical |
| Hint deviation | 1 (5%) | ⚠️ Medium |
| Wrong aggregation | 1 (5%) | ❌ Critical |
| Impossible schema | 1 (5%) | ❌ Critical |

**Critical Issues Found:**
1. **Example with wrong schema:** References tables (`inspections`, `businesses`) not in provided schema
2. **Aggregation error:** Uses division by constant instead of proper GROUP BY + AVG
3. **Date handling:** Ignores explicit date range hints, uses LIKE instead of BETWEEN

### 3.2 Complex Query Quality (20 examples)

**Score: 5/10** — Most Concerning

| Finding | Impact |
|---------|--------|
| **0 CTEs** despite claiming "advanced patterns" | ❌ Critical |
| **0 Window functions** | ❌ Critical |
| **55% have anti-patterns** | ❌ Critical |
| All "complexity" is subqueries | ⚠️ Limited |

**Anti-Patterns Identified:**

1. **CASE in subquery wrapper** (Example 19):
```sql
-- BAD: Teaches wrong pattern
SELECT T FROM (SELECT CASE WHEN ... THEN col END AS T ...) WHERE T IS NOT NULL

-- GOOD: Use WHERE clause
SELECT col FROM ... WHERE condition
```

2. **Self-join overkill** (Example 8):
```sql
-- BAD: 7 SELECT statements
SELECT T3.Rs_G - T4.Pf_G AS diff FROM (...) AS T3 INNER JOIN (...) AS T4 ...

-- GOOD: 1 SELECT with CASE
SELECT SUM(CASE WHEN type='A' THEN val END) - SUM(CASE WHEN type='B' THEN val END)
```

3. **MIN subquery instead of ORDER BY LIMIT** (Multiple examples):
```sql
-- BAD: Inefficient
WHERE date = (SELECT MIN(date) FROM ...)

-- GOOD: More efficient
ORDER BY date LIMIT 1
```

### 3.3 Simple Query Quality (20 no-JOIN examples)

**Score: 6.5/10**

| Category | Count |
|----------|-------|
| Truly well-formed | 14 (70%) |
| Questionable but acceptable | 2 (10%) |
| **Critical errors** | 2 (10%) |
| Edge cases | 2 (10%) |

**Critical Errors:**

1. **Wrong column in AVG** (soccer_2016):
```sql
-- Question: What are the average extra runs in second innings?
-- WRONG: Averages the filter column (always returns 2.0!)
SELECT AVG(Innings_No) FROM Extra_Runs WHERE Innings_No = 2

-- CORRECT:
SELECT AVG(Extra_Runs) FROM Extra_Runs WHERE Innings_No = 2
```

2. **Missing JOIN returns ID instead of name**:
```sql
-- Question: Find the category with highest average price
-- WRONG: Returns category_id (e.g., 42)
SELECT category_id, AVG(price) ... GROUP BY category_id

-- CORRECT: Returns actual category name
SELECT c.name, AVG(p.price) FROM products p JOIN categories c ON ...
```

---

## 4. Actionable Recommendations

### Priority 1: CRITICAL FIXES (Before Training)

| Action | Count | Effort |
|--------|-------|--------|
| Fix schema misalignment errors | ~100 | Medium |
| Fix aggregation logic errors | ~50 | Low |
| Fix wrong column references | ~30 | Low |
| Assign db_id to "unknown" entries | 1,574 | Low |

**Estimated entries needing fixes: ~200 (2% of dataset)**

### Priority 2: PATTERN REBALANCING (Essential)

| Pattern | Current | Target | Action |
|---------|---------|--------|--------|
| CTE | 36 | 600 | **Add ~564 examples** |
| WINDOW | 12 | 500 | **Add ~488 examples** |
| DISTINCT | 1,277 | 2,200 | Add ~923 examples |
| SUBQUERY | 892 | 1,550 | Add ~658 examples |
| CASE | 733 | 1,400 | Add ~667 examples |
| GROUP BY | 1,691 | 1,400 | Remove ~291 or keep |

### Priority 3: BIRD DATABASE COVERAGE (Essential)

| Database | Examples Needed | Focus |
|----------|-----------------|-------|
| california_schools | 500 | Backtick column names like \`FRPM Count (K-12)\` |
| toxicology | 200 | Chemistry/molecular patterns |
| financial | 200 | Date ranges, aggregations |
| debit_card_specializing | 100 | Transaction patterns |
| superhero | 100 | Simple patterns (avoid regression) |
| thrombosis_prediction | 100 | Medical domain |

**Total: ~1,200 targeted examples needed**

### Priority 4: QUALITY IMPROVEMENTS

1. **Simplify anti-patterns** in complex queries (~8-10 examples)
2. **Remove trivial examples** from simple queries (~4 examples)  
3. **Add progressive difficulty** - from simple JOINs to CTEs
4. **Validate all SQLite syntax** before training

---

## 5. Dataset Modification Plan

### Phase 1: Clean Current Data (1 day)
```
□ Fix 2 critical SQL errors in simple queries
□ Fix ~10 anti-patterns in complex queries  
□ Assign db_id to 1,574 "unknown" entries
□ Run SQLite syntax validation on all entries
□ Remove ~4 trivial examples
```

### Phase 2: Add Missing Patterns (2-3 days)
```
□ Generate 500+ CTE examples
□ Generate 500+ WINDOW function examples
□ Generate 500+ DISTINCT examples
□ Generate 500+ SUBQUERY examples
□ Ensure pattern distribution matches BIRD
```

### Phase 3: Add BIRD Database Examples (2-3 days)
```
□ 500 california_schools examples (backtick columns)
□ 200 toxicology examples
□ 200 financial examples
□ 300 other struggling databases
□ Validate against actual BIRD schemas
```

### Phase 4: Final Validation (1 day)
```
□ Re-run pattern distribution analysis
□ Sample 100 random entries for manual review
□ Execute all SQLs against test databases
□ Verify hint compliance
```

---

## 6. Expected Impact

### Current T9 (Without Fixes)
- **Predicted Accuracy:** 45-50% (minor improvement over T8's 43.74%)
- **Risk:** May regress on databases with no coverage
- **Anti-patterns could teach bad habits**

### Modified T9 (With Fixes)
- **Predicted Accuracy:** 52-58%
- **Pattern distribution matches BIRD**
- **Database coverage prevents regression**
- **Quality improvements aid learning**

### With Inference Improvements (Future)
- Multi-sample voting: +3-5%
- Execution retry: +1-2%
- **Total Potential:** 58-65%

---

## 7. Summary

### What's Good
- ✅ Dataset size is appropriate (10,995 examples)
- ✅ 100% have schemas (CREATE TABLE format)
- ✅ 79% have hints
- ✅ Good database diversity (70 databases)
- ✅ 75-85% of samples are quality

### What Needs Work
- ❌ Near-zero CTE and WINDOW function coverage
- ❌ Zero coverage of BIRD struggling databases
- ❌ Pattern distribution doesn't match BIRD
- ❌ ~2% critical errors that teach bad SQL
- ❌ 1,574 examples missing db_id

### Bottom Line

**T9 is NOT ready for training in current state.**

The dataset has a solid foundation but critical gaps in:
1. Advanced SQL patterns (CTEs, window functions)
2. BIRD database coverage (struggling DBs have 0 examples)
3. Quality (2-5% have significant errors)

**Recommended Action:** Execute the 4-phase modification plan before training. This will take approximately 1 week of effort but is essential to achieve the 60-67% accuracy target.

---

## Appendix: Quick Reference

### Files Analyzed
- `/Users/arnav/programming/lm/data/training/t9/train.jsonl` (10,336 examples)
- `/Users/arnav/programming/lm/data/training/t9/dev.jsonl` (659 examples)

### Specification Reference
- `/Users/arnav/programming/lm/results/qwen2.5-7b/v5/research/T9_SPECIFICATION.md`

### Analysis Artifacts
- `/tmp/t9_chunk1.json` - Random sample (20 examples)
- `/tmp/t9_chunk2.json` - Complex queries (20 examples)  
- `/tmp/t9_chunk3.json` - Simple queries (20 examples)
- `/tmp/complex_sql_evaluation_report.md` - Full complex query analysis
- `/tmp/simple_query_analysis_report.md` - Full simple query analysis

# T11.1 Evaluation Summary

**Generated:** 2026-04-05T07:23:20  
**Predictions:** `/Users/arnav/programming/lm/runs/t11_1_baseline_3090/without-sampling/predictions/predictions_t11_1.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 537/1534 | **35.01%** |
| Exact Match | 141/1534 | 9.19% |
| Exec Failures | 321 | 20.93% |
| Wrong Results | 676 | 44.07% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 384 | 925 | 41.51% | 167 | 374 |
| moderate | 113 | 464 | 24.35% | 110 | 241 |
| challenging | 40 | 145 | 27.59% | 44 | 61 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| superhero | 78 | 129 | 60.47% |
| student_club | 87 | 158 | 55.06% |
| codebase_community | 95 | 186 | 51.08% |
| european_football_2 | 53 | 129 | 41.09% |
| toxicology | 49 | 145 | 33.79% |
| thrombosis_prediction | 52 | 163 | 31.9% |
| card_games | 57 | 191 | 29.84% |
| financial | 19 | 106 | 17.92% |
| formula_1 | 31 | 174 | 17.82% |
| debit_card_specializing | 8 | 64 | 12.5% |
| california_schools | 8 | 89 | 8.99% |

## Error Categories

| Category | Count |
|----------|-------|
| column_error | 294 |
| other_error | 11 |
| aggregate_error | 7 |
| table_error | 4 |
| syntax_error | 3 |
| ambiguous_column | 2 |

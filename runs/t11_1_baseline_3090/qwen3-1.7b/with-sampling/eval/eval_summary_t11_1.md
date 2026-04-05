# T11.1 Evaluation Summary

**Generated:** 2026-04-05T07:08:39  
**Predictions:** `/Users/arnav/programming/lm/runs/t11_1_baseline_3090/with-sampling/predictions/predictions_t11_1.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 501/1534 | **32.66%** |
| Exact Match | 135/1534 | 8.8% |
| Exec Failures | 369 | 24.05% |
| Wrong Results | 664 | 43.29% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 376 | 925 | 40.65% | 178 | 371 |
| moderate | 92 | 464 | 19.83% | 135 | 237 |
| challenging | 33 | 145 | 22.76% | 56 | 56 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| superhero | 73 | 129 | 56.59% |
| codebase_community | 92 | 186 | 49.46% |
| student_club | 77 | 158 | 48.73% |
| european_football_2 | 46 | 129 | 35.66% |
| toxicology | 43 | 145 | 29.66% |
| card_games | 56 | 191 | 29.32% |
| thrombosis_prediction | 41 | 163 | 25.15% |
| financial | 22 | 106 | 20.75% |
| formula_1 | 31 | 174 | 17.82% |
| california_schools | 12 | 89 | 13.48% |
| debit_card_specializing | 8 | 64 | 12.5% |

## Error Categories

| Category | Count |
|----------|-------|
| column_error | 340 |
| syntax_error | 7 |
| table_error | 7 |
| aggregate_error | 6 |
| other_error | 6 |
| ambiguous_column | 3 |

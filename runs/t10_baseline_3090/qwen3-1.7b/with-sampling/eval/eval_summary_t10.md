# T10 Evaluation Summary

**Generated:** 2026-04-05T04:49:52  
**Predictions:** `./runs/t10_baseline_3090/qwen3-1.7b/with-sampling/predictions/predictions_t10.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 541/1534 | **35.27%** |
| Exact Match | 154/1534 | 10.04% |
| Exec Failures | 369 | 24.05% |
| Wrong Results | 624 | 40.68% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 396 | 925 | 42.81% | 172 | 357 |
| moderate | 115 | 464 | 24.78% | 143 | 206 |
| challenging | 30 | 145 | 20.69% | 54 | 61 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| superhero | 70 | 129 | 54.26% |
| codebase_community | 95 | 186 | 51.08% |
| student_club | 80 | 158 | 50.63% |
| european_football_2 | 58 | 129 | 44.96% |
| debit_card_specializing | 23 | 64 | 35.94% |
| toxicology | 51 | 145 | 35.17% |
| card_games | 65 | 191 | 34.03% |
| thrombosis_prediction | 38 | 163 | 23.31% |
| formula_1 | 34 | 174 | 19.54% |
| financial | 18 | 106 | 16.98% |
| california_schools | 9 | 89 | 10.11% |

## Error Categories

| Category | Count |
|----------|-------|
| column_error | 331 |
| syntax_error | 14 |
| other_error | 9 |
| table_error | 8 |
| aggregate_error | 4 |
| ambiguous_column | 3 |

# T12 Evaluation Summary

**Generated:** 2026-04-07T17:43:17  
**Predictions:** `runs/t12_base_2b/predictions/predictions_t12.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 370/1534 | **24.12%** |
| Exact Match | 25/1534 | 1.63% |
| Exec Failures | 552 | 35.98% |
| Wrong Results | 610 | 39.77% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 292 | 925 | 31.57% | 249 | 384 |
| moderate | 64 | 464 | 13.79% | 224 | 175 |
| challenging | 14 | 145 | 9.66% | 79 | 51 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| codebase_community | 73 | 186 | 39.25% |
| student_club | 52 | 158 | 32.91% |
| superhero | 41 | 129 | 31.78% |
| european_football_2 | 37 | 129 | 28.68% |
| toxicology | 41 | 145 | 28.28% |
| card_games | 53 | 191 | 27.75% |
| debit_card_specializing | 13 | 64 | 20.31% |
| formula_1 | 30 | 174 | 17.24% |
| thrombosis_prediction | 16 | 163 | 9.82% |
| financial | 10 | 106 | 9.43% |
| california_schools | 4 | 89 | 4.49% |

## Error Categories

| Category | Count |
|----------|-------|
| column_error | 384 |
| syntax_error | 77 |
| other_error | 47 |
| ambiguous_column | 18 |
| aggregate_error | 17 |
| table_error | 9 |

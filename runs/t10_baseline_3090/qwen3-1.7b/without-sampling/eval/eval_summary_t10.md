# T10 Evaluation Summary

**Generated:** 2026-04-05T02:18:14  
**Predictions:** `./runs/t10_baseline_3090/predictions/predictions_t10.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 584/1534 | **38.07%** |
| Exact Match | 189/1534 | 12.32% |
| Exec Failures | 309 | 20.14% |
| Wrong Results | 641 | 41.79% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 431 | 925 | 46.59% | 147 | 347 |
| moderate | 119 | 464 | 25.65% | 113 | 232 |
| challenging | 34 | 145 | 23.45% | 49 | 62 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| codebase_community | 111 | 186 | 59.68% |
| superhero | 70 | 129 | 54.26% |
| student_club | 84 | 158 | 53.16% |
| european_football_2 | 66 | 129 | 51.16% |
| toxicology | 55 | 145 | 37.93% |
| card_games | 68 | 191 | 35.6% |
| debit_card_specializing | 20 | 64 | 31.25% |
| thrombosis_prediction | 44 | 163 | 26.99% |
| formula_1 | 36 | 174 | 20.69% |
| financial | 19 | 106 | 17.92% |
| california_schools | 11 | 89 | 12.36% |

## Error Categories

| Category | Count |
|----------|-------|
| column_error | 287 |
| other_error | 11 |
| syntax_error | 5 |
| table_error | 3 |
| aggregate_error | 3 |

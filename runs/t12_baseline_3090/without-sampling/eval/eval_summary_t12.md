# T12 Evaluation Summary

**Generated:** 2026-04-06T19:31:02  
**Predictions:** `./runs/t12_baseline_3090/predictions/predictions_t12.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 628/1534 | **40.94%** |
| Exact Match | 207/1534 | 13.49% |
| Exec Failures | 192 | 12.52% |
| Wrong Results | 714 | 46.54% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 468 | 925 | 50.59% | 84 | 373 |
| moderate | 131 | 464 | 28.23% | 78 | 255 |
| challenging | 29 | 145 | 20.0% | 30 | 86 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| student_club | 104 | 158 | 65.82% |
| codebase_community | 107 | 186 | 57.53% |
| superhero | 73 | 129 | 56.59% |
| european_football_2 | 72 | 129 | 55.81% |
| debit_card_specializing | 25 | 64 | 39.06% |
| formula_1 | 61 | 174 | 35.06% |
| card_games | 65 | 191 | 34.03% |
| toxicology | 46 | 145 | 31.72% |
| thrombosis_prediction | 42 | 163 | 25.77% |
| financial | 20 | 106 | 18.87% |
| california_schools | 13 | 89 | 14.61% |

## Error Categories

| Category | Count |
|----------|-------|
| column_error | 172 |
| syntax_error | 10 |
| other_error | 7 |
| aggregate_error | 2 |
| table_error | 1 |

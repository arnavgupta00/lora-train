# T10 Evaluation Summary

**Generated:** 2026-04-05T06:28:21  
**Predictions:** `/Users/arnav/programming/lm/runs/t10_baseline_3090/qwen3-1.7b/without-lora/predictions/predictions_t10.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 36/1534 | **2.35%** |
| Exact Match | 21/1534 | 1.37% |
| Exec Failures | 1487 | 96.94% |
| Wrong Results | 11 | 0.72% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 36 | 925 | 3.89% | 878 | 11 |
| moderate | 0 | 464 | 0.0% | 464 | 0 |
| challenging | 0 | 145 | 0.0% | 145 | 0 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| codebase_community | 11 | 186 | 5.91% |
| formula_1 | 7 | 174 | 4.02% |
| card_games | 7 | 191 | 3.66% |
| superhero | 4 | 129 | 3.1% |
| student_club | 4 | 158 | 2.53% |
| european_football_2 | 2 | 129 | 1.55% |
| toxicology | 1 | 145 | 0.69% |
| california_schools | 0 | 89 | 0.0% |
| debit_card_specializing | 0 | 64 | 0.0% |
| financial | 0 | 106 | 0.0% |
| thrombosis_prediction | 0 | 163 | 0.0% |

## Error Categories

| Category | Count |
|----------|-------|
| syntax_error | 1464 |
| other_error | 10 |
| column_error | 9 |
| empty_prediction | 4 |

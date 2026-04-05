# T11.1 Evaluation Summary

**Generated:** 2026-04-05T07:33:24  
**Predictions:** `/Users/arnav/programming/lm/runs/t11_1_baseline_3090/qwen3-1.7b/without-lora/predictions/predictions_t11_1.jsonl`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | 36/1534 | **2.35%** |
| Exact Match | 18/1534 | 1.17% |
| Exec Failures | 1488 | 97.0% |
| Wrong Results | 10 | 0.65% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
| simple | 36 | 925 | 3.89% | 880 | 9 |
| moderate | 0 | 464 | 0.0% | 463 | 1 |
| challenging | 0 | 145 | 0.0% | 145 | 0 |

## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| codebase_community | 11 | 186 | 5.91% |
| superhero | 7 | 129 | 5.43% |
| student_club | 5 | 158 | 3.16% |
| formula_1 | 4 | 174 | 2.3% |
| card_games | 4 | 191 | 2.09% |
| toxicology | 3 | 145 | 2.07% |
| european_football_2 | 2 | 129 | 1.55% |
| california_schools | 0 | 89 | 0.0% |
| debit_card_specializing | 0 | 64 | 0.0% |
| financial | 0 | 106 | 0.0% |
| thrombosis_prediction | 0 | 163 | 0.0% |

## Error Categories

| Category | Count |
|----------|-------|
| syntax_error | 1477 |
| other_error | 4 |
| column_error | 4 |
| empty_prediction | 3 |

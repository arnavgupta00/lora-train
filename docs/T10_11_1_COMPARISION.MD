# BIRD Evaluation Runs - Comparative Analysis

**Generated:** 2026-04-05  
**Model:** Qwen/Qwen3-1.7B  
**Datasets:** T10 (full schema with hints), T11_1 (compact schema)  
**Evaluation Set:** BIRD Dev (1534 examples)

---

## 1. Run Inventory

| run_name | dataset | model | lora_used | decoding_mode | temperature | top_p | top_k | max_new_tokens | prompts_file | predictions_file | timestamp |
|----------|---------|-------|-----------|---------------|-------------|-------|-------|----------------|--------------|------------------|-----------|
| T10_with_sampling | t10 | Qwen3-1.7B | Yes | sampling | 0.6 | 0.95 | 20 | 256 | bird_dev_t10.jsonl | predictions_t10.jsonl | 2026-04-05T04:49:52 |
| T10_without_sampling | t10 | Qwen3-1.7B | Yes | greedy | 0.0 | 1.0 | 50 | 256 | bird_dev_t10.jsonl | predictions_t10.jsonl | 2026-04-05T02:18:14 |
| T10_without_lora | t10 | Qwen3-1.7B | No | greedy | 0.0 | 1.0 | 50 | 256 | bird_dev_t10.jsonl | predictions_t10.jsonl | 2026-04-05T06:28:21 |
| T11_1_with_sampling | t11_1 | Qwen3-1.7B | Yes | sampling | 0.3 | 0.95 | 20 | 256 | bird_dev_t11_1.jsonl | predictions_t11_1.jsonl | 2026-04-05T07:08:39 |
| T11_1_without_sampling | t11_1 | Qwen3-1.7B | Yes | greedy | 0.0 | 1.0 | 50 | 256 | bird_dev_t11_1.jsonl | predictions_t11_1.jsonl | 2026-04-05T07:23:20 |
| T11_1_without_lora | t11_1 | Qwen3-1.7B | No | greedy | 0.0 | 1.0 | 50 | 256 | bird_dev_t11_1.jsonl | predictions_t11_1.jsonl | 2026-04-05T07:33:24 |

**Note on T10 vs T11_1:**
- T10: Full schema with column hints
- T11_1: Compact schema (avg 53.7% reduction), 90% success rate, 10% fallback to full schema

---

## 2. Headline Metrics Comparison

| run_name | EX (%) | EM (%) | exec_fail_count | exec_fail_rate (%) | wrong_result_count | wrong_result_rate (%) | exact_match_count | execution_correct | total_examples |
|----------|--------|--------|-----------------|---------------------|--------------------|-----------------------|-------------------|-------------------|----------------|
| T10_with_sampling | 35.27 | 10.04 | 369 | 24.05 | 624 | 40.68 | 154 | 541 | 1534 |
| T10_without_sampling | 38.07 | 12.32 | 309 | 20.14 | 641 | 41.79 | 189 | 584 | 1534 |
| T10_without_lora | 2.35 | 1.37 | 1487 | 96.94 | 11 | 0.72 | 21 | 36 | 1534 |
| T11_1_with_sampling | 32.66 | 8.80 | 369 | 24.05 | 664 | 43.29 | 135 | 501 | 1534 |
| T11_1_without_sampling | 35.01 | 9.19 | 321 | 20.93 | 676 | 44.07 | 141 | 537 | 1534 |
| T11_1_without_lora | 2.35 | 1.17 | 1488 | 97.00 | 10 | 0.65 | 18 | 36 | 1534 |

---

## 3. Difficulty Breakdown

| run_name | simple_acc (%) | simple_correct/total | moderate_acc (%) | moderate_correct/total | challenging_acc (%) | challenging_correct/total |
|----------|----------------|----------------------|------------------|------------------------|---------------------|---------------------------|
| T10_with_sampling | 42.81 | 396/925 | 24.78 | 115/464 | 20.69 | 30/145 |
| T10_without_sampling | 46.59 | 431/925 | 25.65 | 119/464 | 23.45 | 34/145 |
| T10_without_lora | 3.89 | 36/925 | 0.00 | 0/464 | 0.00 | 0/145 |
| T11_1_with_sampling | 40.65 | 376/925 | 19.83 | 92/464 | 22.76 | 33/145 |
| T11_1_without_sampling | 41.51 | 384/925 | 24.35 | 113/464 | 27.59 | 40/145 |
| T11_1_without_lora | 3.89 | 36/925 | 0.00 | 0/464 | 0.00 | 0/145 |

### Detailed Per-Difficulty Breakdown

**Simple (925 examples)**

| run_name | correct | exec_failed | wrong_result | accuracy (%) |
|----------|---------|-------------|--------------|--------------|
| T10_with_sampling | 396 | 172 | 357 | 42.81 |
| T10_without_sampling | 431 | 147 | 347 | 46.59 |
| T10_without_lora | 36 | 878 | 11 | 3.89 |
| T11_1_with_sampling | 376 | 178 | 371 | 40.65 |
| T11_1_without_sampling | 384 | 167 | 374 | 41.51 |
| T11_1_without_lora | 36 | 880 | 9 | 3.89 |

**Moderate (464 examples)**

| run_name | correct | exec_failed | wrong_result | accuracy (%) |
|----------|---------|-------------|--------------|--------------|
| T10_with_sampling | 115 | 143 | 206 | 24.78 |
| T10_without_sampling | 119 | 113 | 232 | 25.65 |
| T10_without_lora | 0 | 464 | 0 | 0.00 |
| T11_1_with_sampling | 92 | 135 | 237 | 19.83 |
| T11_1_without_sampling | 113 | 110 | 241 | 24.35 |
| T11_1_without_lora | 0 | 463 | 1 | 0.00 |

**Challenging (145 examples)**

| run_name | correct | exec_failed | wrong_result | accuracy (%) |
|----------|---------|-------------|--------------|--------------|
| T10_with_sampling | 30 | 54 | 61 | 20.69 |
| T10_without_sampling | 34 | 49 | 62 | 23.45 |
| T10_without_lora | 0 | 145 | 0 | 0.00 |
| T11_1_with_sampling | 33 | 56 | 56 | 22.76 |
| T11_1_without_sampling | 40 | 44 | 61 | 27.59 |
| T11_1_without_lora | 0 | 145 | 0 | 0.00 |

---

## 4. Error Category Breakdown

| run_name | column_error | syntax_error | table_error | aggregate_error | other_error | ambiguous_column | empty_prediction |
|----------|--------------|--------------|-------------|-----------------|-------------|------------------|------------------|
| T10_with_sampling | 331 | 14 | 8 | 4 | 9 | 3 | 0 |
| T10_without_sampling | 287 | 5 | 3 | 3 | 11 | 0 | 0 |
| T10_without_lora | 9 | 1464 | 0 | 0 | 10 | 0 | 4 |
| T11_1_with_sampling | 340 | 7 | 7 | 6 | 6 | 3 | 0 |
| T11_1_without_sampling | 294 | 3 | 4 | 7 | 11 | 2 | 0 |
| T11_1_without_lora | 4 | 1477 | 0 | 0 | 4 | 0 | 3 |

**Error Category Distribution (LoRA runs only)**

| error_category | T10_sampling | T10_greedy | T11_1_sampling | T11_1_greedy |
|----------------|--------------|------------|----------------|--------------|
| column_error | 331 (89.7%) | 287 (92.9%) | 340 (92.1%) | 294 (91.6%) |
| syntax_error | 14 (3.8%) | 5 (1.6%) | 7 (1.9%) | 3 (0.9%) |
| table_error | 8 (2.2%) | 3 (1.0%) | 7 (1.9%) | 4 (1.2%) |
| aggregate_error | 4 (1.1%) | 3 (1.0%) | 6 (1.6%) | 7 (2.2%) |
| other_error | 9 (2.4%) | 11 (3.6%) | 6 (1.6%) | 11 (3.4%) |
| ambiguous_column | 3 (0.8%) | 0 (0.0%) | 3 (0.8%) | 2 (0.6%) |

---

## 5. Per-Database Comparison

### Table Format

| db_id | T10_sampling | T10_greedy | T10_no_lora | T11_1_sampling | T11_1_greedy | T11_1_no_lora |
|-------|--------------|------------|-------------|----------------|--------------|---------------|
| california_schools (89) | 10.11% (9) | 12.36% (11) | 0.00% (0) | 13.48% (12) | 8.99% (8) | 0.00% (0) |
| card_games (191) | 34.03% (65) | 35.60% (68) | 3.66% (7) | 29.32% (56) | 29.84% (57) | 2.09% (4) |
| codebase_community (186) | 51.08% (95) | 59.68% (111) | 5.91% (11) | 49.46% (92) | 51.08% (95) | 5.91% (11) |
| debit_card_specializing (64) | 35.94% (23) | 31.25% (20) | 0.00% (0) | 12.50% (8) | 12.50% (8) | 0.00% (0) |
| european_football_2 (129) | 44.96% (58) | 51.16% (66) | 1.55% (2) | 35.66% (46) | 41.09% (53) | 1.55% (2) |
| financial (106) | 16.98% (18) | 17.92% (19) | 0.00% (0) | 20.75% (22) | 17.92% (19) | 0.00% (0) |
| formula_1 (174) | 19.54% (34) | 20.69% (36) | 4.02% (7) | 17.82% (31) | 17.82% (31) | 2.30% (4) |
| student_club (158) | 50.63% (80) | 53.16% (84) | 2.53% (4) | 48.73% (77) | 55.06% (87) | 3.16% (5) |
| superhero (129) | 54.26% (70) | 54.26% (70) | 3.10% (4) | 56.59% (73) | 60.47% (78) | 5.43% (7) |
| thrombosis_prediction (163) | 23.31% (38) | 26.99% (44) | 0.00% (0) | 25.15% (41) | 31.90% (52) | 0.00% (0) |
| toxicology (145) | 35.17% (51) | 37.93% (55) | 0.69% (1) | 29.66% (43) | 33.79% (49) | 2.07% (3) |

### Detailed Per-Database Results (LoRA runs)

**T10_with_sampling**

| db_id | total | correct | exec_failed | wrong_result | accuracy (%) |
|-------|-------|---------|-------------|--------------|--------------|
| california_schools | 89 | 9 | 25 | 55 | 10.11 |
| card_games | 191 | 65 | 40 | 86 | 34.03 |
| codebase_community | 186 | 95 | 29 | 62 | 51.08 |
| debit_card_specializing | 64 | 23 | 14 | 27 | 35.94 |
| european_football_2 | 129 | 58 | 21 | 50 | 44.96 |
| financial | 106 | 18 | 25 | 63 | 16.98 |
| formula_1 | 174 | 34 | 50 | 90 | 19.54 |
| student_club | 158 | 80 | 31 | 47 | 50.63 |
| superhero | 129 | 70 | 32 | 27 | 54.26 |
| thrombosis_prediction | 163 | 38 | 72 | 53 | 23.31 |
| toxicology | 145 | 51 | 30 | 64 | 35.17 |

**T10_without_sampling**

| db_id | total | correct | exec_failed | wrong_result | accuracy (%) |
|-------|-------|---------|-------------|--------------|--------------|
| california_schools | 89 | 11 | 22 | 56 | 12.36 |
| card_games | 191 | 68 | 36 | 87 | 35.60 |
| codebase_community | 186 | 111 | 25 | 50 | 59.68 |
| debit_card_specializing | 64 | 20 | 10 | 34 | 31.25 |
| european_football_2 | 129 | 66 | 26 | 37 | 51.16 |
| financial | 106 | 19 | 18 | 69 | 17.92 |
| formula_1 | 174 | 36 | 46 | 92 | 20.69 |
| student_club | 158 | 84 | 26 | 48 | 53.16 |
| superhero | 129 | 70 | 30 | 29 | 54.26 |
| thrombosis_prediction | 163 | 44 | 42 | 77 | 26.99 |
| toxicology | 145 | 55 | 28 | 62 | 37.93 |

**T11_1_with_sampling**

| db_id | total | correct | exec_failed | wrong_result | accuracy (%) |
|-------|-------|---------|-------------|--------------|--------------|
| california_schools | 89 | 12 | 28 | 49 | 13.48 |
| card_games | 191 | 56 | 41 | 94 | 29.32 |
| codebase_community | 186 | 92 | 42 | 52 | 49.46 |
| debit_card_specializing | 64 | 8 | 18 | 38 | 12.50 |
| european_football_2 | 129 | 46 | 28 | 55 | 35.66 |
| financial | 106 | 22 | 28 | 56 | 20.75 |
| formula_1 | 174 | 31 | 46 | 97 | 17.82 |
| student_club | 158 | 77 | 31 | 50 | 48.73 |
| superhero | 129 | 73 | 33 | 23 | 56.59 |
| thrombosis_prediction | 163 | 41 | 40 | 82 | 25.15 |
| toxicology | 145 | 43 | 34 | 68 | 29.66 |

**T11_1_without_sampling**

| db_id | total | correct | exec_failed | wrong_result | accuracy (%) |
|-------|-------|---------|-------------|--------------|--------------|
| california_schools | 89 | 8 | 33 | 48 | 8.99 |
| card_games | 191 | 57 | 35 | 99 | 29.84 |
| codebase_community | 186 | 95 | 40 | 51 | 51.08 |
| debit_card_specializing | 64 | 8 | 15 | 41 | 12.50 |
| european_football_2 | 129 | 53 | 21 | 55 | 41.09 |
| financial | 106 | 19 | 19 | 68 | 17.92 |
| formula_1 | 174 | 31 | 45 | 98 | 17.82 |
| student_club | 158 | 87 | 25 | 46 | 55.06 |
| superhero | 129 | 78 | 29 | 22 | 60.47 |
| thrombosis_prediction | 163 | 52 | 27 | 84 | 31.90 |
| toxicology | 145 | 49 | 32 | 64 | 33.79 |

---

## 6. Decoding Comparison (Greedy vs Sampling)

### T10 Dataset: Greedy vs Sampling

| metric | T10_greedy | T10_sampling | delta |
|--------|------------|--------------|-------|
| EX (%) | 38.07 | 35.27 | -2.80 |
| EM (%) | 12.32 | 10.04 | -2.28 |
| exec_fail_count | 309 | 369 | +60 |
| exec_fail_rate (%) | 20.14 | 24.05 | +3.91 |
| wrong_result_count | 641 | 624 | -17 |
| execution_correct | 584 | 541 | -43 |

**Difficulty breakdown:**

| difficulty | T10_greedy | T10_sampling | delta |
|------------|------------|--------------|-------|
| simple | 46.59% | 42.81% | -3.78% |
| moderate | 25.65% | 24.78% | -0.87% |
| challenging | 23.45% | 20.69% | -2.76% |

### T11_1 Dataset: Greedy vs Sampling

| metric | T11_1_greedy | T11_1_sampling | delta |
|--------|--------------|----------------|-------|
| EX (%) | 35.01 | 32.66 | -2.35 |
| EM (%) | 9.19 | 8.80 | -0.39 |
| exec_fail_count | 321 | 369 | +48 |
| exec_fail_rate (%) | 20.93 | 24.05 | +3.12 |
| wrong_result_count | 676 | 664 | -12 |
| execution_correct | 537 | 501 | -36 |

**Difficulty breakdown:**

| difficulty | T11_1_greedy | T11_1_sampling | delta |
|------------|--------------|----------------|-------|
| simple | 41.51% | 40.65% | -0.86% |
| moderate | 24.35% | 19.83% | -4.52% |
| challenging | 27.59% | 22.76% | -4.83% |

---

## 7. LoRA vs No-LoRA Comparison

### T10 Dataset: LoRA vs No-LoRA

| metric | T10_greedy_lora | T10_greedy_no_lora | delta |
|--------|-----------------|---------------------|-------|
| EX (%) | 38.07 | 2.35 | -35.72 |
| EM (%) | 12.32 | 1.37 | -10.95 |
| exec_fail_count | 309 | 1487 | +1178 |
| exec_fail_rate (%) | 20.14 | 96.94 | +76.80 |
| wrong_result_count | 641 | 11 | -630 |
| execution_correct | 584 | 36 | -548 |

**Difficulty breakdown:**

| difficulty | with_lora | without_lora | delta |
|------------|-----------|--------------|-------|
| simple | 46.59% | 3.89% | -42.70% |
| moderate | 25.65% | 0.00% | -25.65% |
| challenging | 23.45% | 0.00% | -23.45% |

**Error category shift:**

| error_type | with_lora | without_lora | change |
|------------|-----------|--------------|--------|
| column_error | 287 (92.9%) | 9 (0.6%) | -278 |
| syntax_error | 5 (1.6%) | 1464 (98.5%) | +1459 |
| other_error | 11 (3.6%) | 10 (0.7%) | -1 |
| table_error | 3 (1.0%) | 0 (0.0%) | -3 |
| aggregate_error | 3 (1.0%) | 0 (0.0%) | -3 |
| empty_prediction | 0 (0.0%) | 4 (0.3%) | +4 |

### T11_1 Dataset: LoRA vs No-LoRA

| metric | T11_1_greedy_lora | T11_1_greedy_no_lora | delta |
|--------|-------------------|----------------------|-------|
| EX (%) | 35.01 | 2.35 | -32.66 |
| EM (%) | 9.19 | 1.17 | -8.02 |
| exec_fail_count | 321 | 1488 | +1167 |
| exec_fail_rate (%) | 20.93 | 97.00 | +76.07 |
| wrong_result_count | 676 | 10 | -666 |
| execution_correct | 537 | 36 | -501 |

**Difficulty breakdown:**

| difficulty | with_lora | without_lora | delta |
|------------|-----------|--------------|-------|
| simple | 41.51% | 3.89% | -37.62% |
| moderate | 24.35% | 0.00% | -24.35% |
| challenging | 27.59% | 0.00% | -27.59% |

**Error category shift:**

| error_type | with_lora | without_lora | change |
|------------|-----------|--------------|--------|
| column_error | 294 (91.6%) | 4 (0.3%) | -290 |
| syntax_error | 3 (0.9%) | 1477 (99.3%) | +1474 |
| other_error | 11 (3.4%) | 4 (0.3%) | -7 |
| table_error | 4 (1.2%) | 0 (0.0%) | -4 |
| aggregate_error | 7 (2.2%) | 0 (0.0%) | -7 |
| ambiguous_column | 2 (0.6%) | 0 (0.0%) | -2 |
| empty_prediction | 0 (0.0%) | 3 (0.2%) | +3 |

---

## 8. Prompt Shape Metadata

### T10 (Full Schema with Hints)

- **Prompts file:** `bird_dev_t10.jsonl`
- **Total examples:** 1534
- **With hints:** 1386 (90.4%)
- **No hints:** 148 (9.6%)
- **Schema cache hits:** 1523
- **SQL audit:** Passed
- **System prompt hash:** `37082362804b17cd`

### T11_1 (Compact Schema)

- **Source:** T10 prompts (transformed)
- **Total examples:** 1534
- **Success rate:** 90.0% (1349 examples)
- **Fallback rate:** 10.0% (153 examples - reverted to full schema)
- **Widened rate:** 2.1% (32 examples - schema expanded after initial compaction)
- **Average reduction (all):** 53.7%
- **Average reduction (successful):** 59.7%
- **System prompt hash:** `37082362804b17cd` (matches T10)

**Reduction distribution:**

| reduction_range | count |
|-----------------|-------|
| 0-20% | 220 |
| 20-40% | 245 |
| 40-60% | 310 |
| 60-80% | 490 |
| 80-100% | 269 |

**Fallback reasons:**

| reason | count |
|--------|-------|
| over_compacted | 145 |
| no_tables_matched | 8 |

**Per-database schema reduction (T11_1):**

| db_id | total | success | widened | fallback | avg_reduction (%) |
|-------|-------|---------|---------|----------|-------------------|
| california_schools | 89 | 80 | 2 | 7 | 71.5 |
| card_games | 191 | 147 | 3 | 41 | 63.3 |
| codebase_community | 186 | 139 | 11 | 36 | 51.5 |
| debit_card_specializing | 64 | 59 | 0 | 5 | 66.0 |
| european_football_2 | 129 | 120 | 0 | 9 | 35.5 |
| financial | 106 | 96 | 5 | 5 | 57.1 |
| formula_1 | 174 | 151 | 0 | 23 | 53.0 |
| student_club | 158 | 132 | 3 | 23 | 43.4 |
| superhero | 129 | 127 | 0 | 2 | 39.6 |
| thrombosis_prediction | 163 | 153 | 8 | 2 | 78.2 |
| toxicology | 145 | 145 | 0 | 0 | 38.5 |

---

## 9. Example Failures Appendix

### T10_with_sampling - Sample Execution Errors

**Question 3 (california_schools, simple):**
- **Error:** `near "FROM": syntax error`
- **Predicted SQL:** `SELECT T2.MailingStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.\`Free Meal Count (K-12)\` = ( SELECT MAX(\`Free Meal Count (K-12)\` FROM frpm ) )`
- **Error type:** syntax_error

**Question 10 (california_schools, simple):**
- **Error:** `no such column: T2.FRPM_COUNT`
- **Predicted SQL:** `SELECT T2.FRPM_COUNT FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE T1.rtype = 'Reading' ORDER BY T1.AvgScrRead DESC LIMIT 1`
- **Error type:** column_error

**Question 12 (california_schools, moderate):**
- **Error:** `no such column: T2.Free Meal Count (Ages 5-17)`
- **Predicted SQL:** `SELECT T2.\`Free Meal Count (Ages 5-17)\` / T2.\`Enrollment (Ages 5-17)\` FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.NumGE1500 / T2.NumTstTakr > 0.3 ORDER BY T2.\`Free Meal Count (Ages 5-17)\` / T2.\`Enrollment (Ages 5-17)\` DESC LIMIT 1`
- **Error type:** column_error

### T10_with_sampling - Sample Wrong Results

**Question 0 (california_schools, simple):**
- **Predicted SQL:** `SELECT MAX(\`Percent (%) Eligible Free (K-12)\`) FROM frpm WHERE \`County Name\` = 'Alameda County'`
- **Gold SQL:** `SELECT \`Free Meal Count (K-12)\` / \`Enrollment (K-12)\` FROM frpm WHERE \`County Name\` = 'Alameda' ORDER BY (CAST(\`Free Meal Count (K-12)\` AS REAL) / \`Enrollment (K-12)\`) DESC LIMIT 1`

**Question 1 (california_schools, moderate):**
- **Predicted SQL:** `SELECT CAST(\`Free Meal Count (Ages 5-17)\` AS REAL) / \`Enrollment (Ages 5-17)\` FROM frpm ORDER BY CAST(\`Free Meal Count (Ages 5-17)\` AS REAL) / \`Enrollment (Ages 5-17)\` ASC LIMIT 3`
- **Gold SQL:** `SELECT \`Free Meal Count (Ages 5-17)\` / \`Enrollment (Ages 5-17)\` FROM frpm WHERE \`Educational Option Type\` = 'Continuation School' AND \`Free Meal Count (Ages 5-17)\` / \`Enrollment (Ages 5-17)\` IS NOT NULL ORDER BY \`Free Meal Count (Ages 5-17)\` / \`Enrollment (Ages 5-17)\` ASC LIMIT 3`

### T10_without_lora - Sample Execution Errors

**Question 0 (california_schools, simple):**
- **Error:** `near "<": syntax error`
- **Predicted SQL:** `<think> Okay, let's tackle this question. The user wants the highest eligible free rate for K-12 students in Alameda County. First, I need to figure out which tables to use. The relevant tables here are 'frpm' and 'schools'. The 'schools' table has a 'County' column, and the 'frpm' table has 'County`
- **Error type:** syntax_error
- **Note:** Model output included reasoning tokens instead of SQL

### T11_1_with_sampling - Sample Execution Errors

**Question 10 (california_schools, simple):**
- **Error:** `no such column: T2.FRPMCount`
- **Predicted SQL:** `SELECT T2.FRPMCount FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.NCESSchool = ( SELECT T1.NCESSchool FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T2.\`Percent (%) Eligible FRPM (Ages 5-17)\` DESC LIMIT 1 )`
- **Error type:** column_error

**Question 17 (california_schools, simple):**
- **Error:** `no such column: s2.Writing`
- **Predicted SQL:** `SELECT s."School", AVG(s2."Writing") AS avg_score, s2."Charter" FROM "schools" s LEFT JOIN "frpm" s2 ON s."CDSCode" = s2."CDSCode" WHERE s2."Writing" > 499 GROUP BY s."School" ORDER BY avg_score DESC, s2."Charter" DESC`
- **Error type:** column_error

### T11_1_with_sampling - Sample Wrong Results

**Question 0 (california_schools, simple):**
- **Predicted SQL:** `SELECT CAST(\`Free Meal Count (K-12)\` AS REAL) * 100 / \`Enrollment (K-12)\` FROM frpm WHERE \`County Name\` = 'Alameda County' AND \`District Name\` = 'Alameda Union High School District'`
- **Gold SQL:** `SELECT \`Free Meal Count (K-12)\` / \`Enrollment (K-12)\` FROM frpm WHERE \`County Name\` = 'Alameda' ORDER BY (CAST(\`Free Meal Count (K-12)\` AS REAL) / \`Enrollment (K-12)\`) DESC LIMIT 1`

**Question 1 (california_schools, moderate):**
- **Predicted SQL:** `SELECT CAST(\`Free Meal Count (Ages 5-17)\` AS REAL) / \`Enrollment (Ages 5-17)\` FROM frpm WHERE \`School Type\` = 'Continuation School' ORDER BY CAST(\`Free Meal Count (Ages 5-17)\` AS REAL) / \`Enrollment (Ages 5-17)\` ASC LIMIT 3`
- **Gold SQL:** `SELECT \`Free Meal Count (Ages 5-17)\` / \`Enrollment (Ages 5-17)\` FROM frpm WHERE \`Educational Option Type\` = 'Continuation School' AND \`Free Meal Count (Ages 5-17)\` / \`Enrollment (Ages 5-17)\` IS NOT NULL ORDER BY \`Free Meal Count (Ages 5-17)\` / \`Enrollment (Ages 5-17)\` ASC LIMIT 3`

---

## Summary Tables

### Cross-Dataset Comparison (LoRA Greedy Runs)

| metric | T10_greedy | T11_1_greedy | delta |
|--------|------------|--------------|-------|
| EX (%) | 38.07 | 35.01 | -3.06 |
| EM (%) | 12.32 | 9.19 | -3.13 |
| exec_fail_rate (%) | 20.14 | 20.93 | +0.79 |
| wrong_result_rate (%) | 41.79 | 44.07 | +2.28 |
| simple accuracy (%) | 46.59 | 41.51 | -5.08 |
| moderate accuracy (%) | 25.65 | 24.35 | -1.30 |
| challenging accuracy (%) | 23.45 | 27.59 | +4.14 |

### Best Performing Configuration

| dataset | configuration | EX (%) | EM (%) | exec_fail_rate (%) |
|---------|---------------|--------|--------|--------------------|
| T10 | greedy + LoRA | 38.07 | 12.32 | 20.14 |
| T11_1 | greedy + LoRA | 35.01 | 9.19 | 20.93 |

---

**End of Report**

# BIRD Evaluation Data

This folder contains the official BIRD dev set for benchmark evaluation.

## Important: 0% Overlap by Design!

The 11 databases in this eval set are **intentionally different** from the 69 databases 
used in training. This is how BIRD benchmark works - it tests generalization to unseen schemas.

## Databases in Eval Set (NEVER in training)
- california_schools
- card_games  
- codebase_community
- debit_card_specializing
- european_football_2
- financial
- formula_1
- student_club
- superhero
- thrombosis_prediction
- toxicology

## Files
- `bird_dev.jsonl`: 1,534 examples in ChatML-like format
  - Contains question, gold_sql, db_id, difficulty
  - Schema must be extracted from actual SQLite files during evaluation

## Usage
Run evaluation with `evaluation/run_bird_eval.sh` which:
1. Loads actual SQLite databases from BIRD benchmark
2. Extracts DDL schemas using `SELECT sql FROM sqlite_master`
3. Generates predictions
4. Computes execution accuracy

# BIRD New (2025-11-06) T12 Pipeline

This folder provides an alternate T12 pipeline for the updated BIRD dev dataset:
`birdsql/bird_sql_dev_20251106`.

## Files

- `fetch_bird_new.sh` - download updated dev set from Hugging Face.
- `build_prompts_bird_new.sh` - build T12 prompts from the new dev JSON.
- `predict_bird_new.sh` - run prediction on `bird_new_dev_t12.jsonl`.
- `eval_bird_new.sh` - evaluate predictions against the new dev JSON.
- `error_correction_bird_new.sh` - run V2 error correction on new-dev eval outputs.
- `eval_repaired_bird_new.sh` - evaluate repaired outputs and compare against baseline eval.

## Quick Run

```bash
cd /Users/arnav/programming/lm

# 1) Fetch updated dev set
bash data/training/t12/bird_new/fetch_bird_new.sh

# 2) Build prompts
bash data/training/t12/bird_new/build_prompts_bird_new.sh

# 3) Predict
bash data/training/t12/bird_new/predict_bird_new.sh \
  Qwen/Qwen3.5-2B \
  runs/t12_bird_new/predictions \
  runs/t12_baseline_3090

# 4) Evaluate
bash data/training/t12/bird_new/eval_bird_new.sh \
  runs/t12_bird_new/predictions/predictions_t12.jsonl \
  runs/t12_bird_new/eval

# 5) Error correction
bash data/training/t12/bird_new/error_correction_bird_new.sh \
  runs/t12_bird_new/predictions/predictions_t12.jsonl \
  runs/t12_bird_new/eval/per_example_results.jsonl \
  runs/t12_bird_new/error_correction \
  Qwen/Qwen3.5-2B \
  runs/error_correction_qwen3_5_2b_3090

# 6) Evaluate repaired predictions
bash data/training/t12/bird_new/eval_repaired_bird_new.sh \
  runs/t12_bird_new/error_correction/repaired_predictions_t12.jsonl \
  runs/t12_bird_new/eval/eval_report_t12.json \
  runs/t12_bird_new/error_correction/eval
```

## Notes

- The new dev set has `1534` examples (JSON array pretty-printed over `12273` lines).
- It uses the same 11 database IDs as your existing `data/bird_eval_datasets/dev_databases`,
  so no additional DB package is required in this workspace setup.
- Compared to your old `dev.json`, `619` aligned rows were updated (question/evidence/SQL changes).
- `eval_bird_new.sh` disables tied-append by default (`--no_tied_append`) because this update
  does not ship a local tied-append file in this folder.

# T12 Error Correction

V1 repairs only **non-executing SQL** from the T12 greedy LoRA baseline.
Wrong-result cases are classified but **not** auto-repaired.

## Inputs

- Predictions: `runs/t12_baseline_3090/qwen3.5-2b/without-sampling/predictions/predictions_t12.jsonl`
- Per-example eval: `runs/t12_baseline_3090/qwen3.5-2b/without-sampling/eval/per_example_results.jsonl`
- Eval report: `runs/t12_baseline_3090/qwen3.5-2b/without-sampling/eval/eval_report_t12.json`
- Prompts: `data/training/t12/bird_dev_t12.jsonl`
- Databases: `data/bird_eval_datasets/dev_databases`

## Full Repair Run

```bash
cd /Users/arnav/programming/lm

python data/training/t12/error-correction/run_error_correction.py \
  --predictions runs/t12_baseline_3090/predictions/predictions_t12.jsonl \
  --eval_results runs/t12_baseline_3090/eval/per_example_results.jsonl \
  --prompts data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t12/error-correction \
  --model_id Qwen/Qwen3.5-2B \
  --max_repair_attempts 2 \
  --generation_batch_size 61 \
  --min_repairability_score 0.5
```

## Small Validation Batch

Use this before a full run.

```bash
cd /Users/arnav/programming/lm

python data/training/t12/error-correction/run_error_correction.py \
  --predictions data/training/t12/error-correction/validation-batch/predictions_execfail_validation.jsonl \
  --eval_results data/training/t12/error-correction/validation-batch/per_example_results_execfail_validation.jsonl \
  --prompts data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t12/error-correction/validation-batch/output \
  --model_id Qwen/Qwen3.5-2B \
  --enable_thinking \
  --max_repair_attempts 2 \
  --generation_batch_size 8 \
  --min_repairability_score 0.5
```

## Evaluate Repaired Predictions

```bash
cd /Users/arnav/programming/lm

python data/training/t12/error-correction/evaluate_repaired.py \
  --repaired_predictions data/training/t12/error-correction/repaired_predictions_t12.jsonl \
  --original_eval runs/t12_baseline_3090/qwen3.5-2b/without-sampling/eval/eval_report_t12.json \
  --prompts data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t12/error-correction
```

## Evaluate Validation Batch

```bash
cd /Users/arnav/programming/lm

python data/training/t12/error-correction/evaluate_repaired.py \
  --repaired_predictions data/training/t12/error-correction/validation-batch/output/repaired_predictions_t12.jsonl \
  --original_eval runs/t12_baseline_3090/qwen3.5-2b/without-sampling/eval/eval_report_t12.json \
  --prompts data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t12/error-correction/validation-batch/output
```

## Outputs

Main run writes:

- `repaired_predictions_t12.jsonl`
- `repair_log_t12.jsonl`
- `quarantined_repairs_t12.jsonl`
- `repair_summary_t12.json`
- `repair_eval_report_t12.json`
- `repair_eval_summary_t12.md`

## Notes

- Repair decoding is greedy: `do_sample=False`
- Repair model is `Qwen/Qwen3.5-2B`
- `max_repair_attempts` is capped at `2`
- GPU generation is batched with `--generation_batch_size`
- Progress, throughput, and ETA are shown through a `tqdm` progress bar
- High-diff or structurally suspicious repairs go to quarantine

# T10 Error Correction

V1.1 adds a **deterministic tiny-fix path** that runs before LLM repair for the safest, smallest fixes.

V1 repairs only **non-executing SQL** from the T10 greedy LoRA baseline.
Wrong-result cases are classified but **not** auto-repaired.

## V1.1 Features

- **Deterministic fast-path repair**: Handles exact identifier typos, pure alias swaps, missing backticks, and obvious wrong-table-side errors without calling the LLM
- **Narrowed alias classification**: Only classifies as `alias_error` when fix is truly a pure alias swap
- **Improved pilot filtering**: Skips difficult repair classes (generic_exec_error, derived_metric_error, join_backbone_error)
- **Enhanced logging**: Tracks `repair_mode` (deterministic vs llm) and detailed deterministic attempt metadata
- **No-thinking mode default**: Conservative decoding without thinking tags for remaining LLM repairs

## Inputs

- Predictions: `runs/t10_baseline_3090/qwen3-1.7b/without-sampling/predictions/predictions_t10.jsonl`
- Per-example eval: `runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/per_example_results.jsonl`
- Eval report: `runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_report_t10.json`
- Prompts: `data/training/t10/bird_dev_t10.jsonl`
- Databases: `data/bird_eval_datasets/dev_databases`

## Full Repair Run

**V1.1 (no-thinking, with deterministic path):**
```bash
cd /Users/arnav/programming/lm

python data/training/t10/error-correction/run_error_correction.py \
  --predictions runs/t10_baseline_3090/qwen3-1.7b/without-sampling/predictions/predictions_t10.jsonl \
  --eval_results runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/per_example_results.jsonl \
  --prompts data/training/t10/bird_dev_t10.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t10/error-correction \
  --model_id Qwen/Qwen3-1.7B \
  --max_repair_attempts 2 \
  --generation_batch_size 32 \
  --min_repairability_score 0.5
```

**V1.0 (with thinking - legacy):**
```bash
cd /Users/arnav/programming/lm

python data/training/t10/error-correction/run_error_correction.py \
  --predictions runs/t10_baseline_3090/qwen3-1.7b/without-sampling/predictions/predictions_t10.jsonl \
  --eval_results runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/per_example_results.jsonl \
  --prompts data/training/t10/bird_dev_t10.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t10/error-correction \
  --model_id Qwen/Qwen3-1.7B \
  --enable_thinking \
  --max_repair_attempts 2 \
  --generation_batch_size 32 \
  --min_repairability_score 0.5
```

## Small Validation Batch

Use this before a full run.

**V1.1 (no-thinking, recommended):**
```bash
cd /Users/arnav/programming/lm

python data/training/t10/error-correction/run_error_correction.py \
  --predictions data/training/t10/error-correction/validation-batch/predictions_execfail_validation.jsonl \
  --eval_results data/training/t10/error-correction/validation-batch/per_example_results_execfail_validation.jsonl \
  --prompts data/training/t10/bird_dev_t10.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t10/error-correction/validation-batch/output \
  --model_id Qwen/Qwen3-1.7B \
  --max_repair_attempts 2 \
  --generation_batch_size 8 \
  --min_repairability_score 0.5
```

## Evaluate Repaired Predictions

```bash
cd /Users/arnav/programming/lm

python data/training/t10/error-correction/evaluate_repaired.py \
  --repaired_predictions data/training/t10/error-correction/repaired_predictions_t10.jsonl \
  --original_eval runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_report_t10.json \
  --prompts data/training/t10/bird_dev_t10.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t10/error-correction
```

## Evaluate Validation Batch

```bash
cd /Users/arnav/programming/lm

python data/training/t10/error-correction/evaluate_repaired.py \
  --repaired_predictions data/training/t10/error-correction/validation-batch/output/repaired_predictions_t10.jsonl \
  --original_eval runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_report_t10.json \
  --prompts data/training/t10/bird_dev_t10.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t10/error-correction/validation-batch/output
```

## Outputs

Main run writes:

- `repaired_predictions_t10.jsonl`
- `repair_log_t10.jsonl`
- `quarantined_repairs_t10.jsonl`
- `repair_summary_t10.json`
- `repair_eval_report_t10.json`
- `repair_eval_summary_t10.md`

## Notes

- V1.1 introduces deterministic repair path for safest fixes before LLM
- Repair decoding is greedy: `do_sample=False`
- Default mode is **no-thinking** (conservative, precise repairs)
- Use `--enable_thinking` flag to enable thinking mode (legacy V1.0 behavior)
- Repair model is `Qwen/Qwen3-1.7B`
- `max_repair_attempts` is capped at `2`
- GPU generation is batched with `--generation_batch_size`
- Progress, throughput, and ETA are shown through a `tqdm` progress bar
- High-diff or structurally suspicious repairs go to quarantine
- V1.1 skips difficult repair classes in pilot: generic_exec_error, derived_metric_error, join_backbone_error

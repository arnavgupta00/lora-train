# T12 Error Correction

V2 repairs **all non-correct SQL** from the T12 baseline (exec-failed and wrong-result cases).

Acceptance rule is strict:
- a repair is accepted only if repaired SQL execution results match gold SQL execution results.

Wrong-result cases now use dedicated semantic repair prompts (not exec-error wording).

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
  --generation_batch_size 61
```

## Full Repair Run with LoRA Adapter

```bash
cd /Users/arnav/programming/lm

python data/training/t12/error-correction/run_error_correction.py \
  --predictions runs/t12_baseline_3090/predictions/predictions_t12.jsonl \
  --eval_results runs/t12_baseline_3090/eval/per_example_results.jsonl \
  --prompts data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir data/training/t12/error-correction \
  --model_id Qwen/Qwen3.5-2B \
  --adapter_path runs/error_correction_qwen3_5_2b_3090 \
  --max_repair_attempts 2 \
  --generation_batch_size 61
```

## Run on `t12_sc_n7` Outputs with HF Adapter

Use your deployed HF adapter repo ID directly:

```bash
cd /Users/arnav/programming/lm

python data/training/t12/error-correction/run_error_correction.py \
  --predictions runs/t12_sc_n7/predictions_sc_t12.jsonl \
  --eval_results runs/t12_sc_n7/per_example_results_sc_t12.jsonl \
  --prompts data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir runs/t12_sc_n7/error_correction_hf_adapter \
  --model_id Qwen/Qwen3.5-2B \
  --adapter_path Arnav3035/error-correction-qwen3-5-2b-3090 \
  --max_repair_attempts 2 \
  --generation_batch_size 61
```

For this script, only these `t12_sc_n7` inputs are required:
- `runs/t12_sc_n7/predictions_sc_t12.jsonl`
- `runs/t12_sc_n7/per_example_results_sc_t12.jsonl`

These are useful artifacts but not direct inputs to `run_error_correction.py`:
- `runs/t12_sc_n7/candidates_sc_t12.jsonl`
- `runs/t12_sc_n7/evaluation_report_sc_t12.json`
- `runs/t12_sc_n7/eval_summary_sc_t12.md`

Use a workspace-relative adapter path (for example `runs/...`) or a full path like
`/workspace/lora-train/runs/...`. Avoid `/runs/...` unless that directory actually exists
at filesystem root.

If `--adapter_path` is used, install PEFT:

```bash
pip install peft
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
  --generation_batch_size 8
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

Evaluation script writes:

- `repair_per_example_results.jsonl`
- `repair_eval_report_t12.json`
- `repair_eval_summary_t12.md`

## Notes

- Repair decoding is greedy: `do_sample=False`
- Repair model is `Qwen/Qwen3.5-2B`
- Optional adapter: `--adapter_path runs/error_correction_qwen3_5_2b_3090` or `--adapter_path Arnav3035/error-correction-qwen3-5-2b-3090`
- `max_repair_attempts` is capped at `2`
- GPU generation is batched with `--generation_batch_size`
- `--min_repairability_score` is deprecated in V2 and ignored (all non-correct examples are attempted)
- Progress, throughput, and ETA are shown through a `tqdm` progress bar
- High-diff or structurally suspicious repairs go to quarantine

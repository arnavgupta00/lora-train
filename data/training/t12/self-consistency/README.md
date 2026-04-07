# T12 Self-Consistency (N=7, Sampling-Only)

This folder contains a production-style self-consistency runner for T12 prompts.

Script:
- `run_self_consistency_t12.py`

## What It Does

1. Loads prompts from `bird_dev_t12.jsonl`.
2. Loads base model `Qwen/Qwen3.5-2B` and adapter `Arnav3035/garuda-sql-2b`.
3. Generates `n=7` sampled SQL candidates per example (`do_sample=True`, `temperature>0`).
4. Applies execution-aware voting:
- execute each candidate,
- group successful candidates by result signature,
- select SQL from the largest execution-result group,
- fallback to most frequent SQL when all candidates fail.
5. Evaluates voted predictions against gold SQL on BIRD dev databases.

## Parallelization

- Batched generation via `--batch_size`.
- Parallel voting via `--vote_workers`.
- Parallel final evaluation via `--eval_workers`.

## Guaranteed Sampling Mode

This runner intentionally disallows greedy/self-consistency mismatch:
- `n_samples` must be `7`.
- `temperature` must be `> 0.0`.
- Decoding is always `do_sample=True`, `num_beams=1`.

## Run Command

From repo root:

```bash
python data/training/t12/self-consistency/run_self_consistency_t12.py \
  --base_model_id Qwen/Qwen3.5-2B \
  --adapter_repo Arnav3035/garuda-sql-2b \
  --prompts_file data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir runs/t12_sc_n7 \
  --n_samples 7 \
  --temperature 0.5 \
  --top_p 0.9 \
  --top_k 50 \
  --batch_size 8 \
  --vote_workers 8 \
  --eval_workers 8
```

## Quick Pilot Run (fast sanity check)

```bash
python data/training/t12/self-consistency/run_self_consistency_t12.py \
  --output_dir runs/t12_sc_n7_pilot \
  --limit 200 \
  --n_samples 7 \
  --temperature 0.5
```

## Resume After Interruption / OOM

If a run stops mid-way, restart with the same `--output_dir` and add `--resume`.
The script will load existing `predictions_sc_t12.jsonl`, skip completed items,
continue from remaining examples, and print running EX progress.

```bash
python data/training/t12/self-consistency/run_self_consistency_t12.py \
  --base_model_id Qwen/Qwen3.5-2B \
  --adapter_repo Arnav3035/garuda-sql-2b \
  --prompts_file data/training/t12/bird_dev_t12.jsonl \
  --db_dir data/bird_eval_datasets/dev_databases \
  --output_dir runs/t12_sc_n7 \
  --n_samples 7 \
  --temperature 0.5 \
  --top_p 0.9 \
  --top_k 50 \
  --batch_size 4 \
  --min_batch_size 1 \
  --vote_workers 8 \
  --eval_workers 8 \
  --resume
```

If OOM happens again, lower `--batch_size` (for example `2`) and/or reduce
`--max_new_tokens`.

## Output Files

In `--output_dir`:
- `generation_config_sc_t12.json`
- `predictions_sc_t12.jsonl`
- `candidates_sc_t12.jsonl`
- `evaluation_report_sc_t12.json`
- `per_example_results_sc_t12.jsonl`
- `eval_summary_sc_t12.md`

## Notes

- Requires: `transformers`, `torch`, `peft`.
- Uses execution-based voting inspired by execution-guided SQL decoding and self-consistency decoding.
- For reproducibility, set `--seed` (default `42`).

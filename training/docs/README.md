# NL2SQL LoRA Fine-Tuning (Runpod)

This folder contains a minimal LoRA-only (not QLoRA) training + execution-match eval workflow for the dataset in `dataset/t2/`.

## Runpod quickstart (RTX PRO 6000 96GB)

Recommended volume: `250GB` mounted at `/runpod-volume`.

From repo root:

```bash
bash finetune_nl2sql/run_qwen14b_lora.sh
bash finetune_nl2sql/run_qwen32b_lora.sh
```

## Resuming From A Checkpoint

If a run gets interrupted, you can resume training from a saved checkpoint directory (e.g. `.../checkpoint-300`).

Example:

```bash
export DATASET_DIR="/workspace/dataset"
export NL2SQL_ADMIN_API_KEY="..."
export RESUME_FROM="/workspace/lora-train/outputs/qwen2.5-14b-instruct-lora-YYYYMMDD_HHMMSS/checkpoint-300"
export OUT_DIR="/workspace/lora-train/outputs/qwen2.5-14b-instruct-lora-YYYYMMDD_HHMMSS"
bash finetune_nl2sql/run_qwen14b_lora.sh
```

## Tailing Logs

Runner scripts write a per-run log file at `<OUT_DIR>/run.log` and print a tail command. Example:

```bash
tail -n 200 -f /workspace/lora-train/outputs/qwen2.5-14b-instruct-lora-YYYYMMDD_HHMMSS/run.log
```

If you want to **train both models first, then eval both**:

```bash
bash finetune_nl2sql/run_train_both_then_eval.sh
```

## Validator API key (required for execution-match)

The evaluator imports the key from `finetune_nl2sql/private_key.py`, which is gitignored.

Option A (recommended): set an env var once and let the runner script write the file:

```bash
export NL2SQL_ADMIN_API_KEY="..."
```

Option B: create it manually:

```bash
cp finetune_nl2sql/private_key.py.example finetune_nl2sql/private_key.py
```

## What gets produced

Each run writes to `/runpod-volume/outputs/<run_name>/`:
- `adapter_model.safetensors` (or equivalent) and PEFT config
- tokenizer files
- training logs/metrics
- `predictions.test.lora.jsonl` + `eval_report.lora.json` (execution-match + exact-match)

If you set `EVAL_BASE=1`, it also writes:
- `predictions.test.base.jsonl` + `eval_report.base.json`

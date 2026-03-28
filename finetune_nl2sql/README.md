# NL2SQL LoRA Fine-Tuning (Runpod)

This folder contains a minimal LoRA-only (not QLoRA) training + execution-match eval workflow for the dataset in `dataset/t2/`.

## Runpod quickstart (RTX PRO 6000 96GB)

Recommended volume: `250GB` mounted at `/runpod-volume`.

From repo root:

```bash
bash finetune_nl2sql/run_qwen14b_lora.sh
bash finetune_nl2sql/run_qwen32b_lora.sh
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
- `predictions.test.jsonl` + `eval_report.json` (execution-match + exact-match)

# lora-train

Minimal LoRA-only NL2SQL fine-tuning + execution-match evaluation (with base-vs-LoRA comparison).

## What’s In Here

- `dataset/t2/`: ChatML JSONL dataset (train/dev/test)
- `finetune_nl2sql/`: LoRA training + execution-match eval scripts (Runpod-friendly)
- `reports/`: run notes + metrics

## Latest Run (2026-03-28)

Headline: LoRA improved strict execution-match vs the base models, but accuracy is still low on a mixed multi-schema test set.

See:

- `RESULTS_SUMMARY.md`
- `reports/2026-03-28_run.md`

## Reproduce

Runpod instructions: `finetune_nl2sql/README.md`.


# lora-train

Minimal LoRA-only NL2SQL fine-tuning + execution-match evaluation (with base-vs-LoRA comparison).

## What’s In Here

- `dataset/t2/`: ChatML JSONL dataset (train/dev/test)
- `finetune_nl2sql/`: LoRA training + execution-match eval scripts (Runpod-friendly)
- `.github/skills/nl2sql-schema-gen-v2/`: Copilot workflow to generate + provision 30 new schemas per run (Cloudflare D1)
- `.github/skills/nl2sql-example-gen-v1/`: Copilot workflow to top up NL→SQL examples per schema to a target split ratio
- `schema_registry.json`: local registry of generated schemas (used to avoid duplicates)
- `reports/`: run notes + metrics

## Latest Run (2026-03-28)

Headline: LoRA improved strict execution-match vs the base models, but accuracy is still low on a mixed multi-schema test set.

See:

- `RESULTS_SUMMARY.md`
- `reports/2026-03-28_run.md`

## Reproduce

Runpod instructions: `finetune_nl2sql/README.md`.

## Scale Up (Many Schemas)

If you want to grow schema coverage without duplicating prior work, use the Copilot workflow:

- `.github/skills/nl2sql-schema-gen-v2/FLOW.md`

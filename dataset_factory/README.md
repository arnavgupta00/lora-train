# Dataset Factory (Schema-General NL2SQL)

Goal: generate large, schema-diverse NL→SQL datasets where **SQL is algorithmic** and only the **natural language** is LLM-assisted (Copilot/Claude Opus, etc.).

This is the scalable path for "generalize to unseen schemas" because you can produce many schemas cheaply, enforce deterministic SQL rules, and validate everything in SQLite locally.

## Workflow (high level)

1. Generate `schema_specs/*.json` using Copilot prompts in `dataset_factory/copilot_prompts.md`.
2. Build and validate gold SQL algorithmically:
   - `python3 dataset_factory/build_sql.py --spec_dir schema_specs --out_dir builds/`
3. Create paraphrase tasks (batch JSON) to paste into Copilot:
   - `python3 dataset_factory/make_paraphrase_tasks.py --build_dir builds --out paraphrase_tasks.json`
4. Paste `paraphrase_tasks.json` into Copilot, get `paraphrases.json`.
5. Merge paraphrases into final JSONL splits:
   - `python3 dataset_factory/merge_paraphrases.py --build_dir builds --paraphrases paraphrases.json --out_dir dataset_generated/`

## Output

The final dataset is written as JSONL with `messages[]` compatible with `finetune_nl2sql/`.

## Notes

- Everything runs in local SQLite for validation. You can later ingest into Cloudflare D1 if you want, but D1-per-schema does not scale to hundreds of schemas without infra work.
- Splits are **schema-aware** by default (train/dev/test contain disjoint schema sets).


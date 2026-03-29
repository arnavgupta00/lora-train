# Stage 2: Ingest Examples Into Dataset Service

This stage is mechanical: send the generated examples to the dataset service in batches.

---

## Inputs

- Config: `.github/skills/nl2sql-example-gen-v1/config/config.json`
- Plan: `{run_dir}/stage00_plan.json`
- Generated examples: `{run_dir}/stage01_generated_examples.json`

---

## Required Env

- `ADMIN_API_KEY` (name configurable in config)

---

## Run

```bash
python3 scripts/nl2sql_example_gen/ingest_examples.py \
  --config .github/skills/nl2sql-example-gen-v1/config/config.json \
  --plan {run_dir}/stage00_plan.json \
  --examples {run_dir}/stage01_generated_examples.json \
  --out {run_dir}/stage02_ingest_report.json
```

This posts to:
- `POST {base_url}/v1/examples/batch`

Behavior:
- Ingests in chunks.
- Tracks accepted rows per `(schema_id, split)`.
- Stops sending once the deficit is satisfied.
- Records rejections with error messages for debugging/regeneration.


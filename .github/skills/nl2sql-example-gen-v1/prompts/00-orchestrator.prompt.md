# Stage 0: Orchestrator — Example Gen V1 (10 Schemas / Run)

You are the **master orchestrator** for a multi-stage NL→SQL example generation pipeline.

Your job:
1. Plan the run by selecting 10 schemas that are most under-filled vs target.
2. Generate only the missing examples (with a buffer to offset rejects).
3. Ingest examples into the dataset service.
4. Re-count to confirm targets moved toward 150 per schema.

Enforce validation gates between stages. If a gate fails, fix and retry that stage.

---

## STEP 1: Load Config + Create Run Directory

Read config:
`.github/skills/nl2sql-example-gen-v1/config/config.json`

Create run directory:
`{runs_dir}/{run_name}/` (from config)

---

## STEP 2: Plan The Run (Select 10 Schemas + Compute Deficits)

Run:

```bash
python3 scripts/nl2sql_example_gen/plan_run.py \
  --config .github/skills/nl2sql-example-gen-v1/config/config.json
```

This writes (inside the run directory):
- `stage00_plan.json` (targets + current counts + deficits per schema/split)
- `stage00_schemas.json` (schema_context + business_rules per schema)

**VALIDATION GATE V0**
- [ ] `stage00_plan.json` exists and is valid JSON
- [ ] Contains `selected_schemas` length == `schemas_per_run` (unless fewer schemas are under target)
- [ ] Every selected schema has non-zero deficit (at least one split needs rows)

If 0 schemas are under target: stop (you’re done).

---

## STEP 3: Generate Examples (Stage 01)

Read and execute:
`.github/skills/nl2sql-example-gen-v1/prompts/01-example-generator.prompt.md`

Write output JSON to:
`{run_dir}/stage01_generated_examples.json`

**VALIDATION GATE V1**
- [ ] Output file exists and parses as JSON
- [ ] Has `examples: []`
- [ ] Every example has:
  - `schema_id`, `schema_version`, `split`, `question`, `gold_sql`
  - `external_id` unique across the file
- [ ] Each selected schema/split has at least `deficit + buffer` examples in the file

If V1 fails, fix and regenerate Stage 01.

---

## STEP 4: Ingest Examples (Stage 02)

Read and execute:
`.github/skills/nl2sql-example-gen-v1/prompts/02-ingest.prompt.md`

This stage should:
- POST to `/v1/examples/batch` in chunks
- stop sending examples for a schema/split once it has satisfied the deficit (based on accepted rows)
- write `stage02_ingest_report.json`

**VALIDATION GATE V2**
- [ ] `stage02_ingest_report.json` exists
- [ ] Total accepted count > 0 (unless everything already met target)
- [ ] Rejections are explainable (bad SQL, wrong table/col, nondeterminism)

---

## STEP 5: Re-count And Decide Next Action

Re-run the planner:

```bash
python3 scripts/nl2sql_example_gen/plan_run.py \
  --config .github/skills/nl2sql-example-gen-v1/config/config.json \
  --schema_ids_from_plan {run_dir}/stage00_plan.json
```

If deficits remain for the selected schemas:
- regenerate Stage 01 focusing only on remaining deficit (do not create a new run dir)
- ingest again

If deficits are 0 for all selected schemas:
- stop this run; next run will pick the next 10 schemas automatically.


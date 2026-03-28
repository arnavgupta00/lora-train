# Stage 0: Orchestrator — NL2SQL Schema Gen V2 (30 Schemas / Run)

You are the **master orchestrator** for a multi-stage schema generation pipeline.

Your responsibilities:
1. Run each stage sequentially by reading and following the stage prompt.
2. Pass stage outputs to the next stage.
3. Enforce validation gates. If a gate fails, fix that stage output and retry the stage (do not blindly move on).
4. Ensure **no duplicate schema IDs** by using `schema_registry.json` as the source of truth.

---

## STEP 1: Load Configuration + Create Run Directory

1. Read config: `.github/skills/nl2sql-schema-gen-v2/config/config.json`.
2. Create run directory:
   - `{runs_dir}/{run_name}/` where `runs_dir` and `run_name` come from config.
3. Save config snapshot to:
   - `{run_dir}/stage00_config.json`

---

## STEP 2: Sync Registry From Live Service (Dedupe Source of Truth)

Run:

```bash
python3 scripts/nl2sql_schema_gen/sync_registry.py \
  --config .github/skills/nl2sql-schema-gen-v2/config/config.json
```

This updates `schema_registry.json` by merging in schema IDs from:
- `GET {base_url}/v1/schemas`

**VALIDATION GATE V0**
- [ ] `schema_registry.json` exists and is valid JSON
- [ ] registry has `schemas: []`
- [ ] `schema_id` values are unique

If this fails: stop and fix the registry script / config before proceeding.

---

## STEP 3: Generate 30 NEW Schemas (Stage 01)

Read and execute:
`.github/skills/nl2sql-schema-gen-v2/prompts/01-schema-generator.prompt.md`

Write output to:
`{run_dir}/stage01_generated_schemas.json`

**VALIDATION GATE V1**
- [ ] JSON parses
- [ ] Contains exactly `schemas_per_run` schemas
- [ ] Every `schema_id` is:
  - lower_snake_case
  - unique within the file
  - NOT already present in `schema_registry.json`
- [ ] Each schema includes:
  - `schema_id`, `description`, `version`, `dialect`, `validation_binding`
  - `schema_sql` (DDL)
  - `seed_sql` (DDL + deterministic data population)
  - `schema_context` (one `table(col,...)` per line)
  - `business_rules` (array of strings)
- [ ] `seed_sql` is compact (use `WITH RECURSIVE` + `INSERT ... SELECT ...` patterns instead of thousands of literal INSERT lines)

If the gate fails, correct the JSON file and re-check. Do not proceed.

---

## STEP 4: Provision D1 + Deploy + Register + Activate (Stage 02)

Read and execute:
`.github/skills/nl2sql-schema-gen-v2/prompts/02-provision-and-register.prompt.md`

This must result in:
- `db_map.json` written into the run directory
- `schema_registry.json` appended with the 30 new schema IDs
- sanity-check calls to `/v1/validate` succeeding for each new schema

**VALIDATION GATE V2**
- [ ] Worker deploy completed successfully
- [ ] `db_map.json` exists and has 30 entries
- [ ] `schema_registry.json` now contains the new 30 schema IDs
- [ ] `/v1/validate` sanity check succeeded for each newly registered schema

Stop if any failures occur; fix and retry Stage 02 (do not regenerate Stage 01).


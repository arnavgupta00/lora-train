# Stage 2: Provision D1 + Deploy + Register + Activate + Sanity Check

You are the provisioning agent. This stage should be mostly mechanical and automated.

---

## Inputs

- Config: `.github/skills/nl2sql-schema-gen-v2/config/config.json`
- Generated schemas: `{run_dir}/stage01_generated_schemas.json`
- Registry: `schema_registry.json`

---

## Required Environment Variables

- Cloudflare auth (Wrangler): `CLOUDFLARE_API_TOKEN`
- Dataset service admin key: `ADMIN_API_KEY`

Note: env var names are configurable in `config.json`.

---

## Run The Automation Script

Run:

```bash
python3 scripts/nl2sql_schema_gen/provision_and_register.py \
  --config .github/skills/nl2sql-schema-gen-v2/config/config.json \
  --schemas {run_dir}/stage01_generated_schemas.json \
  --run_dir {run_dir}
```

What it does:
- Creates one D1 DB per schema
- Executes `seed_sql` to create tables + seed data
- Patches the dataset service `wrangler.toml` with new `[[d1_databases]]` bindings
- Deploys the dataset service Worker
- Registers each schema (`POST /v1/schemas`) and activates it (`POST /v1/schemas/{id}/{ver}/activate`)
- Runs a per-schema `/v1/validate` sanity check (so we know execution routing is correct)
- Updates `schema_registry.json` (append new schema IDs + db ids)

---

## Validation Gates

- [ ] Script exits 0
- [ ] `{run_dir}/db_map.json` exists and has one entry per schema_id
- [ ] Registry updated (new schema_ids present)
- [ ] Sanity check succeeded for each schema (no routing failures)

If any schema fails:
- Do NOT regenerate Stage 01.
- Fix the cause (bad seed_sql, invalid DDL, binding mismatch, deploy failure) and rerun Stage 02.


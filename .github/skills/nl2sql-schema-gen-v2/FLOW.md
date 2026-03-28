# NL2SQL Schema Gen V2 — 30 Schemas Per Run (D1-Provisioned, Registry-Deduped)

## Overview

This is a **3-stage Copilot prompt chain** that generates **30 new benchmark schemas per run**, provisions a **Cloudflare D1 database per schema** (so `/v1/validate` execution works), registers + activates the schemas in the **NL2SQL Dataset Service**, and updates a **local registry JSON** so future runs do not duplicate schema IDs.

This is intentionally structured like `product-selection-v2`: orchestrator + stage prompts + validation gates + resumability.

---

## Key Files

- Config: `.github/skills/nl2sql-schema-gen-v2/config/config.json`
- Prompts:
  - `.github/skills/nl2sql-schema-gen-v2/prompts/00-orchestrator.prompt.md`
  - `.github/skills/nl2sql-schema-gen-v2/prompts/01-schema-generator.prompt.md`
  - `.github/skills/nl2sql-schema-gen-v2/prompts/02-provision-and-register.prompt.md`
- State:
  - `schema_registry.json` (source of truth for “already generated”)
- Automation:
  - `scripts/nl2sql_schema_gen/sync_registry.py`
  - `scripts/nl2sql_schema_gen/provision_and_register.py`

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ 00-orchestrator.prompt.md                                         │
│ - loads config + makes run dir                                    │
│ - syncs local schema_registry.json from remote service            │
│ - runs Stage 01 to generate schemas JSON                          │
│ - runs Stage 02 automation to provision/register/activate          │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ 01-schema-generator.prompt  │
                 │ - generate 30 NEW schemas   │
                 │ - strict JSON output        │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ 02-provision-and-register   │
                 │ - create D1 per schema      │
                 │ - seed DB via seed_sql      │
                 │ - patch wrangler.toml       │
                 │ - deploy worker             │
                 │ - register + activate       │
                 │ - sanity check /v1/validate │
                 │ - update schema_registry    │
                 └─────────────────────────────┘
```

---

## Resumability / Dedupe Strategy

- `schema_registry.json` is the dedupe list. Every run must load it and refuse to generate any `schema_id` already present.
- Stage 00 always syncs remote schemas from the live service first (so you also dedupe against work done elsewhere).
- Run artifacts are written to `runs/nl2sql-schema-gen/{run_name}/`:
  - `stage00_config.json` (config snapshot)
  - `stage01_generated_schemas.json` (the exact 30 schemas created by Copilot)
  - `db_map.json` (schema_id -> D1 db name/id/binding)

---

## Prerequisites (One-Time)

1. You have the NL2SQL Dataset Service repo checked out locally (path configured in `config.json`).
2. Wrangler is usable in that repo:
   - `npx wrangler --version` works
   - You are authenticated (via `CLOUDFLARE_API_TOKEN`)
3. You have the dataset service admin key exported:
   - `export ADMIN_API_KEY="..."` (env name is configurable)

---

## How To Run In Copilot

In VS Code Copilot Chat, paste:

```text
Read .github/skills/nl2sql-schema-gen-v2/prompts/00-orchestrator.prompt.md and execute the full pipeline it describes.
```

If anything fails at a validation gate, fix that stage output and rerun from the failing stage (do not regenerate a whole new set unless required).


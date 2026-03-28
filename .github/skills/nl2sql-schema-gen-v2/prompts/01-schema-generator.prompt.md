# Stage 1: Schema Generator — Generate 30 New Schemas (Strict JSON)

You are generating new schemas for NL→SQL benchmarking and training.

Hard requirement: **DO NOT generate any `schema_id` already present in `schema_registry.json`.**

---

## Inputs You MUST Read

1. Config: `.github/skills/nl2sql-schema-gen-v2/config/config.json`
2. Existing registry: `schema_registry.json`

---

## Output (Write This File)

Write a JSON file to:

`{runs_dir}/{run_name}/stage01_generated_schemas.json`

where `runs_dir` and `run_name` come from config.

Format:

```json
{
  "schemas": [
    {
      "schema_id": "lower_snake_case_unique",
      "description": "One sentence describing the domain",
      "version": "v1",
      "dialect": "sqlite",
      "validation_binding": "DYNAMIC_<SCHEMA_ID_UPPER_SNAKE>_V1_DB",

      "schema_sql": "CREATE TABLE ...; ...",
      "seed_sql": "CREATE TABLE ...; WITH RECURSIVE ... INSERT ...; ...",

      "schema_context": "table(col1,col2,...)\\nother_table(colA,colB,...)",
      "business_rules": [
        "Short rule 1",
        "Short rule 2"
      ]
    }
  ]
}
```

---

## Schema Requirements

- Generate exactly `schemas_per_run` schemas (from config).
- Each schema:
  - 6–12 tables
  - at least 3 foreign keys
  - at least one 2-hop join path (A -> B -> C)
  - at least one numeric measure suited for SUM/AVG
  - supports common analytic shapes: top-N, group-by, date filtering, left joins
- Domain style: “business app” (SaaS, logistics, retail, fintech, manufacturing, etc.). Avoid PII and real names.
- SQLite dialect only.

---

## Seed SQL Requirements (Critical)

The `seed_sql` must be executable in SQLite/D1 and must be **compact**.

Do this:
- Use `WITH RECURSIVE seq(n) AS (...)` to generate row indices.
- Use `INSERT INTO table(...) SELECT ... FROM seq;` to populate 50–500 rows per table deterministically.
- Use stable deterministic expressions (e.g., `n`, `n % 10`, `date('now', '-'||n||' days')`).

Avoid this:
- Thousands of literal `INSERT VALUES (...)` lines.

Goal:
- Every table should end up with non-trivial data so execution-match is meaningful.

---

## Registry Dedupe Instruction (Must Follow)

Before picking schema IDs, read `schema_registry.json` and build a set of existing IDs.
Then generate new IDs that do not overlap (even if “similar domain”).

If any schema ID collides, you must rename it before finalizing output.


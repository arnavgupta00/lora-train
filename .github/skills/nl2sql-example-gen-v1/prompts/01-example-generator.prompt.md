# Stage 1: Example Generator — Produce NL Questions + Gold SQL (Strict JSON)

You generate NL→SQL examples for the selected schemas and splits.

Your output will be ingested by the dataset service which **validates SQL by execution**. Invalid SQL will be stored as rejected, so aim for correctness.

---

## Inputs You MUST Read

From the current run directory:
- `{run_dir}/stage00_plan.json`
- `{run_dir}/stage00_schemas.json`

Also read config:
- `.github/skills/nl2sql-example-gen-v1/config/config.json`

---

## Output File

Write:
`{run_dir}/stage01_generated_examples.json`

Format:

```json
{
  "examples": [
    {
      "external_id": "example-gen-run-001:b2b_saas:train:0001:4f2c9a1b",
      "schema_id": "b2b_saas",
      "schema_version": "v1",
      "split": "train",
      "question": "Which organizations are active?",
      "gold_sql": "SELECT name FROM organizations WHERE status = 'active' ORDER BY name",
      "tags": ["filter", "order_by"],
      "metadata": { "source": "copilot-example-gen-v1" }
    }
  ]
}
```

`external_id` must be **globally unique across all time** (the service enforces uniqueness). Include:
- the `run_name` from config
- the schema_id + split
- a counter
- a short random suffix (8+ hex chars)

---

## How Many Examples To Generate

For each selected schema, `stage00_plan.json` tells you:
- current counts
- target counts per split
- deficits per split
- buffer policy (from config)

Generate at least:

`deficit + buffer`

for each `(schema_id, split)` that has a deficit.

Over-generation is allowed because some examples may be rejected by validation.
The ingest script will stop once accepted rows satisfy the deficit.

---

## Quality Requirements (Make Validation Pass)

- SQL must be valid SQLite.
- Use only tables/columns in `schema_context`.
- Respect `business_rules` when relevant.
- Prefer deterministic queries:
  - include stable `ORDER BY` when returning multiple rows
  - avoid `RANDOM()` and time-dependent nondeterministic logic
- Include a variety of shapes:
  - simple filters, aggregations, joins, left joins, top-N, date filtering
  - multi-hop joins (2 joins) for some examples

---

## Split Guidance

- `train`: broad distribution, easier and medium complexity
- `dev`: harder edge cases and compositional queries
- `test`: difficult but still solvable; remember test rows will be stored as `pending_review` even if valid

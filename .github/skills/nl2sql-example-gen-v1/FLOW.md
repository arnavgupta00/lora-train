# NL2SQL Example Gen V1 — Top Up to 150 Examples Per Schema (10 Schemas / Run)

## Overview

This Copilot workflow generates **NL→SQL question/answer pairs** and stores them in the **NL2SQL Dataset Service**.

Per schema target:
- **150 total accepted examples** per schema
- split mix is a fixed ratio: **train : dev : test = 70 : 16 : 16**
  - The workflow converts that ratio into exact integer targets that sum to 150 (e.g. 103 / 24 / 23).

Per run:
- operates on **10 schemas** (chosen automatically as the most under-filled)
- generates only what’s missing (with a small over-generate buffer to offset validation rejects)

Everything is resumable and deduped by reading the dataset service inventory on each run.

---

## Key Files

- Config: `.github/skills/nl2sql-example-gen-v1/config/config.json`
- Prompts:
  - `.github/skills/nl2sql-example-gen-v1/prompts/00-orchestrator.prompt.md`
  - `.github/skills/nl2sql-example-gen-v1/prompts/01-example-generator.prompt.md`
  - `.github/skills/nl2sql-example-gen-v1/prompts/02-ingest.prompt.md`
- Scripts:
  - `scripts/nl2sql_example_gen/plan_run.py` (fetch schemas + count examples + compute deficits + pick 10)
  - `scripts/nl2sql_example_gen/ingest_examples.py` (POST /v1/examples/batch in chunks, stop when targets met)
- State:
  - `schema_registry.json` (not strictly required, but useful for visibility)

Run artifacts go into `runs/nl2sql-example-gen/{run_name}/` and are gitignored.

---

## How To Run In Copilot

Paste into Copilot Chat:

```text
Read .github/skills/nl2sql-example-gen-v1/prompts/00-orchestrator.prompt.md and execute the full pipeline it describes.
```


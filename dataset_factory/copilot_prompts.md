# Copilot Prompts (Claude Opus) for Dataset Factory

These prompts are designed so each Copilot request can produce a **batch** of work in a strict machine-ingestible format.

## 1) Schema Spec Batch Prompt (10–20 schemas per request)

Paste this prompt into Copilot, ask it to output **only JSON**.

```text
You are generating schema specs for a SQLite NL→SQL benchmark dataset.

Return ONLY valid JSON (no Markdown). Output must be:
{
  "schemas": [ ... ]
}

Generate 10 schema specs. Each schema must be in the "business app" style: 6–12 tables, realistic relationships, and columns that support joins and aggregation. Vary naming conventions across schemas (snake_case, camelCase, abbreviations), but keep identifiers SQL-safe.

Schema format:
{
  "schema_id": "lower_snake_case_unique",
  "description": "one sentence",
  "dialect": "sqlite",
  "tables": [
    {
      "name": "table_name",
      "columns": [
        {"name": "id", "type": "INTEGER", "pk": true},
        {"name": "created_at", "type": "TEXT", "semantic_type": "datetime"},
        {"name": "status", "type": "TEXT", "enum": ["active","inactive"]},
        ...
      ],
      "fks": [
        {"column": "org_id", "ref_table": "organizations", "ref_column": "id"}
      ]
    }
  ],
  "business_rules": [
    "Short rules like: Revenue is SUM(invoices.total_amount_usd).",
    "Active means status='active'."
  ],
  "seed_profile": {
    "rows_per_table": 200,
    "datetime_range_days": 365
  }
}

Constraints:
- Every schema must have at least 3 foreign keys and at least 1 many-to-one chain of length >=2 (for 2-hop joins).
- Include at least one numeric measure column suitable for SUM/AVG (e.g., amount_usd, units, duration_minutes).
- Use TEXT for dates (ISO 8601 strings).
- No PII. Use synthetic names/emails like user_123@example.com.
```

## 2) Paraphrase Batch Prompt (25 SQL queries per request)

This prompt takes a batch of SQL items and asks Copilot to produce 3 paraphrases each.

```text
You are generating natural language questions for NL→SQL training.

Return ONLY valid JSON (no Markdown), format:
{
  "items": [
    {
      "id": "<string>",
      "questions": ["q1", "q2", "q3"]
    }
  ]
}

Rules:
- Questions must be answerable using ONLY the schema and business rules provided.
- Preserve the required output contract implied by the SQL (e.g., if the SQL returns provider_name + avg_days, the question must ask for that).
- Vary wording across paraphrases: synonyms, different phrasing, implicit vs explicit.
- Do not mention SQL, tables, or columns directly unless the user would plausibly say it.
- Keep each question <= 25 words.

Here are the items:
<PASTE_ITEMS_JSON_HERE>
```

## 3) Repair Prompt (for validator rejects)

If an algorithmic SQL query fails due to being empty/trivial, do NOT ask Copilot to "invent SQL".
Instead ask it to propose a schema tweak (enum values, relationships) in the same JSON schema format.


# T11 Mixed-Schema Dataset

T11 is a text-to-SQL training dataset that mixes full-schema and compact-schema examples to test whether mixed exposure helps the model learn better schema focus.

## Design

**T11 = T10 + 50% compact-schema variants**

- ~45% compact-schema examples (oracle-derived from gold SQL)
- ~55% full-schema examples (unchanged from T10)
- Same T10 prompt contract preserved exactly
- Gold SQL targets unchanged

## Files

- `train_t11.jsonl` - 14,034 training examples
- `dev_t11.jsonl` - 665 dev examples
- `build_t11.py` - Main builder script
- `t11_utils.py` - Utility functions (DDL parsing, SQL extraction, schema compacting)
- `build_summary.json` - Build statistics and metadata

## Message Format

```json
{
  "messages": [
    {"role": "system", "content": "<T10 system prompt>"},
    {"role": "user", "content": "Schema:\n<full or compact DDL>\n\nHints:\n<hints>\n\nQuestion:\n<question>"},
    {"role": "assistant", "content": "<gold SQL>"}
  ],
  "db_id": "database_name",
  "schema_mode": "full" | "compact",
  "compaction_status": "success" | "not_applicable" | "fallback_*",
  "original_schema_len": 1234,
  "compact_schema_len": 567
}
```

## Compact Schema Construction

For training/dev construction, compact schemas are derived using the gold SQL:

1. **Parse full schema** - Extract tables, columns, PK, FK from DDL
2. **Parse gold SQL** - Extract referenced tables and columns (sqlparse + regex)
3. **Keep required elements**:
   - All tables referenced in SQL
   - Bridge tables for FK join connectivity
   - All SQL-referenced columns
   - PK and FK columns for joinability
4. **Apply safety margin** - Small tables (≤5 cols) keep all columns
5. **Render compact DDL** - Same multiline format as original

## Fallback Cases

The builder falls back to full schema when:

- SQL parsing is uncertain (`fallback_parse_error`)
- Compaction produces empty/invalid result (`fallback_empty_result`)
- SQL coverage validation fails (`fallback_sql_coverage`)

## Build Statistics

From latest build:
- Train: 6,380 compact, 7,654 full (349 parse fallbacks, 47 empty fallbacks)
- Dev: 297 compact, 368 full (9 parse fallbacks)
- Average schema reduction: 91.2%
- 228 SQL coverage fallbacks

## Rebuilding

```bash
cd data/training/t11
python build_t11.py [--dry-run] [--verbose]
```

Options:
- `--dry-run` - Process without writing output files
- `--verbose` - Print detailed progress
- `--no-validate` - Skip validation checks

## Prompt Contract

Same as T10:

```
System: You are an expert SQL assistant. Generate SQLite queries from natural language questions...

User: Schema:
<DDL>

Hints:
<hints or None>

Question:
<question>
```

## Key Constraints

- No planner stages or reasoning slots
- No relevant-tables prose or custom summary format
- Standard DDL format preserved
- Gold SQL labels unchanged
- Deterministic build (per-db alternating assignment)
- Exact identifier preservation (backticks, quotes)

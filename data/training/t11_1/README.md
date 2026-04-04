# T11.1 Mixed-Schema Dataset (Softer Compaction)

T11.1 is a corrected version of T11, fixing over-aggressive compaction (91% avg reduction → target 65-80%).

## Changes from T11

| Aspect | T11 | T11.1 |
|--------|-----|-------|
| Small-table threshold | ≤5 cols → keep all | **≤8 cols → keep all** |
| Question/hint extras | Up to 2 | **Up to 5 (normal) / 8 (widened)** |
| Primary-table bonus | None | **Up to 5/8 adjacent columns** |
| Min cols per table | None | **4 (normal) / 5 (widened)** |
| Over-compaction guards | None | **Reject >80% reduction, <300 chars on large schemas** |
| Re-widen retry | None | **Retry with wider margins before full fallback** |
| Compact assignment | 50/50 alternating | **Hash-based 55% (post-fallback ≈ 50%)** |
| Avg reduction | 91.2% | **Target 65-80%** |

## Design

**T11.1 = T10 + ~50% compact-schema variants (softer compaction)**

- Same T10 prompt contract preserved exactly
- Gold SQL targets unchanged
- Compact schemas are wider than T11 — more realistic, less oracle-minimal
- Re-widen retry saves more compact examples that would have fallen back to full in T11

## Re-widen Retry Flow

```
1. Build compact schema with normal T11.1 margins
2. Check over-compaction guards
3. If over-compacted → retry with widened margins:
   - extras: 5 → 8
   - primary-table bonus: 5 → 8
   - min cols per table: 4 → 5
4. If STILL over-compacted → fallback to full schema
```

## Files

- `train_t11_1.jsonl` - Training examples
- `dev_t11_1.jsonl` - Dev examples
- `build_t11_1.py` - Main builder script
- `t11_1_utils.py` - Utility functions
- `build_summary.json` - Build statistics

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
  "compaction_widened": true | false,
  "original_schema_len": 1234,
  "compact_schema_len": 567
}
```

## Rebuilding

```bash
cd data/training/t11_1
python build_t11_1.py [--dry-run] [--verbose]
```

Options:
- `--dry-run` - Process without writing output files
- `--verbose` - Print detailed progress
- `--no-validate` - Skip validation checks
- `--t10-dir PATH` - Override T10 dataset location
- `--output-dir PATH` - Override output directory

## Key Constraints

- No planner stages or reasoning slots
- No relevant-tables prose or custom summary format
- Standard DDL format preserved
- Gold SQL labels unchanged
- Deterministic build (hash-based assignment)
- Exact identifier preservation (backticks, quotes)

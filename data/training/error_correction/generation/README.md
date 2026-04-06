# SQL Error-Correction Dataset v1

## Overview

Supervised dataset for training SQL error-correction models.

**Task**: Given schema, question, hints, broken SQL, and optional error message,
output the corrected SQL query only.

## Statistics

| Split | Count |
|-------|-------|
| Internal Train | 14049 |
| Internal Dev | 1452 |
| Clean Train | 11813 |
| Clean Dev | 1230 |

## Files

### Main Datasets

- `train_error_repair_v1.jsonl` - Full training set (internal)
- `dev_error_repair_v1.jsonl` - Full dev set (internal)
- `train_error_repair_v1_clean.jsonl` - Benchmark-safe training set
- `dev_error_repair_v1_clean.jsonl` - Benchmark-safe dev set

### Reports

- `dataset_manifest.json` - Build configuration and statistics
- `family_summary.json` - Examples by database family
- `failure_type_summary.json` - Examples by failure type
- `source_mix_summary.json` - Examples by source type
- `schema_context_summary.json` - Examples by schema context type
- `contamination_report.json` - Contamination routing decisions
- `duplicate_report.json` - Deduplication statistics
- `verification_report.json` - Verification results
- `subagent_usage_report.json` - Subagent proposal usage and outcomes

### Additional Files

- `rejected_examples.jsonl` - Examples that failed validation
- `internal_only_examples.jsonl` - Internal-only examples before clean filtering
- `samples/` - Sample examples for inspection

## Example Format

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Schema:\n...\n\nHints:\n...\n\nQuestion:\n...\n\nBroken SQL:\n..."},
    {"role": "assistant", "content": "SELECT ..."}
  ],
  "metadata": {...}
}
```

## Contamination Policy

- **Internal datasets**: May contain examples derived from BIRD dev/eval
- **Clean datasets**: Only non-benchmark sources (BIRD train, Spider, custom)

Clean datasets are safe for use when evaluating on BIRD benchmark.

## Usage

```python
import json

with open('train_error_repair_v1_clean.jsonl') as f:
    for line in f:
        example = json.loads(line)
        messages = example['messages']
        metadata = example['metadata']
```

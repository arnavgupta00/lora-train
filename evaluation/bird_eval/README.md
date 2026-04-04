# BIRD Eval Pipeline: Full vs Compact Schema

A complete evaluation pipeline for comparing model performance with full-schema vs compact-schema prompting on the BIRD benchmark.

## Quick Start

```bash
# 1. Build compact-schema prompts (from existing full-schema prompts)
python evaluation/bird_eval/build_compact_prompts.py \
    --full_prompts_file data/training/t10/bird_dev_t10.jsonl \
    --output_dir evaluation/bird_eval/prompts

# 2a. Run predictions with FULL schema
python evaluation/bird_eval/predict_bird_eval.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --adapter_dir "./runs/my_adapter" \
    --prompts_file data/training/t10/bird_dev_t10.jsonl \
    --output_dir ./results/full \
    --mode full

# 2b. Run predictions with COMPACT schema
python evaluation/bird_eval/predict_bird_eval.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --adapter_dir "./runs/my_adapter" \
    --prompts_file evaluation/bird_eval/prompts/bird_eval_prompts_compact.jsonl \
    --output_dir ./results/compact \
    --mode compact

# 3a. Evaluate FULL schema predictions
python evaluation/bird_eval/evaluate_bird_eval.py \
    --predictions_file ./results/full/predictions_full.jsonl \
    --output_dir ./results/full/eval \
    --mode full

# 3b. Evaluate COMPACT schema predictions
python evaluation/bird_eval/evaluate_bird_eval.py \
    --predictions_file ./results/compact/predictions_compact.jsonl \
    --output_dir ./results/compact/eval \
    --mode compact

# 4. Compare full vs compact results
python evaluation/bird_eval/compare_results.py \
    --full_eval ./results/full/eval/eval_report_full.json \
    --compact_eval ./results/compact/eval/eval_report_compact.json \
    --output_dir ./results/comparison
```

---

## T10 Prompt Contract

All prompts follow the exact T10 format for consistency with training.

### System Prompt (exact, no modifications)

```
You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Copy table names and column names exactly from the schema.
Never invent normalized identifiers.
If an identifier contains spaces, punctuation, %, hyphens, slashes, or parentheses, use it exactly and wrap it in backticks.
Use only the tables and columns that exist in the schema.
Only output the SQL query, nothing else.
```

### User Prompt Structure

```
Schema:
{multiline_ddl_schema}

Hints:
{hint_text_or_None}

Question:
{question}
```

### Rules

- ✅ Always include `Hints:` line
- ✅ If no hint exists, write exactly: `Hints:\nNone`
- ✅ Schema must be multiline DDL (CREATE TABLE with proper indentation)
- ❌ Never include `/no_think` or `/think`
- ❌ Never use old prompt variants

---

## Directory Structure

```
evaluation/bird_eval/
├── __init__.py
├── compact_schema.py          # Schema compaction logic
├── build_compact_prompts.py   # Build compact prompts from existing full prompts
├── predict_bird_eval.py       # Run inference (full or compact mode)
├── evaluate_bird_eval.py      # Compute metrics
├── compare_results.py         # Full vs compact comparison
└── README.md                  # This file
```

---

## Pipeline Components

### 1. `compact_schema.py` - Schema Compaction

Compact schema extraction using **question + hints only** (no gold SQL leakage).

**Key Features:**

1. **Heuristic extraction** from question+hints only
   - Match table/column names in question and hints
   - Case-insensitive and fuzzy word matching
   - Identify primary table (most mentions)

2. **Primary-table bonus** - Most relevant table keeps 8+ extra columns

3. **Bridge-table preservation** - FK-connected tables between selected tables always included

4. **Stricter fallback thresholds** (target 65-80% reduction)
   - Compact < 400 chars AND original > 2000 chars → fallback
   - Reduction > 85% on schemas > 3000 chars → fallback
   - Any table < 3 columns → fallback

5. **Widen-once-before-full-fallback** - Retry with wider margins before full fallback

6. **Per-example metadata** - Every compact prompt includes:
   ```json
   {
     "compaction_metadata": {
       "original_schema_length": 5432,
       "compact_schema_length": 1876,
       "reduction_percent": 65.5,
       "primary_table": "frpm",
       "tables_kept": ["frpm", "schools"],
       "tables_dropped": ["satscores"],
       "compaction_status": "success",
       "fallback_reason": null,
       "pass_number": 1
     }
   }
   ```

### 2. `build_compact_prompts.py` - Build Compact Prompts

Generates compact-schema prompts from the existing full-schema prompts.

**Input:**
- `data/training/t10/bird_dev_t10.jsonl` - Existing full-schema prompts (1534 examples)

**Outputs:**
- `bird_eval_prompts_compact.jsonl` - Compact prompts with per-example metadata
- `compacting_summary.json` - Aggregate statistics

```bash
python evaluation/bird_eval/build_compact_prompts.py \
    --full_prompts_file data/training/t10/bird_dev_t10.jsonl \
    --output_dir evaluation/bird_eval/prompts
```

### 3. `predict_bird_eval.py` - Run Predictions

Generate SQL predictions using prebuilt prompts (full or compact).

**Features:**
- Batch inference with configurable generation parameters
- Raw output storage before normalization
- Generation config logging

**Outputs:**
- `predictions_{mode}.jsonl` - Normalized predictions
- `raw_outputs_{mode}.jsonl` - Raw model outputs
- `generation_config.json` - Exact parameters

```bash
# Full schema
python evaluation/bird_eval/predict_bird_eval.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --prompts_file data/training/t10/bird_dev_t10.jsonl \
    --output_dir ./results/full \
    --mode full

# Compact schema
python evaluation/bird_eval/predict_bird_eval.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --prompts_file evaluation/bird_eval/prompts/bird_eval_prompts_compact.jsonl \
    --output_dir ./results/compact \
    --mode compact
```

**Generation Arguments:**
- `--batch_size` - Batch size (default: 8)
- `--max_new_tokens` - Max tokens to generate (default: 256)
- `--temperature` - Sampling temperature (default: 0.0 = greedy)
- `--do_sample` - Enable sampling
- `--adapter_dir` - Path to LoRA adapter (optional)

### 4. `evaluate_bird_eval.py` - Evaluate Predictions

Compute metrics for SQL predictions against BIRD gold standard.

**Metrics:**
- **EX (Execution Accuracy)** - Primary metric
- **EM (Exact Match)**
- **Exec-fail rate** - Predictions that threw SQL errors
- **Wrong-result rate** - Predictions that executed but returned wrong results
- Per-difficulty breakdown (simple/moderate/challenging)
- Per-database breakdown
- Error categorization (syntax, column, table, etc.)

**Outputs:**
- `eval_report_{mode}.json` - Structured metrics
- `eval_summary_{mode}.md` - Human-readable markdown
- `per_example_results.jsonl` - Per-example details

```bash
python evaluation/bird_eval/evaluate_bird_eval.py \
    --predictions_file ./results/full/predictions_full.jsonl \
    --output_dir ./results/full/eval \
    --mode full
```

### 5. `compare_results.py` - Compare Full vs Compact

Compare evaluation results between full-schema and compact-schema modes.

**Features:**
- Side-by-side accuracy comparison
- Per-difficulty delta
- Per-database delta
- Per-example analysis (which examples each mode wins)

**Outputs:**
- `comparison_report.json` - Structured comparison
- `comparison_summary.md` - Human-readable summary

```bash
python evaluation/bird_eval/compare_results.py \
    --full_eval ./results/full/eval/eval_report_full.json \
    --compact_eval ./results/compact/eval/eval_report_compact.json \
    --output_dir ./results/comparison
```

---

## Output File Formats

### `bird_eval_prompts_compact.jsonl`

```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "What is the highest eligible...",
  "gold_sql": "SELECT ...",
  "difficulty": "simple",
  "evidence": "Eligible free rate...",
  "t10_prompt": {
    "system": "You are an expert SQL assistant...",
    "user": "Schema:\nCREATE TABLE...\n\nHints:\n...\n\nQuestion:\n..."
  },
  "compaction_metadata": {
    "original_schema_length": 5432,
    "compact_schema_length": 1876,
    "reduction_percent": 65.5,
    "primary_table": "frpm",
    "tables_kept": ["frpm", "schools"],
    "tables_dropped": ["satscores"],
    "compaction_status": "success",
    "fallback_reason": null,
    "pass_number": 1
  }
}
```

### `predictions_{mode}.jsonl`

```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "...",
  "predicted_sql": "SELECT ...",
  "gold_sql": "SELECT ...",
  "difficulty": "simple",
  "compaction_metadata": {...}  // Only for compact mode
}
```

### `eval_report_{mode}.json`

```json
{
  "predictions_file": "...",
  "mode": "full",
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total_examples": 1534,
    "execution_accuracy": 65.5,
    "execution_correct": 1005,
    "exact_match_accuracy": 30.2,
    "exec_fail_count": 150,
    "exec_fail_rate": 9.8,
    "wrong_result_count": 379,
    "wrong_result_rate": 24.7
  },
  "per_difficulty": {...},
  "per_database": [...],
  "error_categories": {...}
}
```

### `comparison_report.json`

```json
{
  "timestamp": "2024-01-15T10:35:00",
  "overall": {
    "full_execution_accuracy": 65.5,
    "compact_execution_accuracy": 66.2,
    "execution_accuracy_delta": 0.7,
    ...
  },
  "per_difficulty": {...},
  "per_database": [...],
  "example_analysis": {
    "both_correct": 950,
    "both_wrong": 450,
    "compact_wins_count": 55,
    "full_wins_count": 79,
    "net_delta": -24
  }
}
```

---

## Key Constraints

1. **No gold-SQL leakage** - Compact schema extraction uses only question + hints + full schema
2. **No planner/reranker/self-correction** - Single-pass inference only
3. **No SQL repair** - Predictions used as-is
4. **Deterministic** - Same inputs → same outputs
5. **Reproducible** - Generation config and git commit logged in manifest
6. **Auditable** - Per-example compaction metadata for traceability

---

## Data Sources

- **Full-schema prompts**: `data/training/t10/bird_dev_t10.jsonl` (perfected and tested)
- **BIRD dev questions**: `data/bird_eval_datasets/dev.json`
- **BIRD databases**: `data/bird_eval_datasets/dev_databases/`
- **Tied-append alternatives**: `data/bird_eval_datasets/dev_tied_append.json`

---

## Example Output Structure

After running the full pipeline:

```
evaluation/bird_eval/
├── prompts/
│   ├── bird_eval_prompts_compact.jsonl   # 1534 compact prompts
│   └── compacting_summary.json           # Compaction stats

results/
├── full/
│   ├── predictions_full.jsonl
│   ├── predictions_full.json
│   ├── raw_outputs_full.jsonl
│   ├── generation_config.json
│   └── eval/
│       ├── eval_report_full.json
│       ├── eval_summary_full.md
│       └── per_example_results.jsonl
├── compact/
│   ├── predictions_compact.jsonl
│   ├── predictions_compact.json
│   ├── raw_outputs_compact.jsonl
│   ├── generation_config.json
│   └── eval/
│       ├── eval_report_compact.json
│       ├── eval_summary_compact.md
│       └── per_example_results.jsonl
└── comparison/
    ├── comparison_report.json
    └── comparison_summary.md
```

---

## Troubleshooting

### ImportError: No module named 't10_utils'

Ensure you're running from the repository root or add the path:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/lm/data/training/t10
```

### Database not found errors

Check that the BIRD databases are extracted:
```bash
ls data/bird_eval_datasets/dev_databases/
```

### Low compact success rate

- Check `compacting_summary.json` for fallback reasons
- Common issues:
  - Question doesn't mention any schema identifiers
  - Schema is already small (all examples kept)
  - Over-compaction threshold triggered

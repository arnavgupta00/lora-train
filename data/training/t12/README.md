# T12 Training, Prediction, and Evaluation Pipeline

A complete T12-aligned baseline pipeline for text-to-SQL training and evaluation on the BIRD benchmark.

## Quick Start

```bash
# 1. Build prebuilt prompts for BIRD dev set (one-time)
python data/training/t12/build_eval_prompts.py \
    --bird_dev_json data/bird_eval_datasets/dev.json \
    --db_dir data/bird_eval_datasets/dev_databases \
    --output data/training/t12/bird_dev_t12.jsonl

# 2. Train on T12 data (using YAML config - recommended)
./data/training/t12/train_t12.sh \
    --config training/configs/t12_baseline_3090.yaml

# OR train with CLI arguments
./data/training/t12/train_t12.sh \
    --model_id "Qwen/Qwen3.5-2B" \
    --output_dir "./runs/t12_sft_001"

# 3. Generate predictions
python data/training/t12/predict_t12.py \
    --model_id "Qwen/Qwen3.5-2B" \
    --adapter_dir "./runs/t12_baseline_3090" \
    --prompts_file data/training/t12/bird_dev_t12.jsonl \
    --output_dir "./runs/t12_baseline_3090/predictions"

# 4. Evaluate predictions
python data/training/t12/evaluate_t12.py \
    --predictions_file ./runs/t12_baseline_3090/predictions/predictions_t12.jsonl \
    --output_dir ./runs/t12_baseline_3090/eval
```

---

## T12 Prompt Contract

All training, prediction, and evaluation prompts must follow this exact format.

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

## Pipeline Components

### 1. `t12_utils.py` - Shared Utilities

Core utilities used by all scripts:

```python
from t12_utils import (
    T12_SYSTEM_PROMPT,           # Exact system prompt constant
    build_t12_prompt,            # Build prompt from schema/hints/question
    validate_t12_prompt,         # Validate T12 compliance
    format_schema_multiline,     # Format schema as multiline DDL
    get_ddl_schema_from_db,      # Extract schema from SQLite database
    normalize_sql,               # Normalize model output
    check_prompt_parity,         # Verify eval prompt matches training
    get_t12_system_prompt_hash,  # Get hash for verification
)
```

### 2. `build_eval_prompts.py` - Prebuilt Prompts Generator

Generates `bird_dev_t12.jsonl` with pre-formatted T12 prompts for all 1534 BIRD dev examples.

**Features:**
- Extracts DDL schema from each SQLite database (cached)
- Maps BIRD `evidence` field → T12 `Hints:` (or `None`)
- Validates all prompts for T12 compliance
- **Prompt parity check**: Verifies eval system prompt matches training
- **SQL preservation audit**: Verifies gold_sql unchanged from dev.json

**Output format:**
```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "What is the highest eligible...",
  "gold_sql": "SELECT ...",
  "difficulty": "simple",
  "evidence": "Eligible free rate...",
  "t12_prompt": {
    "system": "You are an expert SQL assistant...",
    "user": "Schema:\nCREATE TABLE...\n\nHints:\n...\n\nQuestion:\n..."
  }
}
```

### 3. `train_t12.sh` - Training Launcher

Wrapper for `training/train_lora.py` with T12 defaults and enhanced logging.

**Features:**
- Pre-validates T12 prompt format before training
- **YAML config file support** (recommended for reproducibility)
- **Enhanced logging**: Loss and gradient norm logged every step
- Pre-configured paths to T12 data files
- Passes through all arguments to train_lora.py

**YAML Config Example (`training/configs/t12_baseline_3090.yaml`):**
```yaml
run_name: t12_baseline_3090
model_id: Qwen/Qwen3.5-2B
method: lora_sft

# Paths
train_jsonl: data/training/t12/train_t12.jsonl
dev_jsonl: data/training/t12/dev_t12.jsonl
output_dir: runs/t12_baseline_3090

# Precision
bf16: true
gradient_checkpointing: true
max_seq_length: 4096

# Batch size (effective = 2 * 16 = 32)
per_device_train_batch_size: 2
gradient_accumulation_steps: 16

# Learning rate
learning_rate: 1.5e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
max_grad_norm: 1.0

# Training
num_train_epochs: 2

# LoRA
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Logging
logging_steps: 10
eval_steps: 200
save_steps: 200
```

**Training with config:**
```bash
./train_t12.sh --config training/configs/t12_baseline_3090.yaml
```

**Training with CLI:**
```bash
./train_t12.sh \
    --model_id "Qwen/Qwen3.5-2B" \
    --output_dir "./runs/t12_sft_001" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --lora_r 16
```

**Logging Output:**
The training script now logs:
- `loss` - Training loss at each logging step
- `grad_norm` - Gradient norm (useful for detecting gradient explosions/vanishing)
- `learning_rate` - Current learning rate
- `epoch` - Current epoch

Example log output:
```
{'loss': 0.523, 'grad_norm': 0.847, 'learning_rate': 1.48e-4, 'epoch': 0.15, 'step': 100}
```

### 4. `predict_t12.py` - Prediction Script

Generates predictions using prebuilt T12 prompts.

**Features:**
- Loads prebuilt prompts (no schema extraction at inference)
- Batch inference with configurable generation parameters
- **Raw output storage**: Saves outputs before normalization
- **Generation config logging**: Exact parameters saved

**Output files:**
- `predictions_t12.jsonl` - Normalized predictions
- `predictions_t12.json` - JSON array format (for evaluate.py compatibility)
- `raw_outputs_t12.jsonl` - Raw model outputs
- `generation_config.json` - Exact generation parameters

### 5. `evaluate_t12.py` - Evaluation Script

Evaluates predictions against BIRD gold SQL.

**Features:**
- Reuses core logic from `data/bird_eval_datasets/evaluate.py`
- **Explicit exec-fail vs wrong-result split**
- Per-difficulty and per-database breakdown
- Error categorization
- **Run manifest** for reproducibility

**Metrics:**
- Execution Accuracy (EX) - primary metric
- Exact Match (EM)
- Exec Fail Count/Rate - predictions that threw SQL errors
- Wrong Result Count/Rate - predictions that executed but returned wrong results

**Output files:**
- `eval_report_t12.json` - Structured metrics
- `eval_summary_t12.md` - Human-readable markdown
- `run_manifest.json` - Reproducibility metadata
- `per_example_results.jsonl` - Per-example results

---

## Hints/Evidence Source

BIRD `dev.json` contains an `evidence` field for each question:

```json
{
  "question_id": 0,
  "db_id": "california_schools",
  "question": "What is the highest eligible free rate...",
  "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
  "SQL": "SELECT ..."
}
```

This `evidence` field maps to the T12 `Hints:` section:
- Non-empty evidence → `Hints:\n{evidence}`
- Empty evidence → `Hints:\nNone`

---

## Example Rendered Prompts

### Training Example

**System:**
```
You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Copy table names and column names exactly from the schema.
Never invent normalized identifiers.
If an identifier contains spaces, punctuation, %, hyphens, slashes, or parentheses, use it exactly and wrap it in backticks.
Use only the tables and columns that exist in the schema.
Only output the SQL query, nothing else.
```

**User:**
```
Schema:
CREATE TABLE frpm
(
    CDSCode TEXT not null primary key,
    `County Name` TEXT null,
    `Free Meal Count (K-12)` INTEGER null,
    `Enrollment (K-12)` INTEGER null
)

Hints:
Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?
```

**Assistant:**
```sql
SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1
```

### Eval Example (No Hints)

**User:**
```
Schema:
CREATE TABLE schools
(
    CDSCode TEXT not null primary key,
    School TEXT null,
    MailStreet TEXT null
)
CREATE TABLE frpm
(
    CDSCode TEXT not null primary key,
    `FRPM Count (K-12)` REAL null
)

Hints:
None

Question:
What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
```

---

## Output File Formats

### predictions_t12.jsonl

```json
{"question_id": 0, "db_id": "california_schools", "question": "...", "predicted_sql": "SELECT ...", "gold_sql": "SELECT ...", "difficulty": "simple"}
```

### eval_report_t12.json

```json
{
  "predictions_file": "...",
  "timestamp": "2024-01-15T12:30:00",
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

### run_manifest.json

```json
{
  "timestamp": "2024-01-15T12:35:00",
  "predictions_file": "...",
  "t12_system_prompt_hash": "a1b2c3d4e5f6g7h8",
  "git_commit": "abc123def456",
  "generation_config": {...},
  "summary": {...}
}
```

---

## What Was Reused vs Changed

### Reused (unchanged)

| Component | Source | Notes |
|-----------|--------|-------|
| Training logic | `training/train_lora.py` | Entire script, handles chat format natively |
| SQL execution | `evaluate.py` | `execute_sql()`, `results_match()` |
| Tied-append handling | `evaluate.py` | Alternative gold SQL matching |
| Schema formatting | `t12_transform.py` | `format_schema_multiline()` |

### Changed for T12

| Change | Old Behavior | T12 Behavior |
|--------|--------------|--------------|
| System prompt | Various versions | Single canonical T12 prompt |
| Hints handling | Optional, various formats | Always present, `None` if empty |
| `/no_think` | Added in eval prompts | Never included |
| Prompt building | At inference time | Prebuilt, validated |
| Error reporting | Single "errors" count | Split into exec-fail vs wrong-result |

---

## File Structure

```
data/training/t12/
├── train_t12.jsonl          # Training data (14,034 examples)
├── dev_t12.jsonl            # Dev data (665 examples)
├── bird_dev_t12.jsonl       # Prebuilt BIRD dev prompts (1,534 examples)
├── train_t12.sh             # Training launcher
├── build_eval_prompts.py    # Prebuilt prompts generator
├── predict_t12.py           # Prediction script
├── evaluate_t12.py          # Evaluation script
├── t12_utils.py             # Shared utilities
└── README.md                # This file
```

---

## Constraints

- ✅ Minimal and readable implementation
- ✅ Reuses prior working logic
- ✅ No compact schema, planner steps, reranking, or self-correction
- ✅ No hidden post-processing tricks
- ✅ Deterministic and reproducible
- ✅ Does not alter gold SQL or database content
- ✅ Does not redesign the benchmark

# Error Correction Evaluation Guide

## Overview

The error-correction evaluation implements a retry loop that can fix execution errors in generated SQL:

```
┌──────────────────────────────────────────────────────────────┐
│   Question + Schema                                          │
│         │                                                    │
│         ▼                                                    │
│   SFT Model (no_think) ──► Generate SQL                     │
│         │                                                    │
│         ▼                                                    │
│   Execute SQL                                                │
│         │                                                    │
│    ┌────┴────┐                                               │
│    ▼         ▼                                               │
│ SUCCESS    ERROR ──► Corrector (thinking) ──► Execute       │
│    │         │              │                    │           │
│    │         │              └────── Retry (max 3) ──────┘   │
│    │         │                                               │
│    └─────────┴──────────────► Final Result                  │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Test on 100 examples first:

```bash
python3 evaluation/eval_error_correction.py \
  --model_id Qwen/Qwen3-1.7B \
  --adapter_dir /workspace/outputs/qwen3-1.7b-sft-20260402_152110 \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_ec_test \
  --max_retries 3 \
  --limit 100
```

### Full evaluation:

```bash
python3 evaluation/eval_error_correction.py \
  --model_id Qwen/Qwen3-1.7B \
  --adapter_dir /workspace/outputs/qwen3-1.7b-sft-20260402_152110 \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_error_correction \
  --max_retries 3 \
  --initial_max_tokens 256 \
  --correction_max_tokens 1024 2>&1 | tee eval_ec.log
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | Required | Base model (e.g., `Qwen/Qwen3-1.7B`) |
| `--adapter_dir` | "" | LoRA adapter directory |
| `--bird_dev_json` | Required | Path to BIRD dev.json |
| `--db_dir` | Required | Path to BIRD databases |
| `--output_dir` | Required | Output directory |
| `--batch_size` | 16 | Batch size for initial generation |
| `--initial_max_tokens` | 256 | Tokens for initial gen (no_think) |
| `--correction_max_tokens` | 1024 | Tokens for correction (with thinking) |
| `--correction_batch_size` | 8 | Batch size for corrections |
| `--max_retries` | 3 | Max correction attempts per error |
| `--num_workers` | 4 | Parallel SQL execution workers |
| `--limit` | 0 | Limit examples (0=all) |

## How It Works

### Phase 1: Initial Generation
- Uses `/no_think` mode for fast generation
- Batch size: 16 (configurable)
- Max tokens: 256 (just SQL, no reasoning)
- Speed: ~30 examples/minute

### Phase 2: Initial Evaluation
- Executes all generated SQL in parallel
- Categorizes errors:
  - `column_not_found` - wrong column name
  - `table_not_found` - wrong table name
  - `syntax_error` - general syntax issues
  - `syntax_near` - near "X" errors
  - `unrecognized_token` - backtick issues
  - `ambiguous_column` - needs table alias

### Phase 3: Error Correction Loop
- Uses `/think` mode with chain-of-thought
- Max tokens: 1024 (reasoning + SQL)
- Batched generation for efficiency
- Retries up to 3 times per error

### Correction Prompt
The corrector uses a detailed prompt that includes:
1. Database schema
2. Original question
3. Failed SQL
4. Error message
5. Step-by-step analysis instructions

## Output Files

| File | Description |
|------|-------------|
| `evaluation_report.json` | Summary metrics and samples |
| `predictions.json` | Initial and final SQL for each example |
| `full_results.json` | Complete results with correction attempts |

## Example Report

```json
{
  "total_examples": 1534,
  "initial_execution_accuracy": 34.75,
  "final_execution_accuracy": 48.5,
  "accuracy_improvement": 13.75,
  
  "correction_stats": {
    "errors_attempted": 428,
    "fixed_at_attempt_1": 150,
    "fixed_at_attempt_2": 45,
    "fixed_at_attempt_3": 15,
    "fixed_total": 210,
    "unfixable": 218
  },
  
  "error_categories": {
    "column_not_found": 245,
    "syntax_error": 98,
    "syntax_near": 45,
    "table_not_found": 32,
    "other": 8
  }
}
```

## Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Accuracy | 34.75% | 45-52% | +10-17% |
| Errors | 27.9% | 8-12% | -16-20% |

## Timing

- Initial generation: ~50 minutes (1534 examples)
- Correction loop: ~45-90 minutes (~430 errors × 3 retries)
- **Total: ~95-140 minutes**

## Tips

1. **Always test on subset first**: Use `--limit 100` to verify everything works

2. **Monitor progress**: The script logs progress every batch

3. **Check memory**: Needs ~8-10GB VRAM

4. **View corrections**: Check `sample_corrections` in report to see examples of fixed SQL

5. **Analyze unfixed**: Check `sample_unfixed` to understand what errors are truly unfixable

## BIRD Benchmark Compliance

Using error correction is **fully allowed** by BIRD benchmark rules:
- Uses only execution error messages (no gold SQL)
- Many leaderboard systems use similar techniques (DIN-SQL, C3)
- Standard practice in text-to-SQL research

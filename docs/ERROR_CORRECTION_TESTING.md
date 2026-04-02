# Error Correction Testing Script

## Overview

This script runs **3 sequential tests** to find the best error-correction configuration for BIRD benchmark SQL generation.

## Tests

| # | Configuration | Initial Gen | Correction | Purpose |
|---|---------------|-------------|------------|---------|
| 1 | **SFT LoRA** | no_think, 256 tokens | no_think, 512 tokens | Test fine-tuned model |
| 2 | **Base Model** | no_think, 256 tokens | no_think, 512 tokens | Test untrained model |
| 3 | **Base + Thinking** | thinking, 512 tokens | thinking, 1024 tokens | Test chain-of-thought |

## Usage

```bash
# Quick test on 50 examples (~10-15 minutes)
bash scripts/test_error_correction.sh --limit 50

# Full evaluation on all 1534 examples (~2-3 hours)
bash scripts/test_error_correction.sh --full

# Custom limit
bash scripts/test_error_correction.sh --limit 100
```

## Output

Results are saved to timestamped directory:
```
eval_ec_comparison_YYYYMMDD_HHMMSS/
├── test1_sft_lora/
│   ├── evaluation_report.json
│   ├── predictions.json
│   └── full_results.json
├── test2_base_nothink/
│   └── ...
├── test3_base_thinking/
│   └── ...
├── test1_sft_lora.log
├── test2_base_nothink.log
└── test3_base_thinking.log
```

The script prints a comparison table at the end:

```
| Test | Config | Initial Acc | Final Acc | Improvement | Fixed |
|------|--------|-------------|-----------|-------------|-------|
| test1_sft_lora | 34.7% | 45.2% | +10.5% | 142 |
| test2_base_nothink | 12.3% | 18.5% | +6.2% | 85 |
| test3_base_thinking | 15.1% | 28.4% | +13.3% | 178 |
```

## What to Look For

### Good Signs
- `Improvement` > 10%
- `Fixed` count > 100 (out of ~430 errors)
- Debug logs show different SQL in corrections

### Bad Signs
- `Improvement` = 0%
- `Fixed` = 0
- Debug logs show identical SQL (not correcting)

## Expected Results

Based on initial tests:
- **Test 1 (SFT LoRA)**: May struggle if model wasn't trained with error correction
- **Test 2 (Base no_think)**: Lower initial accuracy but may correct better
- **Test 3 (Base thinking)**: Likely best correction ability with chain-of-thought

## Debugging

Each test logs debug output for first 5 corrections:
```
[DEBUG] Correction raw output #1:
  Failed SQL: SELECT T2.MailingStreet FROM ...
  Error: no such column: T2.MailingStreet
  Raw output: SELECT T2.MailStreet FROM ...
```

Check these logs to see if model is actually changing the SQL.

## Cloud GPU Setup

```bash
# On RunPod/VastAI instance
cd /workspace/lora-train
git pull origin main

# Run tests
bash scripts/test_error_correction.sh --limit 50

# Copy results back
# (from local machine)
scp -r root@<IP>:/workspace/lora-train/eval_ec_comparison_* \
  ./results/qwen3-1.7b/v1/
```

## Files Modified

- `evaluation/eval_error_correction.py` - Main evaluation script
- `scripts/test_error_correction.sh` - This testing script
- `docs/ERROR_CORRECTION_GUIDE.md` - Usage guide

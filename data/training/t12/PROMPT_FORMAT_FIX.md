# T12 Prompt Format Fix

## Date
April 6, 2026

## Problem
Training validation failed when launching T12 training due to T12 additions not matching the T12 prompt contract.

## Issues Found

### 1. System Prompt Format (All 2,048 T12 additions)
**Problem**: System prompt contained `\n` escape sequences instead of actual newlines.

**Example (broken)**:
```
You are an expert SQL assistant. Generate SQLite queries from natural language questions.\nGiven a database schema and a question, generate the correct SQL query.\n...
```

**Expected**:
```
You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
...
```

**Root Cause**: When generating T12 examples, the system prompt was being serialized with escaped newlines instead of preserving the multi-line format.

### 2. Missing Hints Section (110 out of 2,048 examples)
**Problem**: User prompts missing the required "Hints:" section.

**Example (broken)**:
```
Schema:
CREATE TABLE ...

Question:
How many users are there?
```

**Expected**:
```
Schema:
CREATE TABLE ...

Hints:
None

Question:
How many users are there?
```

**Root Cause**: Some generation passes created examples without the Hints section, which is required by the T12 contract.

## Solution

### Script Created: `fix_t12_prompts.py`

The script processes all T12 additions (lines 14035-16082) and:

1. **Fixes system prompts**:
   - Replaces `\n` with actual newlines
   - Validates against canonical `T12_SYSTEM_PROMPT`

2. **Fixes user prompts**:
   - Inserts `Hints:\nNone\n\n` before `Question:` if missing
   - Preserves all other prompt structure

3. **Validates**:
   - Uses `validate_t12_messages()` from `t12_utils.py`
   - Ensures all examples match T12 contract

### Execution

```bash
cd /Users/arnav/programming/lm/data/training/t12
python3 fix_t12_prompts.py
mv train_t12_fixed.jsonl train_t12.jsonl
```

## Results

- ✅ All 16,082 examples now pass T12 validation
- ✅ System prompts match canonical T12_SYSTEM_PROMPT exactly
- ✅ All user prompts have required Schema → Hints → Question structure
- ✅ Backbone examples (14,034) unchanged (already correct)
- ✅ T12 additions (2,048) all fixed

## Validation

Final validation confirms:
```
T12 Validation PASSED - all prompts follow T12 contract
  - Backbone: 14,034 examples
  - T12 additions: 2,048 examples
  Total: 16,082 examples
```

## Impact

- **Before**: Training would fail validation immediately
- **After**: Training can proceed with properly formatted prompts
- **Data Quality**: No semantic changes to examples, only format fixes
- **Benchmark Cleanliness**: Unchanged (all T12 additions still on synthetic non-eval schemas)

## Files Modified

- `train_t12.jsonl` - Fixed in place (original backed up in git history)

## Files Created

- `fix_t12_prompts.py` - One-time fix script (kept for documentation)
- `PROMPT_FORMAT_FIX.md` - This document

## Next Steps

Training can now proceed:
```bash
bash data/training/t12/train_t12.sh --config training/configs/t12_baseline_3090.yaml
```

# SQL Error-Correction Dataset Build Status

## Build Information
- **Version**: v1
- **Size**: B (target 18k train, 2k dev)
- **Started**: 2026-04-06 07:27:16
- **Status**: IN PROGRESS

## Progress Summary

### Phase 1: Real Failure Ingestion ✅ COMPLETE
- **Total Ingested**: 1,126 real failures
- **Source**: T10 and T11_1 eval failures
- **Checkpoint Files**: 
  - `checkpoint_after_real_internal_v1.jsonl` (1,126 examples)
  - Saved to disk successfully

### Phase 2: Synthetic Generation 🔄 IN PROGRESS  
- **Target**: ~10,800 examples (60% of 18k)
- **Current Progress**: 1,000+ examples generated
- **Checkpoint Files**:
  - `progress_synthetic_internal_v1.jsonl` (updating every 500 examples)
- **Note**: Synthetic generation is CPU-intensive, ~20 minutes per 500 examples

### Phase 3-5: Pending
- Deduplication
- Train/Dev Splitting  
- Final Export

## Files Created

### Checkpoints (Incremental Saves)
```
checkpoint_after_real_internal_v1.jsonl       (1,126 examples)
progress_real_internal_v1.jsonl              (1,100 examples)
progress_synthetic_internal_v1.jsonl         (updating)
```

### Example Format
Each example is SFT-ready with:
- System prompt (SQL repair assistant)
- User prompt (schema, question, hints, broken SQL, error)
- Assistant response (corrected SQL only)
- Rich metadata (failure type, contamination source, pool, verification)

### Sample Example
```json
{
  "messages": [...],
  "metadata": {
    "example_id": "real_t10_california_schools_0",
    "source_type": "real_failure",
    "failure_type": "wrong_return_field",
    "contamination_source": "bird_dev_direct",
    "benchmark_clean": false,
    "pool": "internal",
    "verification_passed": true
  }
}
```

## Expected Final Output

Once build completes:
```
data/training/error_correction/
├── train_error_repair_v1.jsonl          (~18k internal)
├── dev_error_repair_v1.jsonl            (~2k internal)
├── train_error_repair_v1_clean.jsonl    (benchmark-safe subset)
├── dev_error_repair_v1_clean.jsonl      (benchmark-safe subset)
├── dataset_manifest.json
├── family_summary.json
├── failure_type_summary.json
├── contamination_report.json
└── samples/
    ├── real_failures_examples.jsonl
    └── synthetic_examples.jsonl
```

## To Monitor Progress

```bash
# Check log
tail -f build.log

# Check line counts
wc -l *_v1.jsonl

# Check process
ps aux | grep build_dataset
```

## To Resume if Interrupted

The build saves progress incrementally, so if interrupted:
1. Checkpoints are already saved
2. Can manually combine files if needed
3. Or restart build (will regenerate)

## Performance Notes

- Real failure ingestion: ~7 minutes (1,126 examples)
- Synthetic generation: ~20 min per 500 examples  
- Full B dataset build: Estimated 3-4 hours total
- Incremental saves every 100 (real) / 500 (synthetic) examples

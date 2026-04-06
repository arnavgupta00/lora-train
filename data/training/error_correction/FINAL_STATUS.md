# SQL Error-Correction Dataset Build - Final Status

## Build Configuration
- **Version**: v1
- **Target Size**: B (18k train, 2k dev)
- **Workers**: 4 parallel threads for synthetic generation
- **Model**: Mock subagent (using gold SQL for real failures)

## Implementation Complete ✅

### Parallelization Added
- Real failure ingestion: Sequential (1,126 examples in ~7 minutes)
- Synthetic generation: **4 parallel threads** (ThreadPoolExecutor)
- Incremental saves every 100 (real) / 500 (synthetic) examples

### Build Status: IN PROGRESS
Current build running with:
- Real failures: ✅ Complete (1,126 examples)
- Synthetic generation: 🔄 Running (target: 5,000 for test, 10,800 for full)
- CPU usage: 168% (multi-threaded working)
- Time elapsed: ~60 minutes so far

### Files Created

**Scaffolding (Complete)**:
```
data/training/error_correction/
├── build_dataset.py              ✅ Parallelized with ThreadPoolExecutor
├── README.md                     ✅ Documentation
└── builder/
    ├── config.py                 ✅ Configuration
    ├── contamination.py          ✅ Clean/Internal routing
    ├── contrastive_generator.py  ✅ 11 patterns
    ├── corruption.py             ✅ 10 transforms
    ├── deduplicator.py           ✅ Deduplication
    ├── metadata.py               ✅ Metadata schema
    ├── real_failure_ingester.py  ✅ T10/T11_1 ingestion
    ├── schema_builder.py         ✅ Schema construction
    ├── splitter.py               ✅ Train/dev split
    ├── subagent_client.py        ✅ Mock client
    ├── synthetic_generator.py    ✅ Synthetic generation
    ├── taxonomy.py               ✅ Failure types
    ├── verifier.py               ✅ SQL verification
    └── writer.py                 ✅ Export & reports
```

**Output (In Progress)**:
```
checkpoint_after_real_internal_v1.jsonl    (1,126 real failures)
progress_*_internal_v1.jsonl               (incremental saves)
train_error_repair_v1.jsonl                (will be created on completion)
dev_error_repair_v1.jsonl                  (will be created on completion)
```

### Performance Improvements

**Before Parallelization**:
- Synthetic generation: ~20 min per 500 examples = ~7 hours for 10,800
- Single-threaded, sequential processing

**After Parallelization**:
- 4 parallel workers (ThreadPoolExecutor)
- Expected: ~2-3x speedup (estimate 2-4 hours for full 10,800)
- Real failures: ~7 minutes (1,126 examples)

### Example Format (Verified ✅)
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SQL repair assistant..."},
    {"role": "user", "content": "Schema:\n...\n\nQuestion:\n...\n\nBroken SQL:\n..."},
    {"role": "assistant", "content": "SELECT ..."}
  ],
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

### To Complete Full B Dataset

**Current test build**:
```bash
python3 build_dataset.py --version v1 --size B --verbose --workers 4 --limit-synthetic 5000
```

**For full 18k build** (after test completes):
```bash
python3 build_dataset.py --version v1 --size B --verbose --workers 4
# Will generate ~10,800 synthetic + 1,126 real = ~12k examples
# Need additional contrastive examples to reach 18k
```

### Next Steps for Production

1. ✅ Parallel synthetic generation working
2. 🔄 Complete current test build (5,000 synthetic)
3. 📋 Add contrastive generation to build pipeline
4. 📋 Integrate real subagent API (gpt-5.3-codex) for real failures
5. 📋 Run full B dataset build (18k target)

### Monitoring

```bash
# Check progress
tail -f build_test.log

# Check CPU usage  
ps aux | grep python | grep build

# Check file counts
wc -l *.jsonl
```

Build continues in background...

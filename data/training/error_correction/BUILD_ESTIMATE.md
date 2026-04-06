# Full B Dataset Build - Time Estimate

## Command
```bash
python3 build_dataset.py --version v1 --size B --verbose --workers 4
```

## Performance Data (Measured)

### Phase 1: Real Failure Ingestion
- **Examples**: 1,126 real failures (T10 + T11_1)
- **Time**: ~7 minutes
- **Processing rate**: ~160 examples/minute
- **Method**: Sequential (no parallelization needed)

### Phase 2: Synthetic Generation  
- **Target**: 10,800 examples (60% of 18k)
- **Workers**: 4 parallel threads (ThreadPoolExecutor)
- **Measured performance**: 
  - Without parallelization: ~20 min per 500 examples
  - With 4 workers: ~8-10 min per 500 examples (observed)
  - **Speedup**: ~2-2.5x

### Phase 3: Deduplication
- **Examples**: ~12k total (1,126 real + 10,800 synthetic)
- **Time**: ~2-3 minutes (fast in-memory operations)

### Phase 4: Train/Dev Split
- **Time**: <1 minute (stratified sampling)

### Phase 5: Export & Reports
- **Time**: ~2-3 minutes (writing JSONL + generating reports)

## Total Estimated Time

| Phase | Time Estimate | Notes |
|-------|--------------|-------|
| Phase 1: Real Failures | 7 min | 1,126 examples |
| Phase 2: Synthetic | **180-220 min** | 10,800 examples, 4 workers |
| Phase 3: Deduplication | 3 min | In-memory |
| Phase 4: Split | 1 min | Stratified |
| Phase 5: Export | 3 min | JSONL + reports |
| **TOTAL** | **~195-235 minutes** | **3.25 - 3.9 hours** |

### Best Case Scenario
- **3 hours 15 minutes** (if workers achieve 2.5x speedup)

### Realistic Scenario
- **3 hours 30 minutes** (average speedup of 2.2x)

### Conservative Scenario  
- **3 hours 54 minutes** (with some overhead)

## Expected Output

### Dataset Sizes
- **Internal Train**: ~10,800 examples (90% of ~12k after dedup)
- **Internal Dev**: ~1,200 examples (10% of ~12k)
- **Clean Train**: 0 examples (all BIRD dev-derived → internal only)
- **Clean Dev**: 0 examples

*Note: Full 18k target requires adding contrastive examples (~6k more)*

### Output Files
```
data/training/error_correction/
├── train_error_repair_v1.jsonl          (~10,800 examples, ~30-35MB)
├── dev_error_repair_v1.jsonl            (~1,200 examples, ~3-4MB)
├── train_error_repair_v1_clean.jsonl    (0 examples)
├── dev_error_repair_v1_clean.jsonl      (0 examples)
├── dataset_manifest.json
├── family_summary.json
├── failure_type_summary.json
├── contamination_report.json
├── deduplication_report.json
├── verification_report.json
└── samples/
    ├── real_failures_examples.jsonl     (100 samples)
    └── synthetic_examples.jsonl         (100 samples)
```

## Monitoring During Build

### Real-time Progress
```bash
# Monitor log output
tail -f build_v1.log

# Check CPU usage (should be 150-200% with 4 workers)
ps aux | grep python | grep build

# Count generated examples
wc -l checkpoint_*.jsonl progress_*.jsonl 2>/dev/null
```

### Expected Log Output
```
[HH:MM:SS] Initializing components...
[HH:MM:SS] Phase 1: Ingesting real failures...
[HH:MM:SS]   Progress: 100 real failures processed, saving checkpoint...
...
[HH:MM:SS]   Ingested 1126 real failures
[HH:MM:SS] Phase 2: Generating synthetic examples using 4 workers...
[HH:MM:SS]   Target: 10800 synthetic examples
[HH:MM:SS]   Split into 8 batches across 4 workers
[HH:MM:SS]   Batch 1/8 complete: 1350 examples (1350/10800 total)
[HH:MM:SS]   Batch 2/8 complete: 1350 examples (2700/10800 total)
...
[HH:MM:SS]   Progress: 5000 synthetic examples generated, saving checkpoint...
...
[HH:MM:SS]   Generated 10800 synthetic examples
[HH:MM:SS] Phase 3: Deduplicating...
[HH:MM:SS]   Clean: 0, Internal: 11900
[HH:MM:SS] Phase 4: Splitting train/dev...
[HH:MM:SS]   Internal: 10710 train, 1190 dev
[HH:MM:SS] Phase 5: Writing datasets...
[HH:MM:SS]   Wrote 15 files
[HH:MM:SS] Build complete in 205.3 minutes
```

## Performance Bottleneck

**Synthetic generation** is the main bottleneck:
- SQL corruption transforms require:
  - SQL parsing and manipulation
  - Schema loading and validation
  - Execution verification (run broken SQL and corrected SQL)
  - Result comparison
- Each example takes ~1-2 seconds to generate and verify
- 10,800 examples × 1.5 sec ÷ 4 workers ÷ 60 = ~68 minutes minimum

## Recommendations

### For Fastest Build
1. **Use 4-8 workers** (tested: 4 workers work well)
2. **Run on machine with good CPU** (synthetic generation is CPU-bound)
3. **Use SSD** (for database reads during validation)

### For Full 18k Dataset
After this build completes, add contrastive examples:
```bash
# Would need to implement contrastive generation in pipeline
# Estimated additional time: +60-90 minutes for ~6k contrastive examples
```

## Ready to Run

Everything is set up. Clean slate ready for production build:

```bash
cd /Users/arnav/programming/lm/data/training/error_correction
python3 build_dataset.py --version v1 --size B --verbose --workers 4 > build_v1.log 2>&1 &

# Monitor
tail -f build_v1.log
```

Estimated completion: **~3.5 hours from start**

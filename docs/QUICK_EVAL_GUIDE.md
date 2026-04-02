# Quick Evaluation Progress Guide

## Current Situation

You're running evaluations on a remote RTX 3090 and want to see:
- Download progress (speed, ETA)
- Generation progress (how many SQL generated, speed)
- Evaluation progress (how many predictions tested)

## The Solution

### Option 1: Monitor Live in Terminal (RECOMMENDED)

Stop the background process and run in foreground with real-time visibility:

```bash
# Kill old background eval if still running
pkill -f eval_bird.py

# Run new eval with visible progress
python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_1_base

# You'll see real-time progress:
# Download: [████████░░] 2.5GB/3.09GB [00:45<01:15, 37.1MB/s]
# Generation: [145/350] 41.4% - 2.8 ex/s - ETA: 1.4min
# Evaluation: [350/350] 100.0% - 17.5 ex/s - ETA: 0.0min
```

### Option 2: Background with Logging (for long evals)

If you need to disconnect, use `nohup` + `tee`:

```bash
nohup python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_1_base 2>&1 | tee eval_1_base.log &

# Monitor progress from another terminal:
tail -f eval_1_base.log | grep -E "\[.*\].*%"
```

### Option 3: Monitor Existing Process

If eval is already running in background:

```bash
# Check which evals are running
ps aux | grep eval_bird | grep -v grep

# Monitor progress live
tail -f eval_1_base.log | grep -E "\[.*\].*%|Loading|complete"
```

## What Progress Looks Like

### In Terminal (with tee):
```
2026-04-02 17:36:21 [INFO] Loading model: Qwen/Qwen2.5-1.5B-Instruct
2026-04-02 17:36:21 [INFO] Downloading model weights (this may take a few minutes)...

[████████░░] 2.5GB/3.09GB [00:45<01:15, 37.1MB/s]  ← Download progress (terminal only)

2026-04-02 17:37:19 [INFO] Model weights loaded successfully
2026-04-02 17:37:19 [INFO] Generating SQL (batch_size=16)...
2026-04-02 17:37:35 [INFO] [32/350] 9.1% - 2.2 ex/s - ETA: 2.4min
2026-04-02 17:37:55 [INFO] [64/350] 18.3% - 2.5 ex/s - ETA: 1.9min
...
2026-04-02 17:39:45 [INFO] [350/350] 100.0% - 2.8 ex/s - ETA: 0.0min
2026-04-02 17:39:45 [INFO] Generation complete: 2.4 min (2.8 ex/s)
2026-04-02 17:40:05 [INFO] [350/350] 100.0% - 17.5 ex/s - ETA: 0.0min
2026-04-02 17:40:05 [INFO] Evaluation complete: 0.3min (17.5 ex/s)
```

## Expected Duration

For 350 BIRD examples on RTX 3090:
- **Model download**: 2-5 minutes (shown in terminal)
- **Generation phase**: 2-3 minutes (shown in logs at 10% intervals)
- **Evaluation phase**: 20-30 seconds (shown in logs at 10% intervals)
- **Total**: 5-8 minutes per evaluation

## Key Metrics to Watch

- **[N/total] percentage%**: Progress through examples
- **X.X ex/s**: Speed (examples per second)
- **ETA: X.Xmin**: Estimated time remaining

## Parallel Evaluations

After base model finishes (~8 min), start SFT evaluation:

```bash
# Terminal 1: Base model (started first)
python3 evaluation/eval_bird.py --model_id Qwen/Qwen2.5-1.5B-Instruct ...

# Terminal 2: SFT (started after base starts downloading)
python3 evaluation/eval_bird.py --adapter_dir ./outputs/sft_adapter/ --model_id Qwen/Qwen2.5-1.5B-Instruct ...
```

Both run in parallel, total time: ~15 minutes instead of 16+ minutes.

## Troubleshooting

**"I don't see any progress output"**
→ Eval is likely downloading model. Check terminal size. Wait 2-5 minutes.

**"tail -f shows old logs from previous run"**
→ Check filename. Each eval needs unique `--output_dir` to avoid overwriting results.

**"Process hung during download"**
→ Check internet connection and disk space: `df -h` and `nvidia-smi` for memory.

## See Also

- Full guide: `docs/EVAL_PROGRESS_MONITORING.md`
- Training guide: `docs/TRAINING_GUIDE.md`

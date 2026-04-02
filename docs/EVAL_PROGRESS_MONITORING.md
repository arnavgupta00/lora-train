# Evaluation Progress Monitoring

This document explains how to properly monitor evaluation progress, including download speeds and generation rates.

## The Issue: Progress Bars in Log Files

When running evaluations with output redirection (`> eval.log 2>&1`), you may see:
- HTTP request logs (not progress bars)
- No visible download speed or progress
- Silent model loading and generation phases

**Why?** HuggingFace progress bars are **interactive terminal features** that don't serialize to plain text log files.

## Solution 1: Monitor in Real-Time Terminal (Recommended)

SSH into your remote machine and run the evaluation WITHOUT background redirection:

```bash
# Terminal 1: Run evaluation with visible output
python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen3-1.7B \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_1_base

# You'll see:
# - [████████░░] 2.5GB/3.09GB [00:45<01:15, 37.1MB/s]  (HF download progress)
# - [145/350] 41.4% - 2.8 ex/s - ETA: 1.4min  (generation progress)
# - [350/350] 100.0% - 2.8 ex/s - ETA: 0.0min  (evaluation progress)
```

## Solution 2: Use `tee` to Show AND Log

If you need both terminal visibility AND persistent logs:

```bash
# Run evaluation with tee (shows output and saves to file)
python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen3-1.7B \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_1_base 2>&1 | tee eval_1_base.log

# Or background it with nohup+tee:
nohup python3 evaluation/eval_bird.py ... 2>&1 | tee eval_1_base.log &

# Monitor progress in real-time from another terminal:
tail -f eval_1_base.log | grep -E "\[|%|ETA"
```

## Solution 3: Monitor via Background Process

If evaluations are already running in background:

```bash
# Check current evaluation processes
ps aux | grep eval_bird

# Monitor a specific evaluation's progress
tail -f eval_1_base.log | grep -E "\[.*\].*%"
```

Expected log output (these WILL show in log files):

```
2026-04-02 17:36:21 [INFO] Loading model: Qwen/Qwen3-1.7B
2026-04-02 17:36:21 [INFO] Downloading model weights (this may take a few minutes)...
2026-04-02 17:37:15 [INFO] Model weights loaded successfully
2026-04-02 17:37:16 [INFO] Evaluating 350 examples...
2026-04-02 17:37:17 [INFO] Pre-caching database schemas...
2026-04-02 17:37:19 [INFO] Cached 146 schemas
2026-04-02 17:37:19 [INFO] Generating SQL (batch_size=16)...
2026-04-02 17:37:35 [INFO] [32/350] 9.1% - 2.2 ex/s - ETA: 2.4min
2026-04-02 17:37:55 [INFO] [64/350] 18.3% - 2.5 ex/s - ETA: 1.9min
2026-04-02 17:38:15 [INFO] [96/350] 27.4% - 2.6 ex/s - ETA: 1.6min
...
2026-04-02 17:39:45 [INFO] [350/350] 100.0% - 2.8 ex/s - ETA: 0.0min
2026-04-02 17:39:45 [INFO] Generation complete: 2.4 min (2.8 ex/s)
2026-04-02 17:39:46 [INFO] Evaluating predictions (workers=4)...
2026-04-02 17:40:05 [INFO] [350/350] 100.0% - 17.5 ex/s - ETA: 0.0min
2026-04-02 17:40:05 [INFO] Evaluation complete: 0.3min (17.5 ex/s)
```

## Key Progress Metrics

### Download Phase
- **Line format**: `[████████░░] 2.5GB/3.09GB [00:45<01:15, 37.1MB/s]`
- **Visible in**: Terminal only (not log files)
- **Expected duration**: 2-5 minutes for 1.7B model (~3GB weights)

### Generation Phase (Creating SQL)
- **Line format**: `[32/350] 9.1% - 2.2 ex/s - ETA: 2.4min`
- **Visible in**: Log files ✅
- **Expected duration**: 2-4 minutes for 350 examples

### Evaluation Phase (Running SQL)
- **Line format**: `[350/350] 100.0% - 17.5 ex/s - ETA: 0.0min`
- **Visible in**: Log files ✅
- **Expected duration**: 0.3-0.5 minutes for 350 examples

## Troubleshooting

**Problem**: "I see HTTP request logs but no progress"
- **Cause**: HuggingFace progress bars aren't rendering in your log viewer
- **Solution**: Use `tail -f eval.log` or connect interactively to terminal

**Problem**: "Evaluation is silent for 10+ minutes"
- **Cause**: Model is downloading in background, not visible in logs
- **Solution**: Check free disk space with `df -h`, ensure HF token is set (`export HF_TOKEN=...`)

**Problem**: "tail -f shows nothing new"
- **Cause**: Process isn't running or log file isn't being written
- **Solution**: 
  ```bash
  # Check if process is running
  ps aux | grep eval_bird | grep -v grep
  
  # Check if log file exists and is growing
  ls -lh eval_*.log
  tail eval_*.log
  ```

## Best Practices

1. **For single evaluation**: Run in foreground with real-time terminal visibility
2. **For parallel evaluations**: Use `tee` to capture logs while seeing progress
3. **Monitor generation only**: `tail -f eval.log | grep -E "%|ex/s"`
4. **Monitor memory during eval**: Open another terminal: `watch -n 2 nvidia-smi`

## Parallel Evaluation Commands

```bash
# Terminal 1: Base model evaluation
python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen3-1.7B \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_1_base 2>&1 | tee eval_1_base.log &

# Terminal 2: SFT model evaluation (after base starts)
python3 evaluation/eval_bird.py \
    --model_id Qwen/Qwen3-1.7B \
    --adapter_dir ./outputs/sft_adapter/ \
    --bird_dev_json ./bird_eval/dev.json \
    --db_dir ./bird_eval/dev_databases \
    --output_dir ./eval_2_sft 2>&1 | tee eval_2_sft.log &

# Monitor both in another terminal
watch -n 5 "tail -2 eval_*.log"
```

## Summary

- **Progress bars with speed/ETA**: Visible in terminal in real-time ✅
- **Generation/evaluation logs**: Saved to log files ✅
- **HTTP request details**: Only for debugging (ignored in normal usage)

For the best experience, **use `tee` to capture logs while monitoring in real-time**.

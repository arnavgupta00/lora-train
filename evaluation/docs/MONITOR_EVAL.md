# Monitor Eval Progress

## Real-time log output

```bash
tail -f /workspace/lora-train/eval_lora.log
```

Watch for lines like:
```
Generating SQL for batch...
Processing batch 1/767...
SQL generated: SELECT * FROM...
Validating execution...
```

## Check GPU usage (real-time)

```bash
watch -n 1 nvidia-smi
```

Shows memory usage, GPU%, temp. Exit with `Ctrl+C`.

## Check if process is still running

```bash
ps aux | grep eval_exec
```

If you see the process, it's still running.

## Check current progress (quick check)

```bash
# Count lines in predictions file (number of completed examples)
wc -l /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013/predictions.test.lora.jsonl

# Compare to total (should be 1534)
wc -l /workspace/lora-train/dataset/bird_dev_chatml.jsonl
```

If predictions has 1534 lines, eval is done!

## Why it's slow

- **Validator service** is the bottleneck (not GPU)
  - Validates each SQL execution against database
  - Network + database query overhead
  - ~400-500ms per example
  - 1,534 examples × 0.5s = ~12+ minutes total

## Speed it up more

If you want faster, reduce test set:

```bash
# Test on just 100 examples first
python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 4 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit --limit 100
```

This will finish in ~1 minute and show your rough accuracy!

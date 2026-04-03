# FASTEST Eval (Warning: May OOM on some GPUs)

**45 seconds per batch is VERY slow!** The original command used batch_size=8 before OOM.

## Original command (gen_batch_size=8, caused OOM during first run)

Pull latest code:
```bash
cd /workspace/lora-train && git pull
```

Kill slow eval:
```bash
ps aux | grep eval_exec | grep -v grep | awk '{print $2}' | head -1 | xargs kill -9
```

Try the original batch size that OOMed before:
```bash
nohup python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 21 --validator_batch_size 100 --validator_parallelism 21  > eval_lora.log 2>&1 &
```

Monitor:
```bash
tail -f eval_lora.log
```

## Why it might work now

The original OOM happened during **generation** (KV cache buildup). But you're only using 10GB now, so batch_size=8 might fit!

- batch_size=8: ~3-4 min total (if it doesn't OOM)
- batch_size=4: ~5-7 min total (current, safe)
- batch_size=2: ~10-15 min total (very safe)

## If it OOMs again

Fall back to batch_size=6:
```bash
nohup python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 6 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit > eval_lora.log 2>&1 &
```

## Progress monitoring

Watch progress with emojis:
```bash
tail -f eval_lora.log
```

You'll see:
- `Batch 1/192: Processing examples 1-8/1534` (with batch=8)
- `✓ Validated batch 5/31`

Much faster than 45sec/batch with batch=4!

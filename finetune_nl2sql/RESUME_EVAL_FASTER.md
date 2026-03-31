# Resume LoRA Evaluation (FASTER VERSION for RTX 5090)

Your GPU is underutilized (9.5GB / 32GB). Increase batch size for 4x faster eval!

## Pull the latest code first

```bash
cd /workspace/lora-train && git pull
```

This adds real-time progress indicators like:
- `Batch 1/384: Processing examples 1-4/1534`
- `✓ Validated batch 5/31`

## Kill the slow eval first

Find the process:
```bash
ps aux | grep eval_exec
```

Kill it (replace PID with actual number from above):
```bash
kill <PID>
```

## Run with progress indicators (recommended)

```bash
nohup python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 4 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit > eval_lora.log 2>&1 &
```

Monitor with real-time progress:
```bash
tail -f eval_lora.log
```

Watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Quick test (100 examples only - finishes in ~1 min)

```bash
python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 4 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit --limit 100
```

## What you'll see now

```
📋 Loading test set: 1534 examples from /workspace/lora-train/dataset/bird_dev_chatml.jsonl
🔍 Fetching schema signature map from validator...
🚀 Generating SQL for 1534 examples in 384 batches (batch_size=4)
  Batch 1/384: Processing examples 1-4/1534
  Batch 2/384: Processing examples 5-8/1534
  ...
📊 Validating 1534 examples with validator service (31 batches, parallelism=4)
  ✓ Validated batch 1/31
  ✓ Validated batch 2/31
  ...
```

## Settings

- `gen_batch_size`: 2 → 4 (2x faster, safe for 32GB)
- Should take ~5-7 minutes for full eval
- Will use ~12-15GB VRAM (safe for 32GB GPU)

## Check results

```bash
cat /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013/eval_report.lora.json
```

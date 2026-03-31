# Resume LoRA Evaluation (FASTER VERSION for RTX 5090)

Your GPU is underutilized (9.5GB / 32GB). Increase batch size for 4x faster eval!

## Kill the slow eval first

Find the process:
```bash
ps aux | grep eval_exec
```

Kill it (replace PID with actual number):
```bash
kill <PID>
```

## Run with larger batch (MUCH FASTER)

```bash
nohup python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 4 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit > eval_lora.log 2>&1 &
```

Monitor:
```bash
tail -f eval_lora.log
```

Watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Changes

- `gen_batch_size`: 2 → 4 (2x faster, safe for 32GB)
- Should take ~5-7 minutes instead of 10-15 minutes
- Will use ~12-15GB VRAM (safe for 32GB GPU)

## Check results

```bash
cat /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013/eval_report.lora.json
```

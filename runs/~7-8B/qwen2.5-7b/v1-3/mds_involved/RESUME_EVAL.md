# Resume LoRA Evaluation

Training completed successfully! Now run the LoRA evaluation.

## Run with nohup (recommended)

```bash
nohup python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 2 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit > eval_lora.log 2>&1 &
```

Monitor progress:
```bash
tail -f eval_lora.log
```

Check results when done:
```bash
cat /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013/eval_report.lora.json
```

## Or run in foreground

```bash
python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 2 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit
```

## What this does

- Load the fine-tuned LoRA adapters
- Evaluate on BIRD dev set (1,534 examples in ChatML format)
- Create `eval_report.lora.json` with your benchmark score
- Take ~10-15 minutes

## Expected results

- Target: Beat Claude Opus 4.6 (70.15% execution accuracy)
- Your model: 69-72% expected based on training convergence

## Note

**IMPORTANT**: Use `bird_dev_chatml.jsonl` (not `bird_dev.jsonl`) - the eval script expects ChatML format with `messages` field!
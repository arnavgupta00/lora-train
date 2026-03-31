# Diagnose Eval Hangup

Run these commands on RunPod:

## 1. Is the process still running?

```bash
ps aux | grep eval_exec
```

If you see it → process still running (might be stuck)
If NOT → process crashed/finished

## 2. How many examples processed?

```bash
wc -l /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013/predictions.test.lora.jsonl
```

- Should be 1534 if done
- If it's 0 → never started processing
- If it's <1534 → stuck partway through

## 3. Check GPU usage

```bash
nvidia-smi
```

- If GPU-Util is 0% → process is hung (not running)
- If GPU-Util is 16%+ → still processing (slow but running)

## 4. Check last lines of log

```bash
tail -100 eval_lora.log
```

Look for errors, stack traces, or repeated messages

## 5. Check if validator service is responding

```bash
curl -s http://localhost:8000/health | head -20
```

If it hangs or errors → validator service is down!

## If stuck: Kill and restart with --limit 100

```bash
# Kill the process
ps aux | grep eval_exec | grep -v grep | awk '{print $2}' | xargs kill -9

# Test on 100 examples only (should finish in 1 min)
python3 -u finetune_nl2sql/eval_exec.py --base_model_id Qwen/Qwen2.5-7B-Instruct --adapter_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --test_jsonl /workspace/lora-train/dataset/bird_dev_chatml.jsonl --out_dir /workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013 --max_new_tokens 512 --gen_batch_size 4 --validator_batch_size 50 --validator_parallelism 4 --load_in_8bit --limit 100
```

This will show if the problem is the validator or the eval script itself.

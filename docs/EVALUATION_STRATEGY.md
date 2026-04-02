# Evaluation Strategy for Qwen3-1.7B on BIRD Benchmark

## Quick Reference

| Evaluation | Time | VRAM | Expected Accuracy |
|------------|------|------|-------------------|
| Base Model (no LoRA) | 20-30 min | ~8 GB | 25-35% |
| SFT LoRA | 20-30 min | ~8 GB | 55-60% |
| SFT + Self-Consistency (5 samples) | 40-60 min | ~10 GB | 58-63% |
| GRPO LoRA | 20-30 min | ~8 GB | 58-65% |
| GRPO + Self-Consistency | 40-60 min | ~10 GB | 60-67% |

---

## Parallel Evaluation Strategy (RTX 3090 24GB)

### Memory Constraints
- Each eval uses ~8-10 GB VRAM
- 2 parallel: ~16-20 GB ✅ Safe
- 3 parallel: ~24-30 GB ❌ OOM!

### Optimal 4-Evaluation Pipeline (~70-90 minutes total)

**Phase 1: Run 2 in Parallel (30 mins)**
```bash
# Terminal 1: Base model (no LoRA)
python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_1_base > eval_base.log 2>&1 &

# Wait 2 mins for model to load, then Terminal 2: SFT LoRA
sleep 120
cd /workspace/lora-train && python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir /workspace/outputs/qwen3-1.7b-sft-20260402_152110 \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_2_sft_lora > eval_sft.log 2>&1 &

# Monitor
watch -n 10 'nvidia-smi | head -15'
```

**Phase 2: Run 2 More in Parallel (40 mins)**
```bash
# After Phase 1 completes:

# Terminal 1: Base + Self-Consistency
python3 evaluation/eval_self_consistency.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_3_base_sc \
  --num_samples 5 > eval_base_sc.log 2>&1 &

# Wait 2 mins, then Terminal 2: SFT + Self-Consistency
sleep 120
python3 evaluation/eval_self_consistency.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen3-1.7b-sft-XXXXXX \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_4_sft_sc \
  --num_samples 5 > eval_sft_sc.log 2>&1 &
```

### Alternative: All Sequential (~120 mins, safest)

```bash
cd /workspace/lora-train

# 1. Base model (20-30 mins)
echo "=== Eval 1: Base Model ==="
python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_1_base

# 2. SFT LoRA (20-30 mins)
echo "=== Eval 2: SFT LoRA ==="
python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen3-1.7b-sft-XXXXXX \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_2_sft_lora

# 3. Base + Self-Consistency (30-40 mins)
echo "=== Eval 3: Base + SC ==="
python3 evaluation/eval_self_consistency.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_3_base_sc \
  --num_samples 5

# 4. SFT + Self-Consistency (30-40 mins)
echo "=== Eval 4: SFT + SC ==="
python3 evaluation/eval_self_consistency.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen3-1.7b-sft-XXXXXX \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_4_sft_sc \
  --num_samples 5
```

---

## Checking Results

```bash
# Quick summary of all results
for dir in eval_*; do
  if [[ -f "$dir/results.json" ]]; then
    echo "=== $dir ==="
    python3 -c "import json; d=json.load(open('$dir/results.json')); print(f'Accuracy: {d.get(\"accuracy\", d.get(\"execution_accuracy\", \"N/A\"))}%')"
  fi
done
```

---

## Expected Results Comparison

| Configuration | Expected Accuracy | Notes |
|--------------|-------------------|-------|
| Base model (no training) | 25-35% | Baseline capability |
| SFT LoRA only | 55-60% | **+25-30% from SFT** |
| SFT + Self-Consistency | 58-63% | +3-5% from SC voting |
| GRPO LoRA (if working) | 58-65% | +3-5% from RL |
| GRPO + Self-Consistency | 60-67% | Best expected |

**Key Insight:** SFT alone provides the biggest improvement (+25-30%). GRPO and SC each add +3-5%.

---

## Troubleshooting

### GRPO Training Issues

**Symptom: Reward = 0.000 constantly**
- The reward function is not working
- Model learns nothing from RL
- **Solution:** Skip GRPO, use SFT + Self-Consistency instead

**Symptom: Very slow training (< 2 steps/min)**
- GRPO is inherently slower than SFT (generates + evaluates)
- 14K steps at 1.5 steps/min = 155 hours!
- **Solution:** Reduce epochs or skip GRPO

### OOM During Evaluation

```bash
# Use 8-bit quantization to reduce memory
python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen3-1.7b-sft-XXXXXX \
  --load_in_8bit \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_sft_8bit
```

### Slow Evaluation

```bash
# Reduce batch size if OOM, increase if fast
python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --batch_size 4 \
  ...
```

---

## Recommended Workflow (Skip GRPO)

Given GRPO issues, recommended workflow:

```bash
cd /workspace/lora-train

# Stop GRPO if running
ps aux | grep train_grpo | grep -v grep
# kill <PID>

# Run evaluations
echo "Starting 4-point evaluation..."

# Phase 1: Base + SFT LoRA in parallel
python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_base &

sleep 120

python3 evaluation/eval_bird.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen3-1.7b-sft-* \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_sft

# Wait for base to finish
wait

# Phase 2: SC evaluations
python3 evaluation/eval_self_consistency.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen3-1.7b-sft-* \
  --bird_dev_json ./bird_eval/dev.json \
  --db_dir ./bird_eval/dev_databases \
  --output_dir ./eval_sft_sc \
  --num_samples 5

echo "=== RESULTS ==="
for dir in eval_*; do
  [[ -f "$dir/results.json" ]] && echo "$dir: $(cat $dir/results.json | python3 -c 'import sys,json; print(json.load(sys.stdin).get(\"accuracy\",\"N/A\"))')%"
done
```

**Expected final results with this workflow:**
- Base: ~30%
- SFT: ~55-60%
- SFT + SC: ~58-63%

**Total time:** ~70-90 minutes (vs 140+ hours for broken GRPO)

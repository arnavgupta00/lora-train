# Spider Benchmark for NL2SQL

This directory contains scripts to run your fine-tuned model on the [Spider](https://yale-lily.github.io/spider) benchmark.

## Quick Start

```bash
# 1. Install dependencies
pip install datasets

# 2. Convert Spider to your training format
python3 convert_spider_to_chatml.py

# 3. Fine-tune (on RunPod or local GPU)
python3 ../finetune_nl2sql/train_lora.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --train_jsonl spider_data/spider_train.qwen.jsonl \
  --dev_jsonl spider_data/spider_dev.qwen.jsonl \
  --output_dir outputs/qwen14b-spider-lora \
  --max_seq_len 1024 \
  --pack \
  --lora_r 16 \
  --num_train_epochs 3

# 4. Evaluate
python3 ../finetune_nl2sql/eval_exec.py \
  --base_model_id Qwen/Qwen2.5-14B-Instruct \
  --adapter_dir outputs/qwen14b-spider-lora \
  --test_jsonl spider_data/spider_dev.qwen.jsonl \
  --out_dir outputs/qwen14b-spider-lora
```

## What You Get

| File | Examples | Purpose |
|------|----------|---------|
| `spider_train.qwen.jsonl` | 7,000 | Training |
| `spider_dev.qwen.jsonl` | 1,034 | **Evaluation (report this!)** |

## Expected Results

Based on your 72% accuracy on domain schemas, you should expect:

- **Spider dev execution accuracy: 55-70%** (depends on query complexity)
- This would beat: Mistral-7B (~55%), Llama-3.1-8B (~58%), Phi-3 (~48%)
- Competitive with: GPT-4o-mini (~72%), Claude Haiku (~68%)

## Official Evaluation

For official Spider leaderboard submission, use the test-suite evaluation:
https://github.com/taoyds/test-suite-sql-eval

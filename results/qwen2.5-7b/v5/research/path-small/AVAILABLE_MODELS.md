# Available Small Models for SQL (0.5B - 4B)

Last Updated: April 2026

## Recommended Base Models (Ranked)

### Tier 1: Best for SQL Fine-Tuning

| Model | Size | Released | Special Features | HuggingFace |
|-------|------|----------|------------------|-------------|
| **Qwen2.5-Coder-1.5B-Instruct** | 1.5B | Nov 2024 | Code-specialized, 5.5T tokens | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **Qwen3-1.7B** | 1.7B | Apr 2025 | Built-in thinking mode, 100+ languages | `Qwen/Qwen3-1.7B` |
| **Qwen2.5-Coder-3B-Instruct** | 3B | Nov 2024 | More capacity, same architecture | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| **Qwen3-4B** | 4B | Apr 2025 | Best reasoning in small category | `Qwen/Qwen3-4B` |

### Tier 2: Ultra-Small (Phone-Capable)

| Model | Size | Released | Special Features | HuggingFace |
|-------|------|----------|------------------|-------------|
| **Qwen2.5-Coder-0.5B-Instruct** | 0.5B | Nov 2024 | SLM-SQL achieved 56.87% | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| **Qwen3-0.6B** | 0.6B | Apr 2025 | Thinking mode, newest | `Qwen/Qwen3-0.6B` |

### Tier 3: Alternatives

| Model | Size | Released | Notes | HuggingFace |
|-------|------|----------|-------|-------------|
| deepseek-coder-1.3b-instruct | 1.3B | Jan 2024 | Good but older | `deepseek-ai/deepseek-coder-1.3b-instruct` |
| Llama-3.2-1B-Instruct | 1B | Sep 2024 | General purpose | `meta-llama/Llama-3.2-1B-Instruct` |
| Llama-3.2-3B-Instruct | 3B | Sep 2024 | General purpose | `meta-llama/Llama-3.2-3B-Instruct` |
| gemma-2-2b-it | 2B | Jul 2024 | Google's small model | `google/gemma-2-2b-it` |

---

## Pre-Trained SQL Models (Ready to Use)

These are already fine-tuned for SQL tasks:

| Model | Size | BIRD Dev | Method | HuggingFace |
|-------|------|----------|--------|-------------|
| **SLM-SQL-1.5B** | 1.5B | 67.08% | SFT + GRPO | `cycloneboy/SLM-SQL-1.5B` |
| **SLM-SQL-0.5B** | 0.5B | 56.87% | SFT + GRPO | `cycloneboy/SLM-SQL-0.5B` |
| SLM-SQL-0.6B | 0.6B | TBD | SFT + GRPO | `cycloneboy/SLM-SQL-0.6B` |
| SLM-SQL-1.3B | 1.3B | TBD | SFT + GRPO | `cycloneboy/SLM-SQL-1.3B` |
| text2sql-qwen3-4b | 4B | TBD | Fine-tuned | `mradermacher/text2sql-qwen3-4b-GGUF` |
| gemma-2-2b-text2sql | 2B | TBD | Fine-tuned | `adamwhite625/gemma-2-2b-text2sql-gguf` |

---

## Merge/Revision Models

For self-consistency correction:

| Model | Size | Purpose | HuggingFace |
|-------|------|---------|-------------|
| CscSQL-Merge-1.5B | 1.5B | SQL error correction | `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct` |
| CscSQL-Merge-0.5B | 0.5B | SQL error correction | `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct` |

---

## Training Datasets

| Dataset | Size | Source | HuggingFace |
|---------|------|--------|-------------|
| **SynsQL-Think-916k** | 916K | Synthetic with reasoning | `cycloneboy/SynsQL-Think-916k` |
| **SynsQL-Merge-Think-310k** | 310K | For merge model | `cycloneboy/SynsQL-Merge-Think-310k` |
| bird_train | ~9K | BIRD official | `cycloneboy/bird_train` |
| SynSQL-2.5M | 2.5M | Base synthetic data | Various sources |

---

## Model Selection Guide

### For Maximum "Wow Factor"
**Use: Qwen2.5-Coder-0.5B + Your Training**
- "I trained a 500M parameter model that beats GPT-4"
- Runs on phones
- Most impressive size-to-performance ratio

### For Best Balance
**Use: Qwen2.5-Coder-1.5B or Qwen3-1.7B + Your Training**
- Expected: 65%+ on BIRD
- Runs on any laptop
- Good reasoning capability

### For Maximum Accuracy
**Use: Qwen2.5-Coder-3B or Qwen3-4B + Your Training**
- Expected: 68%+ on BIRD
- Needs 8GB+ VRAM
- Best accuracy in small category

---

## Hardware Requirements by Model

| Model Size | FP16 VRAM | INT8 VRAM | INT4 VRAM | CPU Inference |
|------------|-----------|-----------|-----------|---------------|
| 0.5B | 1.5 GB | 0.8 GB | 0.5 GB | ✅ Fast |
| 1.5B | 4 GB | 2 GB | 1.5 GB | ✅ Usable |
| 3B | 8 GB | 4 GB | 2.5 GB | ⚠️ Slow |
| 4B | 10 GB | 5 GB | 3 GB | ⚠️ Slow |

---

## Recent Developments (2025-2026)

### Qwen3 Series (April 2025)
- Native "thinking mode" with `<think>` tags
- Can switch between thinking and non-thinking
- Better reasoning than Qwen2.5 at same size

### SLM-SQL (July 2025)
- Proved small models can achieve 67%+ on BIRD
- Released training datasets and methodology
- Open source models on HuggingFace

### Struct-SQL (Jan 2026)
- Structured Chain-of-Thought distillation
- 60.42% with only 1000 training samples
- Shows importance of reasoning structure

---

## Quick Start Commands

```bash
# Download recommended model
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct

# Download pre-trained SQL model
huggingface-cli download cycloneboy/SLM-SQL-1.5B

# Download training dataset
huggingface-cli download cycloneboy/SynsQL-Think-916k

# Test inference
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')
print('Model loaded successfully!')
"
```

---

## Model Comparison Summary

| Model | Your Work | Expected BIRD | Time | Cost |
|-------|-----------|---------------|------|------|
| Qwen2.5-Coder-1.5B + T9 + GRPO | Train from scratch | 63-67% | 5 weeks | ~$50 |
| Qwen3-1.7B + Distillation | Distill from GPT-4 | 62-67% | 3 weeks | ~$300 |
| Qwen2.5-Coder-0.5B + T9 + GRPO | Train tiny model | 55-60% | 5 weeks | ~$30 |

**All options are 100% YOUR work** - the base model is just a starting point, like using PyTorch.

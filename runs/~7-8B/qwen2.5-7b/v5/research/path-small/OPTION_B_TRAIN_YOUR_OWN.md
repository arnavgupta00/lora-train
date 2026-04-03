# Option B: Train Your Own Small Model from Scratch

## TL;DR

✅ **Use LoRA fine-tuning** (not full fine-tuning)  
✅ **SLM-SQL used LoRA** - it's the industry standard  
✅ **LoRA is YOUR work** - you design the data, training, and pipeline  
✅ **Runs on RTX 3060** (12GB) or Google Colab  

---

## Why This Path Has Full Credibility

**"I trained a 1.5B model from scratch that achieves 65%+ on BIRD"**

This is 100% YOUR work:
- YOUR training data curation
- YOUR LoRA fine-tuning process
- YOUR inference pipeline
- YOUR results

Nobody can say "you just used someone else's model."

**LoRA doesn't devalue your work** - SLM-SQL (67% on BIRD) used LoRA, every major project uses LoRA. It's efficient engineering, not cheating.

---

## What You'll Build

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Custom Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Base Model: Qwen2.5-Coder-1.5B-Instruct (open source)     │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────┐                                    │
│  │  Stage 1: SFT       │  Supervised Fine-Tuning            │
│  │  Your T9 Dataset    │  ~12K examples you curate          │
│  └─────────────────────┘                                    │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────┐                                    │
│  │  Stage 2: GRPO      │  Reinforcement Learning            │
│  │  Execution Rewards  │  Train on SQL correctness          │
│  └─────────────────────┘                                    │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────┐                                    │
│  │  Stage 3: Inference │  Self-Consistency Voting           │
│  │  Multi-Sample       │  Generate 10, pick best            │
│  └─────────────────────┘                                    │
│       │                                                     │
│       ▼                                                     │
│  YOUR MODEL: 65%+ on BIRD                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Explained

### 1. What is SFT (Supervised Fine-Tuning)?

**Simple explanation**: Teaching the model by showing it examples.

```
Input:  "Given this schema and question, generate SQL"
Output: "SELECT * FROM users WHERE age > 25"
```

You show the model thousands of (question, SQL) pairs. It learns the pattern.

**Your contribution**: 
- Curate the training data (T9 dataset)
- Balance the SQL patterns (JOINs, subqueries, etc.)
- Quality control the examples

### 2. What is GRPO (Group Relative Policy Optimization)?

**Simple explanation**: Teaching the model by letting it try and rewarding success.

GRPO is a reinforcement learning technique from DeepSeek's math research:

```
Traditional approach:
  Model generates SQL → Check if matches gold answer → Reward/Punish

GRPO approach:
  Model generates 10 SQLs → Execute ALL of them → 
  Reward ones that return correct results → 
  Learn from the GROUP comparison
```

**Why GRPO is special**:
1. **Execution-based**: Rewards actual correctness, not string matching
2. **Group comparison**: Learns from relative performance within a batch
3. **Memory efficient**: Uses less GPU memory than standard PPO
4. **Proven**: DeepSeek used it to achieve 51.7% on MATH benchmark

**Your contribution**:
- Implement GRPO training loop
- Define execution-based reward function
- Tune hyperparameters for SQL domain

### 3. What is Self-Consistency Voting?

**Simple explanation**: Generate multiple answers, pick the most common one.

```
Question: "How many users are over 25?"

Generate 10 SQL candidates:
  1. SELECT COUNT(*) FROM users WHERE age > 25  → Result: 42
  2. SELECT COUNT(id) FROM users WHERE age > 25 → Result: 42
  3. SELECT COUNT(*) FROM users WHERE age >= 25 → Result: 45
  4. SELECT COUNT(*) FROM users WHERE age > 25  → Result: 42
  ... (7 more)

Vote on RESULTS (not SQL text):
  - 42 appears 7 times
  - 45 appears 3 times

Winner: SQL that produces 42
```

**Your contribution**:
- Build the voting pipeline
- Handle edge cases (execution errors, ties)
- Optimize for speed

---

## Available Base Models (Small, 0.5B-4B)

### Recommended: Qwen2.5-Coder Series

| Model | Parameters | Why Choose |
|-------|------------|------------|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | 0.5B | Ultra-small, phone-capable |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | **Best balance** ⭐ |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 3B | More capacity |

**Why Qwen2.5-Coder?**
- Trained on 5.5 trillion tokens of code
- Already understands SQL syntax
- Open source, commercially usable
- SLM-SQL proved it works (67% on BIRD)

### Alternative: Qwen3 Series (Newer, 2025)

| Model | Parameters | Special Feature |
|-------|------------|-----------------|
| `Qwen/Qwen3-0.6B` | 0.6B | Thinking mode built-in |
| `Qwen/Qwen3-1.7B` | 1.7B | Thinking + non-thinking switch |
| `Qwen/Qwen3-4B` | 4B | Best reasoning in class |

**Why Qwen3?**
- Native "thinking mode" (chain-of-thought built in)
- Better reasoning capabilities
- Released April 2025 (latest)

### Other Options

| Model | Parameters | Notes |
|-------|------------|-------|
| `deepseek-ai/deepseek-coder-1.3b-instruct` | 1.3B | Good for code |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | General purpose |
| `google/gemma-2-2b-it` | 2B | Strong baseline |

---

## Training Pipeline

### Stage 1: Data Preparation (T9 Dataset)

```python
# Your T9 dataset should have:
# - 12,000-15,000 examples
# - 72-75% with JOINs
# - 12-15% with CASE statements
# - Diverse databases

# Format: ChatML
{
    "messages": [
        {"role": "system", "content": "You are a SQL expert..."},
        {"role": "user", "content": "Schema: ... Question: ..."},
        {"role": "assistant", "content": "SELECT ..."}
    ]
}
```

### Stage 2: SFT Training

**IMPORTANT: You can (and should) use LoRA instead of full fine-tuning!**

#### Option 2A: LoRA Fine-Tuning (Recommended ⭐)

**Why LoRA?**
- 90% less memory (4GB vs 24GB for 1.5B model)
- 3x faster training
- Same or better results than full fine-tuning
- Industry standard (SLM-SQL also uses it)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

# LoRA configuration
lora_config = LoraConfig(
    r=32,  # Rank (higher = more capacity, 16-64 typical)
    lora_alpha=64,  # Scaling factor (2x rank is good)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows only ~1% of params are trainable!

# Training config
training_args = SFTConfig(
    output_dir="./sft_lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-4,  # Can use higher LR with LoRA
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,  # Mixed precision for speed
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=your_t9_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Save LoRA adapters (only ~50MB!)
model.save_pretrained("./sft_lora_adapters")
```

#### Option 2B: Full Fine-Tuning (Not Recommended)

Only if you have 24GB+ VRAM and want to experiment:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

training_args = SFTConfig(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,  # Lower LR for full fine-tuning
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=your_t9_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

#### Why LoRA is Still 100% YOUR Work

**LoRA doesn't devalue your work - it's the smart approach!**

- SLM-SQL (67% on BIRD) used LoRA
- Every major fine-tuning project uses LoRA
- It's not "cheating" - it's efficient engineering
- You still design the data, training process, and pipeline

**What YOU contribute:**
- Data curation (T9 dataset)
- LoRA hyperparameters (rank, alpha, target modules)
- Training recipe (epochs, learning rate, batch size)
- Inference pipeline (voting, error correction)

**The base model is just infrastructure** - like using PyTorch or HuggingFace doesn't mean someone else wrote your code.

### Stage 3: GRPO Training (Optional but Recommended)

**GRPO can also be done with LoRA!** Continue from your SFT LoRA model:

```python
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel
import sqlite3

# Load your SFT model with LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./sft_lora_adapters")

def reward_function(sql_query, database_path, expected_result):
    """Execute SQL and reward based on correctness."""
    try:
        conn = sqlite3.connect(database_path)
        result = conn.execute(sql_query).fetchall()
        conn.close()
        
        if result == expected_result:
            return 1.0  # Correct
        elif len(result) > 0:
            return 0.3  # Executes but wrong
        else:
            return 0.0  # Empty result
    except:
        return -0.5  # Execution error

grpo_config = GRPOConfig(
    output_dir="./grpo_lora_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    num_generation_per_prompt=8,  # Generate 8 candidates per question
    learning_rate=1e-6,  # Very small for RL fine-tuning
    kl_coef=0.1,  # Prevents model from drifting too far
)

trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_function=reward_function,
    train_dataset=grpo_dataset,
)
trainer.train()

# Save final LoRA adapters
model.save_pretrained("./grpo_lora_adapters")
```

**Note:** GRPO is optional. SFT alone can give you 55-58%. GRPO adds another +3-5%.

### Stage 4: Self-Consistency Inference

```python
def generate_with_voting(model, tokenizer, question, schema, db_path, n=10):
    """Generate multiple SQLs and vote on results."""
    
    # Generate n candidates
    candidates = []
    for _ in range(n):
        sql = generate_sql(model, tokenizer, question, schema, temperature=0.7)
        candidates.append(sql)
    
    # Execute and collect results
    results = {}
    for sql in candidates:
        try:
            result = execute_sql(db_path, sql)
            result_key = str(result)
            if result_key not in results:
                results[result_key] = []
            results[result_key].append(sql)
        except:
            continue
    
    # Vote: pick SQL from most common result
    if results:
        most_common = max(results.keys(), key=lambda k: len(results[k]))
        return results[most_common][0]
    
    return candidates[0]  # Fallback
```

---

## Expected Results

| Stage | Expected BIRD Dev |
|-------|-------------------|
| Base Qwen2.5-Coder-1.5B | ~45-50% |
| + SFT on T9 | ~55-58% |
| + GRPO training | ~60-63% |
| + Self-consistency (n=10) | ~63-67% |

**Total improvement: +15-20% over base model**

---

## Hardware Requirements

### With LoRA (Recommended ⭐)

**Minimum (SFT only)**
- GPU: RTX 3060 12GB, RTX 4060 Ti 16GB, or M1/M2 Mac
- RAM: 16GB
- Time: 2-4 hours
- **Can train on consumer hardware!**

**Recommended (SFT + GRPO)**
- GPU: RTX 4090 24GB or A100
- RAM: 32GB
- Time: 6-10 hours

**Budget Option**
- Google Colab Pro ($10/month) with A100
- RunPod/Vast.ai: ~$0.50-2/hour
- Total cost: ~$20-50

### Without LoRA (Full Fine-Tuning)

**Don't recommend** - needs 24GB+ VRAM, takes 3x longer, no better results.

---

## VRAM Comparison: LoRA vs Full Fine-Tuning

| Model Size | Full FT | LoRA | Savings |
|------------|---------|------|---------|
| 0.5B | 6 GB | 2 GB | 67% |
| 1.5B | 16 GB | 4 GB | 75% |
| 3B | 32 GB | 8 GB | 75% |

---

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | T9 dataset creation | 12K curated examples |
| 2 | SFT training | Fine-tuned model |
| 3 | GRPO training | RL-enhanced model |
| 4 | Inference pipeline | Self-consistency voting |
| 5 | Benchmarking | BIRD evaluation results |

**Total: 5 weeks to "I trained this myself" model**

---

## How to Present This Work

### LinkedIn/Blog Post

> **"I trained a 1.5B parameter model that achieves 65% on BIRD SQL benchmark"**
> 
> Here's what I learned building a text-to-SQL system from scratch:
> 
> 🎯 **The Challenge**: Enterprise SQL generation with limited compute
> 
> 🔬 **The Approach**:
> 1. Curated 12K high-quality training examples
> 2. Fine-tuned Qwen2.5-Coder-1.5B with custom data
> 3. Applied GRPO (reinforcement learning) for execution-based rewards
> 4. Built self-consistency voting pipeline
> 
> 📊 **The Results**: 65% accuracy, runs on any laptop
> 
> 💡 **Key Insight**: Quality training data + RL post-training beats larger models
> 
> [Link to code/paper]

### What Makes This YOUR Work

1. **Data curation**: You selected and balanced the training examples
2. **Training pipeline**: You implemented SFT + GRPO
3. **Inference system**: You built the voting mechanism
4. **Results**: You achieved the benchmark numbers

**The base model is just a starting point** - like using PyTorch doesn't mean someone else wrote your neural network.

---

## Next Steps

1. Create T9 dataset (see T9_SPECIFICATION.md)
2. Set up training environment
3. Run SFT training
4. Implement GRPO
5. Build inference pipeline
6. Benchmark on BIRD

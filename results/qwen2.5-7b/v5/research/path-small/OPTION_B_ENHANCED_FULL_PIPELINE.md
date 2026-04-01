# Option B Enhanced: SFT + GRPO + Self-Consistency + Error Correction

## Complete Pipeline for Qwen3-1.7B on BIRD Benchmark

This document covers the full implementation plan including:
1. SFT Training with LoRA
2. GRPO Reinforcement Learning
3. Self-Consistency Voting at Inference
4. Error Detection & Correction
5. Cloud GPU Costing & Setup

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                      TRAINING PHASE (Offline)                          ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  Base Model: Qwen3-1.7B                                               ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Stage 1: SFT       │  Supervised Fine-Tuning with LoRA            ║ │
│  ║  │  T9 Dataset         │  ~12K examples, 3 epochs                     ║ │
│  ║  │  (Non-thinking)     │  Output: sft_lora_adapters/                  ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Stage 2: GRPO      │  Reinforcement Learning                      ║ │
│  ║  │  Execution Rewards  │  +1 correct, 0 wrong, -0.5 error            ║ │
│  ║  │  (Non-thinking)     │  Output: grpo_lora_adapters/                 ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Stage 3: Train     │  Fine-tune on (error, correction) pairs      ║ │
│  ║  │  Error Corrector    │  Can use same base or separate small model   ║ │
│  ║  │  (Optional)         │  Output: corrector_lora_adapters/            ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                     INFERENCE PHASE (Online)                           ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  Input: Question + Schema                                             ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Step 1: Generate   │  Generate 1 SQL answer                       ║ │
│  ║  │  Single SQL         │  Temperature=0 (greedy) or low temp          ║ │
│  ║  │  (Thinking mode)    │  Use /think tag for better reasoning         ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Step 2: Execute    │  Run SQL against database                    ║ │
│  ║  │  & Validate         │  Check: Does it execute successfully?        ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Step 3: Error?     │  If execution error (syntax, column, etc.)   ║ │
│  ║  │  Check              │  → Go to correction                          ║ │
│  ║  │                     │  If success → Return SQL                     ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║       │                                                               ║ │
│  ║       ▼ (if error)                                                    ║ │
│  ║  ┌─────────────────────┐                                              ║ │
│  ║  │  Step 4: Error      │  Feed: original SQL + error message          ║ │
│  ║  │  Correction         │  Corrector model generates fixed SQL         ║ │
│  ║  │  (Validator Model)  │  Can retry up to N times                     ║ │
│  ║  └─────────────────────┘                                              ║ │
│  ║       │                                                               ║ │
│  ║       ▼                                                               ║ │
│  ║  Output: Final SQL (original or corrected)                            ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Training Phase

### Stage 1: SFT (Supervised Fine-Tuning)

**Goal**: Teach the model SQL patterns from your T9 dataset

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Load Qwen3-1.7B
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# LoRA configuration
lora_config = LoraConfig(
    r=32,                    # Rank - higher = more capacity
    lora_alpha=64,           # Scaling factor (2x rank is good)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ~1-2% of total params

# Training config
training_args = SFTConfig(
    output_dir="./sft_lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,    # Save VRAM
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=t9_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Save LoRA adapters
model.save_pretrained("./sft_lora_adapters")
```

**SFT Training Data Format** (Non-thinking mode):
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a SQL expert. Generate accurate SQL queries."
        },
        {
            "role": "user", 
            "content": "Schema:\nCREATE TABLE users (id INT, name TEXT, age INT);\n\nQuestion: Find all users over 25 years old"
        },
        {
            "role": "assistant",
            "content": "SELECT * FROM users WHERE age > 25"
        }
    ]
}
```

### Stage 2: GRPO (Group Relative Policy Optimization)

**Goal**: Optimize for execution correctness using RL

```python
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel
import sqlite3

# Load SFT model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./sft_lora_adapters")

def reward_function(sql_query, database_path, expected_result):
    """
    Execution-based reward function.
    
    Returns:
        +1.0  if execution result matches expected
        +0.3  if executes but wrong result
         0.0  if empty result
        -0.5  if execution error (syntax error, etc.)
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        
        if result == expected_result:
            return 1.0   # Correct!
        elif len(result) > 0:
            return 0.3   # Executes but wrong
        else:
            return 0.0   # Empty result
    except Exception as e:
        return -0.5      # Execution error

# GRPO config
grpo_config = GRPOConfig(
    output_dir="./grpo_lora_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    num_generation_per_prompt=8,   # Generate 8 candidates per question
    learning_rate=1e-6,            # Very small for RL
    kl_coef=0.1,                   # Prevents model drift
    max_grad_norm=0.5,
)

trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_function=reward_function,
    train_dataset=grpo_dataset,
)
trainer.train()

# Save final adapters
model.save_pretrained("./grpo_lora_adapters")
```

### Stage 3: Train Error Corrector (Optional)

**Goal**: Train a model to fix common SQL errors

```python
# Error corrector training data format
corrector_data = [
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a SQL error corrector. Fix the SQL query based on the error message."
            },
            {
                "role": "user",
                "content": """Original SQL: SELECT * FROM user WHERE ages > 25
Error: no such column: ages
Schema: CREATE TABLE users (id INT, name TEXT, age INT);
Fix the SQL query."""
            },
            {
                "role": "assistant",
                "content": "SELECT * FROM users WHERE age > 25"
            }
        ]
    }
]

# Train using same SFT approach
# Can use smaller model (0.5B) for faster inference
```

---

## Part 2: Inference Phase - Generate → Validate → Correct

### Overview

Simple, efficient pipeline:
1. **Generate** one SQL answer
2. **Validate** by executing it
3. **Correct** if there's an error (using a validator/corrector model)

This is faster than self-consistency voting (1 generation vs 10) and more targeted.

### Step 1: SQL Generator

```python
import sqlite3

class SQLGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, question, schema, use_thinking=True):
        """Generate a single SQL query."""
        
        # Use thinking mode for better reasoning
        if use_thinking:
            prompt = f"""<|im_start|>system
You are a SQL expert. Think step by step before generating SQL.<|im_end|>
<|im_start|>user
/think
Schema:
{schema}

Question: {question}

Generate the SQL query to answer this question.<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
You are a SQL expert. Generate accurate SQL queries.<|im_end|>
<|im_start|>user
Schema:
{schema}

Question: {question}

Generate the SQL query.<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,      # Greedy decoding for consistency
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = self.extract_sql(response)
        
        return sql
    
    def extract_sql(self, response):
        """Extract SQL from model response."""
        # Handle thinking mode output
        if "</think>" in response:
            response = response.split("</think>")[-1]
        
        # Clean up
        sql = response.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        
        return sql.strip()
```

### Step 2: Validator (Execute & Check)

```python
class SQLValidator:
    def __init__(self):
        pass
    
    def validate(self, sql, db_path):
        """
        Execute SQL and check if it runs successfully.
        
        Returns:
            {
                "valid": bool,
                "result": list or None,
                "error": str or None,
                "error_type": str or None  # syntax, column, table, etc.
            }
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            
            return {
                "valid": True,
                "result": result,
                "error": None,
                "error_type": None
            }
        except Exception as e:
            error_msg = str(e)
            error_type = self.classify_error(error_msg)
            
            return {
                "valid": False,
                "result": None,
                "error": error_msg,
                "error_type": error_type
            }
    
    def classify_error(self, error_msg):
        """Classify the type of SQL error."""
        error_lower = error_msg.lower()
        
        if "no such column" in error_lower:
            return "column_error"
        elif "no such table" in error_lower:
            return "table_error"
        elif "syntax error" in error_lower:
            return "syntax_error"
        elif "ambiguous column" in error_lower:
            return "ambiguous_column"
        elif "near" in error_lower:
            return "syntax_error"
        else:
            return "unknown_error"
```

### Step 3: Error Corrector (Validator Model)

The corrector model takes the original SQL + error message and generates a fixed version.

```python
class SQLCorrector:
    def __init__(self, corrector_model, corrector_tokenizer):
        """
        Corrector can be:
        - Same model as generator (simpler, saves VRAM)
        - Separate smaller model (0.5B) trained specifically for correction
        - Same model with different LoRA adapters
        """
        self.model = corrector_model
        self.tokenizer = corrector_tokenizer
    
    def correct(self, original_sql, error_message, question, schema):
        """
        Fix SQL based on the error message.
        """
        prompt = f"""<|im_start|>system
You are a SQL error corrector. Fix the SQL query based on the error message.
Pay careful attention to:
- Column names must match exactly as shown in the schema
- Table names must be correct
- SQL syntax must be valid for SQLite<|im_end|>
<|im_start|>user
Schema:
{schema}

Question: {question}

Original SQL: {original_sql}

Error: {error_message}

Generate the corrected SQL query:<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,  # Deterministic for correction
            do_sample=False,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_sql = self.extract_sql(response)
        
        return corrected_sql
    
    def extract_sql(self, response):
        """Extract SQL from response."""
        sql = response.strip()
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0]
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0]
        return sql.strip()
```

### Full Pipeline: Generate → Validate → Correct

```python
class GenerateValidateCorrectPipeline:
    def __init__(self, generator_model, corrector_model, tokenizer, max_corrections=3):
        """
        Args:
            generator_model: Model for initial SQL generation
            corrector_model: Model for error correction (can be same as generator)
            tokenizer: Shared tokenizer
            max_corrections: Maximum correction attempts before giving up
        """
        self.generator = SQLGenerator(generator_model, tokenizer)
        self.validator = SQLValidator()
        self.corrector = SQLCorrector(corrector_model, tokenizer)
        self.max_corrections = max_corrections
    
    def generate_sql(self, question, schema, db_path):
        """
        Full pipeline: Generate → Validate → Correct (if needed).
        
        Returns:
            {
                "sql": final SQL query,
                "valid": whether it executes successfully,
                "result": execution result (if valid),
                "corrections_made": number of correction attempts,
                "history": list of (sql, error) tuples
            }
        """
        history = []
        
        # Step 1: Generate initial SQL
        current_sql = self.generator.generate(question, schema, use_thinking=True)
        
        # Step 2: Validate
        validation = self.validator.validate(current_sql, db_path)
        
        if validation["valid"]:
            # Success on first try!
            return {
                "sql": current_sql,
                "valid": True,
                "result": validation["result"],
                "corrections_made": 0,
                "history": []
            }
        
        # Step 3: Correction loop
        for attempt in range(self.max_corrections):
            # Record the error
            history.append({
                "sql": current_sql,
                "error": validation["error"],
                "error_type": validation["error_type"]
            })
            
            # Correct the SQL
            current_sql = self.corrector.correct(
                original_sql=current_sql,
                error_message=validation["error"],
                question=question,
                schema=schema
            )
            
            # Validate corrected SQL
            validation = self.validator.validate(current_sql, db_path)
            
            if validation["valid"]:
                # Correction successful!
                return {
                    "sql": current_sql,
                    "valid": True,
                    "result": validation["result"],
                    "corrections_made": attempt + 1,
                    "history": history
                }
        
        # Max corrections reached, return last attempt
        history.append({
            "sql": current_sql,
            "error": validation["error"],
            "error_type": validation["error_type"]
        })
        
        return {
            "sql": current_sql,
            "valid": False,
            "result": None,
            "corrections_made": self.max_corrections,
            "history": history
        }


# Usage example
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load trained model
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
    model = PeftModel.from_pretrained(base_model, "./grpo_lora_adapters")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    # Create pipeline (using same model for generation and correction)
    pipeline = GenerateValidateCorrectPipeline(
        generator_model=model,
        corrector_model=model,  # Same model, or load separate corrector
        tokenizer=tokenizer,
        max_corrections=3
    )
    
    # Run inference
    result = pipeline.generate_sql(
        question="Find all users over 25 years old",
        schema="CREATE TABLE users (id INT, name TEXT, age INT);",
        db_path="./database.db"
    )
    
    print(f"Final SQL: {result['sql']}")
    print(f"Valid: {result['valid']}")
    print(f"Corrections made: {result['corrections_made']}")
    if result['history']:
        print("Correction history:")
        for h in result['history']:
            print(f"  - Error: {h['error']}")
```

### Training the Corrector Model (Optional)

You can train a specialized corrector model on error-correction pairs:

```python
# Generate training data from your evaluation runs
correction_training_data = []

for item in evaluation_results:
    if item["had_error"]:
        correction_training_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a SQL error corrector. Fix the SQL based on the error."
                },
                {
                    "role": "user",
                    "content": f"""Schema:
{item['schema']}

Question: {item['question']}

Original SQL: {item['predicted_sql']}

Error: {item['error_message']}

Generate the corrected SQL:"""
                },
                {
                    "role": "assistant",
                    "content": item['gold_sql']  # Use gold SQL as correction target
                }
            ]
        })

# Train corrector with same SFT approach
# Can use smaller model (Qwen3-0.6B) for faster inference
```

### Comparison: Single-Gen vs Self-Consistency

| Aspect | Generate→Validate→Correct | Self-Consistency (n=10) |
|--------|---------------------------|------------------------|
| Speed | **~1-2 sec/query** | ~10-15 sec/query |
| VRAM | Same | Same |
| Accuracy (expected) | 62-65% | 65-67% |
| Simplicity | **Simpler** | More complex |
| When to use | Latency-sensitive | Accuracy-critical |

**Your approach is better for:**
- Real-time applications
- Lower compute costs
- Simpler debugging

**Self-consistency is better for:**
- Benchmark submissions (squeeze every %)
- Batch processing where latency doesn't matter

---

## Part 3: Cloud GPU Setup & Costing

### Recommended Cloud Providers

| Provider | GPU | VRAM | Price/Hour | Best For |
|----------|-----|------|------------|----------|
| **RunPod** | RTX 4090 | 24GB | $0.44-0.74 | Best value ⭐ |
| **RunPod** | RTX 3090 | 24GB | $0.34-0.44 | Budget option |
| **Vast.ai** | RTX 4090 | 24GB | $0.30-0.50 | Variable pricing |
| **Vast.ai** | RTX 3090 | 24GB | $0.25-0.40 | Budget option |
| **Lambda Labs** | A100 40GB | 40GB | $1.10 | Faster training |
| **Google Colab Pro+** | A100 40GB | 40GB | $49.99/month | Easiest setup |

### VRAM Requirements

| Task | Qwen3-1.7B (FP16) | Qwen3-1.7B (LoRA) | Qwen3-1.7B (INT8) |
|------|-------------------|-------------------|-------------------|
| Inference only | 4 GB | 4 GB | 2 GB |
| SFT Training | 20 GB | **8 GB** ✅ | N/A |
| GRPO Training | 24 GB | **12 GB** ✅ | N/A |
| Batch Inference (n=10) | 8 GB | 8 GB | 4 GB |

**With LoRA, a single RTX 3090/4090 (24GB) is sufficient for all training!**

### Cost Breakdown

#### Option 1: RunPod RTX 4090 ($0.44/hr)

| Task | Hours | Cost |
|------|-------|------|
| Environment setup | 1 | $0.44 |
| SFT Training (12K examples, 3 epochs) | 4-6 | $2.20-2.64 |
| GRPO Training (5K examples, 1 epoch) | 6-8 | $2.64-3.52 |
| Error Corrector Training (optional) | 2-3 | $0.88-1.32 |
| Evaluation & debugging | 4 | $1.76 |
| **Total** | **17-22 hrs** | **$8-10** |

#### Option 2: RunPod RTX 3090 ($0.34/hr)

| Task | Hours | Cost |
|------|-------|------|
| Environment setup | 1 | $0.34 |
| SFT Training | 6-8 | $2.04-2.72 |
| GRPO Training | 8-10 | $2.72-3.40 |
| Error Corrector Training | 3-4 | $1.02-1.36 |
| Evaluation & debugging | 5 | $1.70 |
| **Total** | **23-28 hrs** | **$8-10** |

#### Option 3: Vast.ai Budget ($0.25-0.35/hr)

| Task | Hours | Cost |
|------|-------|------|
| Full training pipeline | 25-30 | $6-10 |

### Total Project Cost Estimate

| Component | Cost |
|-----------|------|
| Cloud GPU (training) | $8-15 |
| Cloud GPU (inference/eval) | $2-5 |
| Buffer for debugging | $5 |
| **Total Cloud Compute** | **$15-25** |

**That's it! ~$20 to train a model that could achieve 65%+ on BIRD!**

### Setup Instructions (RunPod)

```bash
# 1. Create account at runpod.io
# 2. Add funds ($25 minimum recommended)
# 3. Deploy a pod with:
#    - GPU: RTX 4090 or RTX 3090
#    - Template: RunPod Pytorch 2.1 (or similar)
#    - Container Disk: 50GB
#    - Volume: 100GB (for models and data)

# 4. Connect via SSH or web terminal

# 5. Setup environment
pip install torch transformers peft trl datasets accelerate bitsandbytes

# 6. Download base model
huggingface-cli download Qwen/Qwen3-1.7B

# 7. Upload your T9 dataset
# 8. Run training scripts
```

### Monitoring GPU Usage

```python
# Add to your training script
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Call periodically during training
print_gpu_memory()
```

---

## Part 4: Expected Results

### Accuracy Progression

| Stage | Expected BIRD Dev | Confidence |
|-------|-------------------|------------|
| Base Qwen3-1.7B (zero-shot) | 45-50% | High |
| + SFT on T9 dataset | 55-58% | High |
| + GRPO training | 60-63% | Medium-High |
| + Validate & Correct (up to 3 retries) | 62-65% | High |

### Comparison with Existing Results

| Model | BIRD Dev | Your Target |
|-------|----------|-------------|
| Your current (Qwen2.5-7B + T8) | 43.74% | Baseline |
| GPT-4 | 54.89% | Beat this |
| SLM-SQL 1.5B | 67.08% | Match this |
| Claude Opus 4 | 70.15% | Stretch goal |

### Inference Speed

| Configuration | Time per Query | Queries per Hour |
|---------------|----------------|------------------|
| Single generation (greedy) | 0.5-1s | 3600-7200 |
| + Validate (execute) | ~0.6-1.1s | 3200-6000 |
| + Correction (if needed, ~16% of queries) | ~1-2s avg | 1800-3600 |

---

## Part 5: Implementation Timeline

### Week 1: Setup & SFT
- Day 1-2: Set up cloud environment, download models
- Day 3-4: Prepare T9 dataset, format for Qwen3
- Day 5-7: Run SFT training, evaluate baseline

### Week 2: GRPO & Inference
- Day 1-3: Implement GRPO training, run training
- Day 4-5: Implement self-consistency voting
- Day 6-7: Evaluate SFT+GRPO with voting

### Week 3: Error Correction & Polish
- Day 1-3: Implement error correction pipeline
- Day 4-5: Train error corrector (optional)
- Day 6-7: Full evaluation, documentation

### Week 4: Optimization & Demo
- Day 1-2: Quantization (INT8/INT4)
- Day 3-4: Speed optimization
- Day 5-7: Create demo, write results

---

## Part 6: Files to Create

```
lm/training/
├── qwen3_sft/
│   ├── train_sft.py           # SFT training script
│   ├── config_sft.yaml        # Training hyperparameters
│   └── prepare_data.py        # Data formatting for Qwen3
├── qwen3_grpo/
│   ├── train_grpo.py          # GRPO training script
│   ├── reward_function.py     # Execution-based rewards
│   └── config_grpo.yaml       # GRPO hyperparameters
├── qwen3_corrector/
│   ├── train_corrector.py     # Error corrector training
│   └── generate_corrector_data.py
└── inference/
    ├── self_consistency.py    # Voting pipeline
    ├── error_detection.py     # Error detection
    ├── error_correction.py    # Error correction
    └── full_pipeline.py       # Complete inference
```

---

## Summary

### What You'll Build

1. **SFT Model**: Qwen3-1.7B fine-tuned on T9 dataset (~55-58% BIRD)
2. **GRPO Model**: RL-enhanced for execution correctness (~60-63% BIRD)
3. **Inference Pipeline**: Self-consistency + error correction (~65-67% BIRD)

### Why This Works

- **SFT**: Teaches SQL patterns from your curated data
- **GRPO**: Optimizes for actual execution success, not just string matching
- **Self-consistency**: Eliminates random errors through voting
- **Error correction**: Handles edge cases and syntax errors

### Total Investment

- **Time**: 3-4 weeks
- **Cost**: ~$20-30 cloud compute
- **Result**: A 1.7B model that could beat GPT-4 on BIRD!

---

## References

1. SLM-SQL: Small Language Models for Text-to-SQL (67% with 1.5B)
2. CSC-SQL: Corrective Self-Consistency (71.72% with 7B)
3. GRPO: Group Relative Policy Optimization (DeepSeek)
4. Self-Consistency: "Self-Consistency Improves Chain of Thought Reasoning"
5. Qwen3 Technical Report (April 2025)

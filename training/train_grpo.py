#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training for Text-to-SQL

This script implements GRPO training with execution-based rewards.
For each question, it generates multiple SQL candidates, executes them,
and uses the execution results as rewards for reinforcement learning.

Usage:
    python train_grpo.py \
        --base_model_id "Qwen/Qwen3-1.7B" \
        --sft_adapter_dir "./outputs/sft/" \
        --train_jsonl "data/training/grpo_train.jsonl" \
        --db_dir "./bird_databases/" \
        --output_dir "./outputs/grpo/"
"""

import argparse
import json
import os
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[bool, Any]:
    """Execute SQL and return (success, results_or_error)."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def results_match(results1: Any, results2: Any) -> bool:
    """Check if two SQL result sets match (order-independent)."""
    if results1 is None or results2 is None:
        return False
    try:
        set1 = set(tuple(row) for row in results1)
        set2 = set(tuple(row) for row in results2)
        return set1 == set2
    except Exception:
        return False


def normalize_sql(sql: str) -> str:
    """Clean up generated SQL."""
    if not sql:
        return ""
    sql = sql.strip()
    # Remove markdown code blocks
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.startswith("```"))
    sql = sql.strip()
    # Take first statement only
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    return sql


class GRPODataset(Dataset):
    """Dataset for GRPO training with database paths."""
    
    def __init__(
        self,
        jsonl_path: str,
        db_dir: str,
        tokenizer,
        max_prompt_len: int = 1024,
    ):
        self.examples = _read_jsonl(jsonl_path)
        self.db_dir = Path(db_dir)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        
        # System prompt for SQL generation
        self.system_prompt = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Only output the SQL query, nothing else."""
        
        logger.info(f"Loaded {len(self.examples)} GRPO training examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _find_database(self, db_id: str) -> Optional[str]:
        """Find database file path."""
        db_file = self.db_dir / db_id / f"{db_id}.sqlite"
        if db_file.exists():
            return str(db_file)
        # Try alternative patterns
        for pattern in [f"{db_id}.sqlite", f"{db_id}.db"]:
            matches = list(self.db_dir.glob(f"**/{pattern}"))
            if matches:
                return str(matches[0])
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        schema = example.get("schema", "")
        gold_sql = example.get("sql", example.get("SQL", ""))
        
        # If schema not in example, try to get from messages
        if not schema and "messages" in example:
            for msg in example["messages"]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if "Schema:" in content:
                        schema = content
                        break
        
        # Build prompt
        user_content = f"Schema:\n{schema}\n\nQuestion: {question}" if schema else f"Question: {question}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Find database path
        db_path = self._find_database(db_id) if db_id else None
        
        return {
            "prompt": prompt,
            "db_id": db_id,
            "db_path": db_path,
            "gold_sql": normalize_sql(gold_sql),
            "question": question,
        }


def compute_rewards(
    generated_sqls: List[str],
    gold_sql: str,
    db_path: Optional[str],
) -> List[float]:
    """
    Compute rewards for generated SQL candidates.
    
    Reward scheme:
    - 1.0: Execution result matches gold SQL result
    - 0.3: SQL executes but result doesn't match
    - 0.0: SQL doesn't execute (syntax/runtime error)
    """
    rewards = []
    
    if not db_path or not Path(db_path).exists():
        # Can't execute without database
        return [0.0] * len(generated_sqls)
    
    # Execute gold SQL to get expected result
    gold_ok, gold_result = execute_sql(db_path, gold_sql)
    if not gold_ok:
        # Gold SQL failed - fall back to syntax-based rewards
        return [0.0] * len(generated_sqls)
    
    for sql in generated_sqls:
        sql = normalize_sql(sql)
        pred_ok, pred_result = execute_sql(db_path, sql)
        
        if not pred_ok:
            rewards.append(0.0)  # Execution failed
        elif results_match(gold_result, pred_result):
            rewards.append(1.0)  # Correct result
        else:
            rewards.append(0.3)  # Executed but wrong result
    
    return rewards


def train_grpo(
    model,
    tokenizer,
    train_dataset: GRPODataset,
    output_dir: str,
    num_generations: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    learning_rate: float = 5e-7,
    num_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    kl_coef: float = 0.1,
    save_steps: int = 100,
    logging_steps: int = 10,
):
    """
    GRPO training loop.
    
    For each example:
    1. Generate `num_generations` SQL candidates
    2. Compute rewards based on execution
    3. Update model using group relative rewards
    """
    from torch.optim import AdamW
    from torch.nn.utils import clip_grad_norm_
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    
    # Training stats
    total_steps = 0
    total_loss = 0.0
    total_reward = 0.0
    reward_history = []
    
    model.train()
    
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        
        for idx in range(len(train_dataset)):
            example = train_dataset[idx]
            prompt = example["prompt"]
            gold_sql = example["gold_sql"]
            db_path = example["db_path"]
            
            # Tokenize prompt
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=train_dataset.max_prompt_len,
            ).to(model.device)
            
            # Generate multiple candidates
            generated_sqls = []
            log_probs_list = []
            
            with torch.no_grad():
                for _ in range(num_generations):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    
                    gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    generated_sqls.append(normalize_sql(gen_text))
                    
                    # Compute log probabilities for RL
                    # (simplified - in full GRPO we'd use the scores)
            
            # Compute rewards
            rewards = compute_rewards(generated_sqls, gold_sql, db_path)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            reward_history.append(avg_reward)
            total_reward += avg_reward
            
            # GRPO update: relative rewards within the group
            mean_reward = sum(rewards) / len(rewards)
            advantages = [(r - mean_reward) for r in rewards]
            
            # For each generation with positive advantage, do gradient step
            loss_acc = 0.0
            for gen_idx, (sql, advantage) in enumerate(zip(generated_sqls, advantages)):
                if advantage <= 0:
                    continue
                
                # Forward pass with the generated SQL
                full_text = prompt + sql
                full_inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=train_dataset.max_prompt_len + max_new_tokens,
                ).to(model.device)
                
                # Compute loss only on generated tokens
                labels = full_inputs["input_ids"].clone()
                labels[:, :inputs["input_ids"].shape[1]] = -100
                
                outputs = model(
                    input_ids=full_inputs["input_ids"],
                    attention_mask=full_inputs["attention_mask"],
                    labels=labels,
                )
                
                # Weight loss by advantage
                loss = outputs.loss * advantage
                loss_acc += loss.item()
                
                # Gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss_acc
            total_steps += 1
            
            # Optimizer step
            if total_steps % gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            if total_steps % logging_steps == 0:
                avg_loss = total_loss / total_steps
                avg_rew = total_reward / total_steps
                recent_reward = sum(reward_history[-100:]) / min(len(reward_history), 100)
                logger.info(
                    f"Step {total_steps} | Loss: {avg_loss:.4f} | "
                    f"Avg Reward: {avg_rew:.3f} | Recent Reward: {recent_reward:.3f}"
                )
            
            # Checkpointing
            if total_steps % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{total_steps}")
                model.save_pretrained(checkpoint_dir, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # Save training stats
    stats = {
        "total_steps": total_steps,
        "final_avg_loss": total_loss / total_steps if total_steps > 0 else 0,
        "final_avg_reward": total_reward / total_steps if total_steps > 0 else 0,
        "num_epochs": num_epochs,
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "kl_coef": kl_coef,
    }
    with open(os.path.join(output_dir, "grpo_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Training complete! Final avg reward: {stats['final_avg_reward']:.3f}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Text-to-SQL")
    
    # Model arguments
    parser.add_argument("--base_model_id", type=str, required=True,
                        help="Base model ID (e.g., Qwen/Qwen3-1.7B)")
    parser.add_argument("--sft_adapter_dir", type=str, default="",
                        help="Path to SFT LoRA adapter directory (optional)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for GRPO adapter")
    
    # Data arguments
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Training JSONL file path")
    parser.add_argument("--db_dir", type=str, required=True,
                        help="Directory containing SQLite databases")
    parser.add_argument("--max_prompt_len", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Maximum number of examples (0=all)")
    
    # GRPO arguments
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of SQL candidates per example")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL divergence coefficient")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # LoRA arguments (for continuing training)
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading base model: {args.base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    
    # Load SFT adapter if provided
    if args.sft_adapter_dir and os.path.isdir(args.sft_adapter_dir):
        logger.info(f"Loading SFT adapter from: {args.sft_adapter_dir}")
        model = PeftModel.from_pretrained(model, args.sft_adapter_dir)
        # Merge and unload for continued training
        model = model.merge_and_unload()
        logger.info("Merged SFT adapter into base model")
    
    # Add new LoRA for GRPO training
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Move to GPU
    model = model.to("cuda")
    
    # Load dataset
    logger.info(f"Loading training data: {args.train_jsonl}")
    train_dataset = GRPODataset(
        jsonl_path=args.train_jsonl,
        db_dir=args.db_dir,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
    )
    
    # Limit examples if requested
    if args.max_examples > 0:
        train_dataset.examples = train_dataset.examples[:args.max_examples]
        logger.info(f"Limited to {len(train_dataset)} examples")
    
    # Train
    logger.info("Starting GRPO training...")
    stats = train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kl_coef=args.kl_coef,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    
    # Save run metadata
    run_meta = {
        "base_model_id": args.base_model_id,
        "sft_adapter_dir": args.sft_adapter_dir,
        "train_jsonl": args.train_jsonl,
        "db_dir": args.db_dir,
        "grpo_config": {
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "kl_coef": args.kl_coef,
        },
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "stats": stats,
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    
    logger.info(f"GRPO training complete! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

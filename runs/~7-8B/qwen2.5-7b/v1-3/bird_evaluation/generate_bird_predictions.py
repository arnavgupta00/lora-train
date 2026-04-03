#!/usr/bin/env python3
"""
Generate SQL predictions for BIRD benchmark using fine-tuned model.
Outputs in official BIRD format for evaluation.
"""
import json
import os
import re
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def normalize_sql(sql: str) -> str:
    """Normalize SQL for consistent output."""
    s = sql.strip()
    # Remove code fences if present
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
        if m:
            s = m.group(1).strip()
    # Only keep first statement
    s = s.split(";")[0].strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_prompt(tokenizer, schema: str, question: str, evidence: str = "") -> str:
    """Build prompt in ChatML format."""
    system_msg = "You are an expert SQL assistant. Generate a valid SQLite query to answer the user's question based on the provided database schema."
    
    user_content = f"Database Schema:\n{schema}\n\n"
    if evidence:
        user_content += f"Hint: {evidence}\n\n"
    user_content += f"Question: {question}"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.inference_mode()
def generate_sql(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate SQL from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return normalize_sql(text)

def load_bird_dev(bird_dir: str) -> list:
    """Load BIRD dev data."""
    # Try multiple possible locations
    possible_paths = [
        Path(bird_dir) / "dev.json",
        Path(bird_dir) / "data" / "dev.json",
        Path(bird_dir) / "DAMO-ConvAI" / "bird" / "data" / "dev.json",
        Path(bird_dir) / "hf_data" / "dev.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"✓ Loaded BIRD dev from {path}: {len(data)} examples")
            return data
    
    raise FileNotFoundError(f"Could not find dev.json in {bird_dir}")

def load_schema(bird_dir: str, db_id: str) -> str:
    """Load schema for a database."""
    # Try to find database and extract schema
    possible_db_dirs = [
        Path(bird_dir) / "dev_databases" / db_id,
        Path(bird_dir) / "data" / "dev_databases" / db_id,
        Path(bird_dir) / "DAMO-ConvAI" / "bird" / "data" / "dev_databases" / db_id,
        Path(bird_dir) / "hf_data" / "dev_databases" / db_id,
    ]
    
    for db_dir in possible_db_dirs:
        schema_file = db_dir / "schema.sql"
        if schema_file.exists():
            return schema_file.read_text()
        
        # Try to extract from SQLite
        sqlite_file = db_dir / f"{db_id}.sqlite"
        if sqlite_file.exists():
            import sqlite3
            conn = sqlite3.connect(str(sqlite_file))
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
            schemas = [row[0] for row in cursor.fetchall()]
            conn.close()
            return "\n\n".join(schemas)
    
    return ""  # Return empty if not found (will use evidence)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--bird_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit examples (0 = all)")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print(f"Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading LoRA adapters from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.to("cuda")
    model.eval()
    
    print(f"Loading BIRD dev data...")
    bird_data = load_bird_dev(args.bird_dir)
    
    if args.limit > 0:
        bird_data = bird_data[:args.limit]
        print(f"Limited to {args.limit} examples")
    
    print(f"\n🚀 Generating predictions for {len(bird_data)} examples...")
    
    predictions = []
    for i, example in enumerate(bird_data):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        evidence = example.get("evidence", "")
        
        # Load schema for this database
        schema = load_schema(args.bird_dir, db_id)
        if not schema:
            schema = f"Database: {db_id}"  # Minimal fallback
        
        prompt = build_prompt(tokenizer, schema, question, evidence)
        pred_sql = generate_sql(tokenizer, model, prompt)
        
        predictions.append({
            "db_id": db_id,
            "question": question,
            "predicted_sql": pred_sql,
            "gold_sql": example.get("SQL", ""),
        })
        
        if (i + 1) % 50 == 0 or i == len(bird_data) - 1:
            print(f"  Progress: {i+1}/{len(bird_data)} ({100*(i+1)/len(bird_data):.1f}%)")
    
    # Save predictions
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✓ Saved predictions to {args.output_file}")
    
    # Also save in BIRD eval format (SQL\t----- bird -----\tdb_id)
    bird_format_file = args.output_file.replace(".json", "_bird_format.txt")
    with open(bird_format_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred['predicted_sql']}\t----- bird -----\t{pred['db_id']}\n")
    print(f"✓ Saved BIRD format to {bird_format_file}")

if __name__ == "__main__":
    main()

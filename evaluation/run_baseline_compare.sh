#!/bin/bash
# =============================================================================
# BIRD Baseline Test - Compare Base Qwen2.5-7B vs LoRA Fine-tuned
# =============================================================================
# This tests whether fine-tuning helped or hurt performance
# =============================================================================

set -e

echo "=============================================="
echo "BIRD Baseline Comparison: Base vs LoRA"
echo "=============================================="
echo ""

# Run on first 100 examples for quick comparison
python3 -u << 'PYTHON_TEST'
import json
import torch
import sqlite3
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import time

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "/workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013"
DEV_JSON = "/workspace/lora-train/bird_eval/dev_20240627/dev.json"
DB_DIR = "/workspace/lora-train/bird_eval/dev_20240627/dev_databases"
NUM_SAMPLES = 100  # Test on first 100 for speed

def get_ddl_schema_from_db(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND sql IS NOT NULL
            ORDER BY name
        """)
        create_statements = []
        for row in cursor.fetchall():
            if row[0]:
                sql = ' '.join(row[0].strip().split())
                create_statements.append(sql + ";")
        conn.close()
        return "\n".join(create_statements)
    except:
        return ""

def find_database(db_id: str) -> str:
    db_file = Path(DB_DIR) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    return None

def execute_sql(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except:
        return False, None

def result_match(gold_results, pred_results):
    if gold_results is None or pred_results is None:
        return False
    return set(tuple(r) for r in gold_results) == set(tuple(r) for r in pred_results)

def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.startswith("```"))
    sql = sql.strip()
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    return sql

SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Only output the SQL query, nothing else."""

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load BIRD data
print(f"Loading BIRD dev data (first {NUM_SAMPLES})...")
with open(DEV_JSON, 'r') as f:
    bird_data = json.load(f)[:NUM_SAMPLES]

# Pre-cache schemas
print("Caching schemas...")
schema_cache = {}
for ex in bird_data:
    db_id = ex.get("db_id", "")
    if db_id and db_id not in schema_cache:
        db_path = find_database(db_id)
        if db_path:
            schema_cache[db_id] = get_ddl_schema_from_db(db_path)

def evaluate_model(model, model_name):
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")
    
    correct = 0
    errors = 0
    
    for i, example in enumerate(bird_data):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        evidence = example.get("evidence", "")
        gold_sql = normalize_sql(example.get("SQL", ""))
        schema = schema_cache.get(db_id, "")
        
        user_content = f"Schema:\n{schema}\n\n"
        if evidence:
            user_content += f"Hint: {evidence}\n\n"
        user_content += f"Question: {question}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        pred_sql = normalize_sql(tokenizer.decode(gen_ids, skip_special_tokens=True))
        
        db_path = find_database(db_id)
        if db_path:
            gold_ok, gold_res = execute_sql(db_path, gold_sql)
            pred_ok, pred_res = execute_sql(db_path, pred_sql)
            
            if not pred_ok:
                errors += 1
            elif gold_ok and result_match(gold_res, pred_res):
                correct += 1
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{NUM_SAMPLES}] Correct: {correct}, Errors: {errors}")
    
    accuracy = 100 * correct / NUM_SAMPLES
    error_rate = 100 * errors / NUM_SAMPLES
    print(f"\n{model_name} Results:")
    print(f"  Execution match: {correct}/{NUM_SAMPLES} ({accuracy:.1f}%)")
    print(f"  Execution errors: {errors}/{NUM_SAMPLES} ({error_rate:.1f}%)")
    return accuracy, error_rate

# ========================================
# Test 1: BASE MODEL (no fine-tuning)
# ========================================
print("\n" + "="*60)
print("Loading BASE model (Qwen2.5-7B-Instruct, NO LoRA)...")
print("="*60)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
).to("cuda")
base_model.eval()

base_acc, base_err = evaluate_model(base_model, "BASE Qwen2.5-7B")

# Free memory
del base_model
torch.cuda.empty_cache()

# ========================================
# Test 2: LORA MODEL (fine-tuned)
# ========================================
print("\n" + "="*60)
print("Loading FINE-TUNED model (Qwen2.5-7B + LoRA)...")
print("="*60)

base_for_lora = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)
lora_model = PeftModel.from_pretrained(base_for_lora, ADAPTER_DIR)
lora_model = lora_model.to("cuda")
lora_model.eval()

lora_acc, lora_err = evaluate_model(lora_model, "FINE-TUNED Qwen2.5-7B + LoRA")

# ========================================
# COMPARISON
# ========================================
print("\n" + "="*60)
print("BASELINE COMPARISON (first 100 examples)")
print("="*60)
print(f"  BASE model:       {base_acc:.1f}% accuracy, {base_err:.1f}% errors")
print(f"  FINE-TUNED model: {lora_acc:.1f}% accuracy, {lora_err:.1f}% errors")
print(f"  Difference:       {lora_acc - base_acc:+.1f}%")
print()

if lora_acc > base_acc:
    print("✅ Fine-tuning HELPED! LoRA model is better.")
elif lora_acc < base_acc:
    print("❌ Fine-tuning HURT! Base model is better. Training data may be problematic.")
else:
    print("⚖️ No difference - fine-tuning had no effect on this sample.")

# Save results
results = {
    "num_samples": NUM_SAMPLES,
    "base_model": {"accuracy": base_acc, "error_rate": base_err},
    "lora_model": {"accuracy": lora_acc, "error_rate": lora_err},
    "difference": lora_acc - base_acc,
    "conclusion": "fine_tuning_helped" if lora_acc > base_acc else "fine_tuning_hurt" if lora_acc < base_acc else "no_difference"
}

os.makedirs("/workspace/lora-train/outputs/bird_evaluation", exist_ok=True)
with open("/workspace/lora-train/outputs/bird_evaluation/baseline_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to /workspace/lora-train/outputs/bird_evaluation/baseline_comparison.json")
PYTHON_TEST

echo ""
echo "=============================================="
echo "✅ Baseline comparison complete!"
echo "=============================================="

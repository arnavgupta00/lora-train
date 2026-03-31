#!/bin/bash
# Complete BIRD Evaluation Script
# Handles zip extraction and runs full evaluation

set -e

WORKSPACE="/workspace/lora-train"
OUTPUT_DIR="${WORKSPACE}/outputs/bird_evaluation"
BIRD_DIR="${WORKSPACE}/bird_eval/dev_20240627"

echo "=============================================="
echo "  BIRD Evaluation - Complete Pipeline"
echo "=============================================="
echo "Start: $(date)"
echo ""

# Extract databases if needed
if [ ! -d "$BIRD_DIR/dev_databases" ]; then
    echo "Extracting BIRD databases..."
    cd "$BIRD_DIR"
    unzip -q dev_databases.zip
    echo "✓ Databases extracted"
    cd "$WORKSPACE"
else
    echo "✓ Databases already extracted"
fi

DB_DIR="$BIRD_DIR/dev_databases"
DEV_JSON="$BIRD_DIR/dev.json"

echo "Database directory: $DB_DIR"
echo "Dev data: $DEV_JSON"
echo ""

# Step 1: Generate predictions
echo "=============================================="
echo ">>> Step 1: Generating SQL predictions"
echo "=============================================="

python3 -u << 'PYTHON_GEN'
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def normalize_sql(sql: str) -> str:
    s = sql.strip()
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
        if m:
            s = m.group(1).strip()
    s = s.split(";")[0].strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "/workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013"
DEV_JSON = "/workspace/lora-train/bird_eval/dev_20240627/dev.json"
OUTPUT_FILE = "/workspace/lora-train/outputs/bird_evaluation/predictions.json"

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

print(f"Loading LoRA adapters...")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.to("cuda")
model.eval()

print(f"Loading BIRD dev data from official dev.json...")
with open(DEV_JSON, 'r') as f:
    bird_data = json.load(f)

print(f"\n🚀 Generating predictions for {len(bird_data)} examples...\n")

predictions = []
for i, example in enumerate(bird_data):
    db_id = example.get("db_id", "")
    question = example.get("question", "")
    evidence = example.get("evidence", "")
    gold_sql = example.get("SQL", "")
    
    # Build prompt (simplified - no full schema, using evidence)
    system_msg = "You are an expert SQL assistant. Generate a valid SQLite query to answer the question."
    
    user_content = f"Database: {db_id}\n\n"
    if evidence:
        user_content += f"Hint: {evidence}\n\n"
    user_content += f"Question: {question}"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate SQL
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    pred_sql = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    predictions.append({
        "db_id": db_id,
        "question": question,
        "predicted_sql": normalize_sql(pred_sql),
        "gold_sql": normalize_sql(gold_sql),
    })
    
    if (i + 1) % 50 == 0 or i == len(bird_data) - 1:
        print(f"  Progress: {i+1}/{len(bird_data)} ({100*(i+1)/len(bird_data):.1f}%)", flush=True)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(predictions, f, indent=2)

print(f"\n✓ Saved {len(predictions)} predictions to {OUTPUT_FILE}")
PYTHON_GEN

echo ""
echo "=============================================="
echo ">>> Step 2: Evaluating predictions"
echo "=============================================="

# Step 2: Run evaluation
python3 -u << PYTHON_EVAL
import json
import sqlite3
import os
import re
from pathlib import Path

DB_DIR = "$DB_DIR"
PRED_FILE = "/workspace/lora-train/outputs/bird_evaluation/predictions.json"
OUTPUT_FILE = "/workspace/lora-train/outputs/bird_evaluation/bird_eval_report.json"

def execute_sql(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, None

def result_match(gold_results, pred_results):
    if gold_results is None or pred_results is None:
        return False
    def to_set(results):
        return set(tuple(row) for row in results)
    return to_set(gold_results) == to_set(pred_results)

def find_database(db_id: str):
    # Try most common pattern first
    db_file = Path(DB_DIR) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    
    # Try other extensions
    for ext in ['.db', '']:
        db_file = Path(DB_DIR) / db_id / f"{db_id}{ext}"
        if db_file.exists():
            return str(db_file)
    
    # Search in subdirectory
    db_dir = Path(DB_DIR) / db_id
    if db_dir.exists() and db_dir.is_dir():
        for db_file in db_dir.glob('*.sqlite'):
            return str(db_file)
        for db_file in db_dir.glob('*.db'):
            return str(db_file)
    
    return None

with open(PRED_FILE, 'r') as f:
    predictions = json.load(f)

print(f"Evaluating {len(predictions)} predictions...\n")

total = 0
exact_match = 0
exec_match = 0
exec_errors = 0
db_not_found = 0

results_detail = []

for i, pred in enumerate(predictions):
    db_id = pred["db_id"]
    pred_sql = pred["predicted_sql"]
    gold_sql = pred["gold_sql"]
    
    db_path = find_database(db_id)
    if not db_path:
        db_not_found += 1
        results_detail.append({"db_id": db_id, "error": "db_not_found"})
        total += 1
        continue
    
    # Exact match
    pred_norm = re.sub(r'\s+', ' ', pred_sql.lower().strip())
    gold_norm = re.sub(r'\s+', ' ', gold_sql.lower().strip())
    is_exact = pred_norm == gold_norm
    if is_exact:
        exact_match += 1
    
    # Execution match
    gold_ok, gold_results = execute_sql(db_path, gold_sql)
    pred_ok, pred_results = execute_sql(db_path, pred_sql)
    
    is_exec_match = False
    if gold_ok and pred_ok:
        is_exec_match = result_match(gold_results, pred_results)
    
    if is_exec_match:
        exec_match += 1
    
    if not pred_ok:
        exec_errors += 1
    
    results_detail.append({
        "db_id": db_id,
        "exact_match": is_exact,
        "exec_match": is_exec_match,
    })
    
    total += 1
    
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{len(predictions)} - Exec: {100*exec_match/total:.1f}%", flush=True)

print("\n" + "=" * 60)
print("  BIRD EVALUATION RESULTS")
print("=" * 60)
print(f"Total examples:      {total}")
print(f"Databases not found: {db_not_found}")
print(f"Exact match:         {exact_match} ({100*exact_match/total:.2f}%)")
print(f"Execution match:     {exec_match} ({100*exec_match/total:.2f}%)")
print(f"Execution errors:    {exec_errors} ({100*exec_errors/total:.2f}%)")
print("=" * 60)

ex_acc = 100 * exec_match / total if total else 0
print("\n📊 Comparison with BIRD Leaderboard:")
print(f"  Your model:           {ex_acc:.2f}%")
print(f"  Claude Opus 4.6:      70.15%")
print(f"  CSC-SQL + Qwen 7B:    71.72%")
print(f"  GPT-4:                54.89%")

if ex_acc > 70.15:
    print("\n🎉 CONGRATULATIONS! You beat Claude Opus 4.6!")
elif ex_acc > 54.89:
    print("\n✅ You beat GPT-4 baseline!")
else:
    print(f"\n💪 Keep training! You're at {ex_acc:.2f}% vs GPT-4's 54.89%")

report = {
    "total": total,
    "databases_not_found": db_not_found,
    "exact_match": exact_match,
    "exact_match_accuracy": exact_match / total if total else 0,
    "execution_match": exec_match,
    "execution_match_accuracy": exec_match / total if total else 0,
    "execution_errors": exec_errors,
    "details": results_detail,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Saved detailed results to {OUTPUT_FILE}")
PYTHON_EVAL

echo ""
echo "=============================================="
echo ">>> Evaluation Complete!"
echo "=============================================="
echo "End: $(date)"

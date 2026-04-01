#!/bin/bash
# =============================================================================
# BIRD Evaluation v2 FAST - Batched Inference
# =============================================================================
# Optimizations:
# - Batch generation (16 examples at once)
# - Padding for efficient GPU utilization
# - Progress indicators
# =============================================================================

set -e

echo "=============================================="
echo "BIRD Benchmark Evaluation v2 FAST (Batched)"
echo "=============================================="
echo ""

BIRD_DIR="/workspace/lora-train/bird_eval/dev_20240627"
DB_DIR="$BIRD_DIR/dev_databases"
DEV_JSON="$BIRD_DIR/dev.json"

# Check if databases are extracted
if [ ! -d "$DB_DIR" ]; then
    echo "Extracting dev_databases.zip..."
    cd "$BIRD_DIR"
    unzip -o dev_databases.zip
    echo "✓ Databases extracted"
fi

echo ""
echo "=============================================="
echo ">>> Step 1: Generate SQL with Batching (BS=16)"
echo "=============================================="

python3 -u << 'PYTHON_GEN'
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
OUTPUT_FILE = "/workspace/lora-train/outputs/bird_evaluation/predictions_v2_fast.json"
BATCH_SIZE = 16

def get_schema_from_db(db_path: str) -> str:
    """Extract CREATE TABLE statements from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            col_defs = [f"{col[1]}" for col in columns]
            schema_parts.append(f"{table}({','.join(col_defs)})")
        
        conn.close()
        return "\n".join(schema_parts)
    except Exception as e:
        return ""

def find_database(db_id: str) -> str:
    """Find SQLite database file."""
    db_file = Path(DB_DIR) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    for pattern in [f"*/{db_id}.sqlite", f"*/{db_id}.db"]:
        matches = list(Path(DB_DIR).glob(pattern))
        if matches:
            return str(matches[0])
    return None

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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # For batch generation

print(f"Loading base model: {MODEL_ID}")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",  # Scaled Dot Product Attention (fast, no extra deps)
    low_cpu_mem_usage=True,
)

print(f"Loading LoRA adapters from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.to("cuda")
model.eval()

print(f"Loading BIRD dev data...")
with open(DEV_JSON, 'r') as f:
    bird_data = json.load(f)

# Pre-extract all schemas
print("Pre-extracting database schemas...")
schema_cache = {}
for example in bird_data:
    db_id = example.get("db_id", "")
    if db_id and db_id not in schema_cache:
        db_path = find_database(db_id)
        if db_path:
            schema_cache[db_id] = get_schema_from_db(db_path)
        else:
            schema_cache[db_id] = ""
print(f"Cached schemas for {len(schema_cache)} databases")

print(f"\n🚀 Generating predictions for {len(bird_data)} examples (batch_size={BATCH_SIZE})...\n")

start_time = time.time()
predictions = []

# Process in batches
for batch_start in range(0, len(bird_data), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(bird_data))
    batch = bird_data[batch_start:batch_end]
    
    # Build prompts for batch
    prompts = []
    batch_metadata = []
    
    for example in batch:
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        evidence = example.get("evidence", "")
        gold_sql = example.get("SQL", "")
        schema = schema_cache.get(db_id, "")
        
        system_msg = "You are a sqlite SQL generator. Return only SQL."
        user_content = f"Schema:\n{schema}\n\n"
        if evidence:
            user_content += f"Hint: {evidence}\n\n"
        user_content += f"Question: {question}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        batch_metadata.append({
            "db_id": db_id,
            "question": question,
            "gold_sql": normalize_sql(gold_sql)
        })
    
    # Tokenize batch with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=False
    ).to(model.device)
    
    # Generate batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
        )
    
    # Decode predictions
    for i, output in enumerate(outputs):
        # Extract only generated tokens
        input_len = inputs["input_ids"][i].shape[0]
        gen_ids = output[input_len:]
        pred_sql = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred_sql = normalize_sql(pred_sql)
        
        predictions.append({
            "db_id": batch_metadata[i]["db_id"],
            "question": batch_metadata[i]["question"],
            "predicted_sql": pred_sql,
            "gold_sql": batch_metadata[i]["gold_sql"],
        })
    
    # Progress
    elapsed = time.time() - start_time
    examples_done = batch_end
    rate = examples_done / elapsed
    remaining = (len(bird_data) - examples_done) / rate if rate > 0 else 0
    
    print(f"[{examples_done}/{len(bird_data)}] {100*examples_done/len(bird_data):.1f}% - "
          f"{rate:.1f} ex/s - ETA: {remaining/60:.1f}min", flush=True)
    
    # Show sample every 5 batches
    if (batch_start // BATCH_SIZE) % 5 == 0 and batch_start > 0:
        sample = predictions[-1]
        print(f"  Sample: {sample['db_id']} - {sample['question'][:60]}...")
        print(f"    Pred: {sample['predicted_sql'][:100]}...")
        print()

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(predictions, f, indent=2)

total_time = time.time() - start_time
print(f"\n✓ Saved {len(predictions)} predictions to {OUTPUT_FILE}")
print(f"✓ Total time: {total_time/60:.1f} minutes ({len(predictions)/total_time:.1f} examples/sec)")
PYTHON_GEN

echo ""
echo "=============================================="
echo ">>> Step 2: Evaluating predictions"
echo "=============================================="

python3 -u << 'PYTHON_EVAL'
import json
import sqlite3
import os
from pathlib import Path
from collections import defaultdict

DB_DIR = "/workspace/lora-train/bird_eval/dev_20240627/dev_databases"
PRED_FILE = "/workspace/lora-train/outputs/bird_evaluation/predictions_v2_fast.json"
OUTPUT_FILE = "/workspace/lora-train/outputs/bird_evaluation/bird_eval_report_v2_fast.json"

def execute_sql(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)

def result_match(gold_results, pred_results):
    if gold_results is None or pred_results is None:
        return False
    def to_set(results):
        return set(tuple(row) for row in results)
    return to_set(gold_results) == to_set(pred_results)

def find_database(db_id: str):
    db_file = Path(DB_DIR) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    return None

print("Loading predictions...")
with open(PRED_FILE, 'r') as f:
    predictions = json.load(f)

print(f"Evaluating {len(predictions)} predictions...\n")

exact_match = 0
exec_match = 0
exec_errors = 0
stats_by_db = defaultdict(lambda: {"total": 0, "exec_match": 0})
sample_errors = []
sample_successes = []

for i, pred in enumerate(predictions):
    db_id = pred["db_id"]
    gold_sql = pred["gold_sql"]
    pred_sql = pred["predicted_sql"]
    
    stats_by_db[db_id]["total"] += 1
    
    # Exact match
    if gold_sql.lower().strip() == pred_sql.lower().strip():
        exact_match += 1
    
    # Execution match
    db_path = find_database(db_id)
    if not db_path:
        exec_errors += 1
        continue
    
    gold_ok, gold_results = execute_sql(db_path, gold_sql)
    pred_ok, pred_results = execute_sql(db_path, pred_sql)
    
    if not pred_ok:
        exec_errors += 1
        if len(sample_errors) < 5:
            sample_errors.append({
                "db_id": db_id,
                "question": pred.get("question", "")[:100],
                "pred_sql": pred_sql[:200],
                "error": str(pred_results)[:100]
            })
    elif gold_ok and result_match(gold_results, pred_results):
        exec_match += 1
        stats_by_db[db_id]["exec_match"] += 1
        if len(sample_successes) < 5:
            sample_successes.append({
                "db_id": db_id,
                "question": pred.get("question", "")[:100],
                "pred_sql": pred_sql[:200],
            })
    
    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{len(predictions)}] Exec match so far: {exec_match}/{i+1} ({100*exec_match/(i+1):.1f}%)")

total = len(predictions)
exec_accuracy = 100 * exec_match / total if total > 0 else 0
exact_accuracy = 100 * exact_match / total if total > 0 else 0

# Per-database breakdown
db_breakdown = []
for db_id, stats in sorted(stats_by_db.items(), key=lambda x: -x[1]["total"]):
    db_breakdown.append({
        "db_id": db_id,
        "total": stats["total"],
        "exec_match": stats["exec_match"],
        "accuracy": 100 * stats["exec_match"] / stats["total"] if stats["total"] > 0 else 0
    })

report = {
    "total_examples": total,
    "exact_match": exact_match,
    "exact_match_pct": round(exact_accuracy, 2),
    "execution_match": exec_match,
    "execution_accuracy": round(exec_accuracy, 2),
    "execution_errors": exec_errors,
    "by_database": db_breakdown,
    "sample_errors": sample_errors,
    "sample_successes": sample_successes,
    "comparison": {
        "your_model": round(exec_accuracy, 2),
        "claude_opus_4.6": 70.15,
        "gpt4_baseline": 54.89,
        "cscsql_qwen7b": 71.72,
        "dailsql_gpt4": 57.41,
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("BIRD BENCHMARK RESULTS v2 FAST")
print("="*60)
print(f"Total examples:     {total}")
print(f"Exact match:        {exact_match} ({exact_accuracy:.2f}%)")
print(f"Execution match:    {exec_match} ({exec_accuracy:.2f}%)")
print(f"Execution errors:   {exec_errors}")
print()
print("Top 5 databases by accuracy:")
for db in sorted(db_breakdown, key=lambda x: -x["accuracy"])[:5]:
    print(f"  {db['db_id']}: {db['accuracy']:.1f}% ({db['exec_match']}/{db['total']})")
print()
print("="*60)
print("LEADERBOARD COMPARISON")
print("="*60)
print(f"  Your Model (Qwen2.5-7B + LoRA): {exec_accuracy:.2f}%")
print(f"  Claude Opus 4.6:                70.15%")
print(f"  CSC-SQL + Qwen 7B:              71.72%")
print(f"  DAIL-SQL + GPT-4:               57.41%")
print(f"  GPT-4 baseline:                 54.89%")
print()
if exec_accuracy > 54.89:
    print("✅ BEATS GPT-4 baseline!")
if exec_accuracy > 57.41:
    print("✅ BEATS DAIL-SQL + GPT-4!")
if exec_accuracy > 70.15:
    print("✅ BEATS Claude Opus 4.6!")
if exec_accuracy > 71.72:
    print("🏆 BEATS CSC-SQL + Qwen 7B!")
print()
print(f"Report saved to: {OUTPUT_FILE}")
PYTHON_EVAL

echo ""
echo "=============================================="
echo "✅ Fast evaluation complete!"
echo "=============================================="

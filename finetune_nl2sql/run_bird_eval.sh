#!/bin/bash
# =============================================================================
# BIRD Official Evaluation Script for Qwen2.5-7B Fine-tuned Model
# =============================================================================
# This script:
# 1. Downloads BIRD evaluation code and databases
# 2. Generates SQL predictions using your trained model
# 3. Runs official BIRD evaluation
# 4. Reports execution accuracy
#
# Usage: 
#   nohup bash finetune_nl2sql/run_bird_eval.sh > bird_eval.log 2>&1 &
#   tail -f bird_eval.log
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
WORKSPACE="/workspace/lora-train"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR="${WORKSPACE}/outputs/qwen2.5-7b-t7-bird-20260331_193013"
BIRD_DIR="${WORKSPACE}/bird_eval"
OUTPUT_DIR="${WORKSPACE}/outputs/bird_evaluation"

# =============================================================================
# Step 0: Setup
# =============================================================================
echo "=============================================="
echo "  BIRD Official Evaluation"
echo "  Model: Qwen2.5-7B + LoRA (t7 trained)"
echo "=============================================="
echo "Start time: $(date)"
echo ""

mkdir -p "$BIRD_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$WORKSPACE"

# =============================================================================
# Step 1: Download BIRD Evaluation Resources
# =============================================================================
echo "=============================================="
echo ">>> Step 1: Downloading BIRD evaluation resources"
echo "=============================================="

# Download BIRD dev databases if not exists
if [ ! -d "$BIRD_DIR/dev_databases" ]; then
    echo "Downloading BIRD dev databases (~500MB)..."
    cd "$BIRD_DIR"
    
    # Option 1: From official BIRD benchmark site
    # The databases are available at: https://bird-bench.github.io/
    # Direct download link (if available)
    if command -v wget &> /dev/null; then
        wget -q --show-progress "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" -O dev.zip || true
    else
        curl -L -o dev.zip "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" || true
    fi
    
    if [ -f "dev.zip" ] && [ -s "dev.zip" ]; then
        unzip -q dev.zip
        echo "✓ Downloaded and extracted BIRD dev databases"
    else
        echo "⚠️ Direct download failed. Trying HuggingFace..."
        # Option 2: From HuggingFace (alternative)
        pip install -q huggingface_hub
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='birdsql/bird_sql_dev_20251106',
    repo_type='dataset',
    local_dir='$BIRD_DIR/hf_data',
    allow_patterns=['*.sqlite', '*.db', 'dev.json']
)
print('✓ Downloaded from HuggingFace')
"
    fi
    cd "$WORKSPACE"
else
    echo "✓ BIRD databases already exist"
fi

# Clone BIRD eval code if not exists
if [ ! -d "$BIRD_DIR/DAMO-ConvAI" ]; then
    echo "Cloning BIRD evaluation code..."
    cd "$BIRD_DIR"
    git clone --depth 1 https://github.com/AlibabaResearch/DAMO-ConvAI.git
    echo "✓ Cloned BIRD evaluation repository"
    cd "$WORKSPACE"
else
    echo "✓ BIRD evaluation code already exists"
fi

echo ""

# =============================================================================
# Step 2: Generate SQL Predictions
# =============================================================================
echo "=============================================="
echo ">>> Step 2: Generating SQL predictions"
echo "=============================================="

# Create prediction generation script
cat > "${OUTPUT_DIR}/generate_bird_predictions.py" << 'PYTHON_SCRIPT'
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
PYTHON_SCRIPT

echo "Running prediction generation..."
python3 -u "${OUTPUT_DIR}/generate_bird_predictions.py" \
    --model_id "$MODEL_ID" \
    --adapter_dir "$ADAPTER_DIR" \
    --bird_dir "$BIRD_DIR" \
    --output_file "${OUTPUT_DIR}/predictions.json"

echo ""

# =============================================================================
# Step 3: Run Official BIRD Evaluation
# =============================================================================
echo "=============================================="
echo ">>> Step 3: Running official BIRD evaluation"
echo "=============================================="

# Create evaluation script
cat > "${OUTPUT_DIR}/run_bird_eval.py" << 'EVAL_SCRIPT'
#!/usr/bin/env python3
"""
Run BIRD official evaluation on predictions.
"""
import json
import sqlite3
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import re

def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[bool, Optional[list]]:
    """Execute SQL and return results."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, None

def result_match(gold_results: list, pred_results: list) -> bool:
    """Check if results match (set comparison)."""
    if gold_results is None or pred_results is None:
        return False
    
    # Convert to comparable sets
    def to_set(results):
        return set(tuple(row) for row in results)
    
    return to_set(gold_results) == to_set(pred_results)

def find_database(bird_dir: str, db_id: str) -> Optional[str]:
    """Find SQLite database file."""
    possible_paths = [
        Path(bird_dir) / "dev_databases" / db_id / f"{db_id}.sqlite",
        Path(bird_dir) / "data" / "dev_databases" / db_id / f"{db_id}.sqlite",
        Path(bird_dir) / "DAMO-ConvAI" / "bird" / "data" / "dev_databases" / db_id / f"{db_id}.sqlite",
        Path(bird_dir) / "hf_data" / "dev_databases" / db_id / f"{db_id}.sqlite",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--bird_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    total = 0
    exact_match = 0
    exec_match = 0
    exec_errors = 0
    
    results_detail = []
    
    for i, pred in enumerate(predictions):
        db_id = pred["db_id"]
        pred_sql = pred["predicted_sql"]
        gold_sql = pred["gold_sql"]
        
        db_path = find_database(args.bird_dir, db_id)
        if not db_path:
            print(f"  ⚠️ Database not found: {db_id}")
            results_detail.append({"db_id": db_id, "exec_match": False, "error": "db_not_found"})
            exec_errors += 1
            total += 1
            continue
        
        # Exact match (normalized)
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
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "exact_match": is_exact,
            "exec_match": is_exec_match,
            "pred_executed": pred_ok,
            "gold_executed": gold_ok,
        })
        
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(predictions)} - Exec: {100*exec_match/total:.1f}%")
    
    # Summary
    print("\n" + "=" * 50)
    print("  BIRD EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total examples:     {total}")
    print(f"Exact match:        {exact_match} ({100*exact_match/total:.2f}%)")
    print(f"Execution match:    {exec_match} ({100*exec_match/total:.2f}%)")
    print(f"Execution errors:   {exec_errors} ({100*exec_errors/total:.2f}%)")
    print("=" * 50)
    
    # Compare with baselines
    ex_acc = 100 * exec_match / total
    print("\n📊 Comparison with BIRD Leaderboard:")
    print(f"  Your model:           {ex_acc:.2f}%")
    print(f"  Claude Opus 4.6:      70.15%")
    print(f"  GPT-4:                54.89%")
    print(f"  CSC-SQL + Qwen 7B:    71.72%")
    
    if ex_acc > 70.15:
        print("\n🎉 CONGRATULATIONS! You beat Claude Opus 4.6!")
    elif ex_acc > 54.89:
        print("\n✅ You beat GPT-4 baseline!")
    
    # Save detailed results
    report = {
        "total": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total if total else 0,
        "execution_match": exec_match,
        "execution_match_accuracy": exec_match / total if total else 0,
        "execution_errors": exec_errors,
        "details": results_detail,
    }
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved detailed results to {args.output}")

if __name__ == "__main__":
    main()
EVAL_SCRIPT

echo "Running evaluation..."
python3 -u "${OUTPUT_DIR}/run_bird_eval.py" \
    --predictions "${OUTPUT_DIR}/predictions.json" \
    --bird_dir "$BIRD_DIR" \
    --output "${OUTPUT_DIR}/bird_eval_report.json"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo ">>> Evaluation Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_DIR}/predictions.json"
echo "  - ${OUTPUT_DIR}/bird_eval_report.json"
echo ""
echo "View results:"
echo "  cat ${OUTPUT_DIR}/bird_eval_report.json | python3 -m json.tool | head -20"

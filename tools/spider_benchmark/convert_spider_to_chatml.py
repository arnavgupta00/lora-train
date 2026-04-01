#!/usr/bin/env python3
"""
Convert Spider dataset to ChatML format for your training pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

SCRIPT_DIR = Path(__file__).parent
SPIDER_DIR = SCRIPT_DIR / "spider_data"


def format_schema_from_example(ex: Dict[str, Any]) -> str:
    """Format schema from Spider example structure."""
    # Spider HF dataset has these fields directly in examples
    table_names = ex.get("table_names", [])
    column_names = ex.get("column_names", [])  # List of [table_idx, col_name]
    
    if not table_names:
        return f"Database: {ex.get('db_id', 'unknown')}"
    
    # Group columns by table
    table_cols: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    
    for table_idx, col_name in column_names:
        if table_idx >= 0:  # -1 is for '*'
            table_cols[table_idx].append(col_name)
    
    # Format as: table_name(col1, col2, col3)
    lines = []
    for i, table_name in enumerate(table_names):
        cols = table_cols.get(i, [])
        if cols:
            lines.append(f"{table_name}({', '.join(cols)})")
        else:
            lines.append(f"{table_name}()")
    
    return "\n".join(lines)


def convert_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single Spider example to ChatML format."""
    db_id = ex["db_id"]
    question = ex["question"]
    query = ex["query"]
    
    schema_text = format_schema_from_example(ex)
    user_content = f"Schema:\n{schema_text}\n\nQuestion:\n{question}"
    
    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": "You are a sqlite SQL generator. Return only SQL."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": query},
        ],
        "source": "spider",
        "db_id": db_id,
    }


def main():
    from datasets import load_dataset
    
    print("Loading Spider from HuggingFace...")
    ds = load_dataset("spider", trust_remote_code=True)
    
    SPIDER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert train set
    train_path = SPIDER_DIR / "spider_train.qwen.jsonl"
    with open(train_path, "w") as f:
        for ex in ds["train"]:
            converted = convert_example(ex)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
    print(f"✓ Train: {len(ds['train'])} examples → {train_path}")
    
    # Convert dev set (what you report!)
    dev_path = SPIDER_DIR / "spider_dev.qwen.jsonl"
    with open(dev_path, "w") as f:
        for ex in ds["validation"]:
            converted = convert_example(ex)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
    print(f"✓ Dev:   {len(ds['validation'])} examples → {dev_path}")
    
    print("\n" + "=" * 60)
    print("READY FOR TRAINING")
    print("=" * 60)
    print(f"\n1. Fine-tune on: {train_path}")
    print(f"2. Evaluate on:  {dev_path}")
    print("\nExample training command:")
    print(f"""
python3 finetune_nl2sql/train_lora.py \\
  --model_id Qwen/Qwen2.5-14B-Instruct \\
  --train_jsonl {train_path} \\
  --dev_jsonl {dev_path} \\
  --output_dir outputs/qwen14b-spider-lora \\
  --max_seq_len 1024 \\
  --pack \\
  --lora_r 16 \\
  --num_train_epochs 3
""")


if __name__ == "__main__":
    main()

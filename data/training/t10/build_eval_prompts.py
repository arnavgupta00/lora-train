#!/usr/bin/env python3
"""
Build Prebuilt T10 Prompts for BIRD Dev Set

Generates bird_dev_t10.jsonl with T10-formatted prompts for all BIRD dev examples.
This allows the prediction script to load prompts directly without schema extraction
at inference time.

Features:
- Extracts DDL schema from SQLite databases
- Maps BIRD 'evidence' field to T10 'Hints:' format
- Validates all prompts for T10 compliance
- Performs prompt parity check against training data
- Performs target SQL preservation audit

Usage:
    python build_eval_prompts.py \
        --bird_dev_json data/bird_eval_datasets/dev.json \
        --db_dir data/bird_eval_datasets/dev_databases \
        --output data/training/t10/bird_dev_t10.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from t10_utils import (
    T10_SYSTEM_PROMPT,
    build_t10_prompt,
    check_prompt_parity,
    find_database,
    get_ddl_schema_from_db,
    get_t10_system_prompt_hash,
    validate_t10_prompt,
)


def load_bird_dev(dev_json_path: str) -> List[Dict[str, Any]]:
    """Load BIRD dev.json file."""
    with open(dev_json_path, 'r') as f:
        return json.load(f)


def build_eval_prompts(
    dev_data: List[Dict[str, Any]],
    db_dir: str,
    training_file: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build T10 prompts for all BIRD dev examples.
    
    Args:
        dev_data: BIRD dev.json data
        db_dir: Path to dev_databases directory
        training_file: Optional path to training file for parity check
    
    Returns:
        (list of prompt records, stats dict)
    """
    stats = {
        "total": len(dev_data),
        "success": 0,
        "failed": 0,
        "with_hints": 0,
        "no_hints": 0,
        "validation_errors": [],
        "schema_cache_hits": 0,
        "parity_check": None,
        "sql_audit_passed": True,
    }
    
    # Prompt parity check
    if training_file and Path(training_file).exists():
        is_match, message = check_prompt_parity(training_file, T10_SYSTEM_PROMPT)
        stats["parity_check"] = {
            "passed": is_match,
            "message": message,
            "training_file": training_file,
        }
        if not is_match:
            print(f"⚠️  PROMPT PARITY CHECK FAILED: {message}")
            print("   Eval system prompt does not match training system prompt!")
        else:
            print(f"✅ Prompt parity check passed: {message}")
    
    # Cache schemas by db_id
    schema_cache: Dict[str, str] = {}
    
    results = []
    
    for i, example in enumerate(dev_data):
        question_id = example.get("question_id", i)
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        evidence = example.get("evidence", "")
        gold_sql = example.get("SQL", "")
        difficulty = example.get("difficulty", "unknown")
        
        # Get schema (with caching)
        if db_id in schema_cache:
            schema = schema_cache[db_id]
            stats["schema_cache_hits"] += 1
        else:
            db_path = find_database(db_dir, db_id)
            if not db_path:
                stats["failed"] += 1
                stats["validation_errors"].append({
                    "question_id": question_id,
                    "error": f"Database not found: {db_id}",
                })
                continue
            
            try:
                schema = get_ddl_schema_from_db(db_path)
                schema_cache[db_id] = schema
            except Exception as e:
                stats["failed"] += 1
                stats["validation_errors"].append({
                    "question_id": question_id,
                    "error": f"Schema extraction failed: {e}",
                })
                continue
        
        # Build T10 prompt
        hints = evidence if evidence and evidence.strip() else None
        prompt = build_t10_prompt(schema, question, hints)
        
        # Track hints stats
        if hints:
            stats["with_hints"] += 1
        else:
            stats["no_hints"] += 1
        
        # Validate T10 compliance
        is_valid, errors = validate_t10_prompt(prompt["system"], prompt["user"], strict=False)
        if not is_valid:
            stats["validation_errors"].append({
                "question_id": question_id,
                "errors": errors,
            })
            # Still include it but note the errors
        
        # Build output record
        record = {
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,  # Preserved exactly from dev.json
            "difficulty": difficulty,
            "evidence": evidence,  # Original evidence for reference
            "t10_prompt": prompt,
        }
        
        results.append(record)
        stats["success"] += 1
        
        # Progress
        if (i + 1) % 200 == 0 or (i + 1) == len(dev_data):
            print(f"  [{i+1}/{len(dev_data)}] Processed {stats['success']} examples")
    
    # Target SQL preservation audit
    print("\n🔍 Running target SQL preservation audit...")
    audit_failures = []
    for record, original in zip(results, dev_data):
        original_sql = original.get("SQL", "")
        if record["gold_sql"] != original_sql:
            audit_failures.append({
                "question_id": record["question_id"],
                "original": original_sql[:100],
                "preserved": record["gold_sql"][:100],
            })
    
    if audit_failures:
        stats["sql_audit_passed"] = False
        stats["sql_audit_failures"] = audit_failures[:5]  # First 5
        print(f"❌ SQL preservation audit FAILED: {len(audit_failures)} mismatches")
    else:
        print("✅ SQL preservation audit passed: all gold SQL preserved exactly")
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build prebuilt T10 prompts for BIRD dev set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bird_dev_json",
        required=True,
        help="Path to BIRD dev.json file",
    )
    parser.add_argument(
        "--db_dir",
        required=True,
        help="Path to dev_databases directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for bird_dev_t10.jsonl",
    )
    parser.add_argument(
        "--training_file",
        default=None,
        help="Path to training file for prompt parity check (default: auto-detect train_t10.jsonl)",
    )
    args = parser.parse_args()
    
    # Auto-detect training file
    training_file = args.training_file
    if not training_file:
        script_dir = Path(__file__).parent
        default_train = script_dir / "train_t10.jsonl"
        if default_train.exists():
            training_file = str(default_train)
            print(f"Auto-detected training file: {training_file}")
    
    print("=" * 60)
    print("Building T10 Prompts for BIRD Dev Set")
    print("=" * 60)
    print(f"BIRD dev.json: {args.bird_dev_json}")
    print(f"Database dir: {args.db_dir}")
    print(f"Output: {args.output}")
    print(f"T10 system prompt hash: {get_t10_system_prompt_hash()}")
    print()
    
    # Load BIRD dev data
    print("Loading BIRD dev data...")
    dev_data = load_bird_dev(args.bird_dev_json)
    print(f"  Loaded {len(dev_data)} examples")
    print()
    
    # Build prompts
    print("Building T10 prompts...")
    results, stats = build_eval_prompts(dev_data, args.db_dir, training_file)
    
    # Write output
    print(f"\nWriting output to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')
    
    # Write stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Total examples:     {stats['total']}")
    print(f"Successfully built: {stats['success']}")
    print(f"Failed:             {stats['failed']}")
    print(f"With hints:         {stats['with_hints']}")
    print(f"No hints:           {stats['no_hints']}")
    print(f"Schema cache hits:  {stats['schema_cache_hits']}")
    print()
    
    if stats["parity_check"]:
        status = "✅ PASSED" if stats["parity_check"]["passed"] else "❌ FAILED"
        print(f"Prompt parity check: {status}")
    
    sql_status = "✅ PASSED" if stats["sql_audit_passed"] else "❌ FAILED"
    print(f"SQL preservation audit: {sql_status}")
    
    if stats["validation_errors"]:
        print(f"\nValidation warnings: {len(stats['validation_errors'])}")
        for err in stats["validation_errors"][:3]:
            print(f"  - Q{err.get('question_id', '?')}: {err.get('error', err.get('errors', 'unknown'))}")
    
    print()
    print(f"Output written to: {args.output}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()

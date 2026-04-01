#!/usr/bin/env python3
"""
BIRD Evaluation Step 2: CPU-based execution evaluation
Compares predicted SQL against gold SQL by executing both and comparing results
Can be run locally on Mac (no GPU needed)
"""
import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

def execute_sql(db_path: str, sql: str):
    """Execute SQL query and return results"""
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
    """Check if two result sets match"""
    if gold_results is None or pred_results is None:
        return False
    def to_set(results):
        return set(tuple(row) for row in results)
    return to_set(gold_results) == to_set(pred_results)

def find_database(db_id: str, db_dir: str):
    """Find database file for given db_id"""
    db_file = Path(db_dir) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    return None

def evaluate_predictions(pred_file: str, db_dir: str, output_file: str):
    """Main evaluation function"""
    
    print("Loading predictions...")
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Evaluating {len(predictions)} predictions...")
    print(f"Database directory: {db_dir}")
    print()
    
    exact_match = 0
    exec_match = 0
    exec_errors = 0
    stats_by_db = defaultdict(lambda: {"total": 0, "exec_match": 0, "errors": 0})
    sample_errors = []
    sample_successes = []
    
    for i, pred in enumerate(predictions):
        db_id = pred["db_id"]
        gold_sql = pred["gold_sql"]
        pred_sql = pred["predicted_sql"]
        
        stats_by_db[db_id]["total"] += 1
        
        # Check exact match
        if gold_sql.lower().strip() == pred_sql.lower().strip():
            exact_match += 1
        
        # Find database
        db_path = find_database(db_id, db_dir)
        if not db_path:
            exec_errors += 1
            stats_by_db[db_id]["errors"] += 1
            print(f"⚠️  Database not found for {db_id}")
            continue
        
        # Execute both SQL queries
        gold_ok, gold_results = execute_sql(db_path, gold_sql)
        pred_ok, pred_results = execute_sql(db_path, pred_sql)
        
        # Check execution match
        if not pred_ok:
            exec_errors += 1
            stats_by_db[db_id]["errors"] += 1
            if len(sample_errors) < 10:
                sample_errors.append({
                    "db_id": db_id,
                    "question": pred.get("question", "")[:100],
                    "pred_sql": pred_sql[:300],
                    "gold_sql": gold_sql[:300],
                    "error": str(pred_results)[:150]
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
        
        # Progress update
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(predictions)}] Exec match: {exec_match}/{i+1} ({100*exec_match/(i+1):.1f}%) | Errors: {exec_errors}")
    
    # Calculate metrics
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
            "errors": stats["errors"],
            "accuracy": round(100 * stats["exec_match"] / stats["total"], 1) if stats["total"] > 0 else 0
        })
    
    # Build report
    report = {
        "version": "baseline_no_lora",
        "total_examples": total,
        "exact_match": exact_match,
        "exact_match_pct": round(exact_accuracy, 2),
        "execution_match": exec_match,
        "execution_accuracy": round(exec_accuracy, 2),
        "execution_errors": exec_errors,
        "execution_error_pct": round(100 * exec_errors / total, 2),
        "by_database": db_breakdown,
        "sample_errors": sample_errors,
        "sample_successes": sample_successes
    }
    
    # Save report
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("BIRD BENCHMARK RESULTS - BASELINE (No LoRA)")
    print("="*60)
    print(f"Total examples:     {total}")
    print(f"Exact match:        {exact_match} ({exact_accuracy:.2f}%)")
    print(f"Execution match:    {exec_match} ({exec_accuracy:.2f}%)")
    print(f"Execution errors:   {exec_errors} ({100*exec_errors/total:.1f}%)")
    print()
    
    print("="*60)
    print("PER-DATABASE BREAKDOWN")
    print("="*60)
    for db in db_breakdown[:5]:
        print(f"  {db['db_id']:<30} {db['accuracy']:>6.1f}% ({db['exec_match']}/{db['total']})")
    print()
    
    print(f"✓ Report saved to: {output_file}")
    print()
    
    return report

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("BIRD Evaluation Step 2: Execute and compare SQL predictions")
        print()
        print("Usage:")
        print("  python3 eval_step2_only.py <predictions.json> <db_dir> [output.json]")
        print()
        print("Example:")
        print("  python3 evaluation/eval_step2_only.py \\")
        print("    results/qwen2.5-7b/v4-usest8/bird_evaluation_t8/predictions_baseline.json \\")
        print("    bird_eval/dev_20240627/dev_databases \\")
        print("    results/qwen2.5-7b/v4-usest8/bird_evaluation_t8/bird_eval_report_baseline.json")
        print()
        print("Note: You need BIRD databases locally. Download from RunPod:")
        print("  scp -r root@RUNPOD_IP:/workspace/lora-train/bird_eval/dev_20240627/dev_databases ./bird_eval/dev_20240627/")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    db_dir = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else pred_file.replace("predictions_", "bird_eval_report_")
    
    # Validate inputs
    if not Path(pred_file).exists():
        print(f"❌ Error: Predictions file not found: {pred_file}")
        sys.exit(1)
    
    if not Path(db_dir).exists():
        print(f"❌ Error: Database directory not found: {db_dir}")
        print()
        print("Download databases from RunPod:")
        print(f"  mkdir -p {Path(db_dir).parent}")
        print(f"  scp -r root@RUNPOD_IP:/workspace/lora-train/bird_eval/dev_20240627/dev_databases {Path(db_dir).parent}/")
        sys.exit(1)
    
    # Run evaluation
    evaluate_predictions(pred_file, db_dir, output_file)

#!/usr/bin/env python3
"""
Evaluate predictions locally using BIRD databases
Run this on your Mac after downloading databases from RunPod
"""
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
import sys

def execute_sql(db_path: str, sql: str):
    """Execute SQL and return results or error"""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)

def results_match(pred_results, gold_results):
    """Check if prediction and gold results match"""
    if pred_results is None or gold_results is None:
        return False
    
    # Normalize: convert to sets of tuples
    try:
        pred_set = set(tuple(row) for row in pred_results)
        gold_set = set(tuple(row) for row in gold_results)
        return pred_set == gold_set
    except:
        return pred_results == gold_results

def evaluate_predictions(pred_file, db_dir):
    """Evaluate predictions against gold SQL using databases"""
    
    with open(pred_file) as f:
        predictions = json.load(f)
    
    total = len(predictions)
    exact_match = 0
    execution_match = 0
    execution_errors = 0
    
    by_db = defaultdict(lambda: {"total": 0, "exec_match": 0, "errors": 0})
    
    print(f"Evaluating {total} predictions...")
    
    for i, pred in enumerate(predictions):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total}")
        
        db_id = pred["db_id"]
        pred_sql = pred["predicted_sql"]
        gold_sql = pred["gold_sql"]
        
        by_db[db_id]["total"] += 1
        
        # Exact match
        if pred_sql.strip() == gold_sql.strip():
            exact_match += 1
        
        # Execution match
        db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            print(f"⚠️  Database not found: {db_path}")
            continue
        
        pred_results, pred_error = execute_sql(str(db_path), pred_sql)
        gold_results, gold_error = execute_sql(str(db_path), gold_sql)
        
        if pred_error:
            execution_errors += 1
            by_db[db_id]["errors"] += 1
            pred["error"] = True
            pred["error_message"] = pred_error
        else:
            pred["error"] = False
        
        if pred_results is not None and gold_results is not None:
            if results_match(pred_results, gold_results):
                execution_match += 1
                by_db[db_id]["exec_match"] += 1
                pred["execution_match"] = True
            else:
                pred["execution_match"] = False
        else:
            pred["execution_match"] = False
    
    # Calculate metrics
    exact_match_pct = (exact_match / total * 100) if total > 0 else 0
    execution_accuracy = (execution_match / total * 100) if total > 0 else 0
    execution_error_pct = (execution_errors / total * 100) if total > 0 else 0
    
    # Per-database stats
    db_stats = []
    for db_id in sorted(by_db.keys()):
        stats = by_db[db_id]
        accuracy = (stats["exec_match"] / stats["total"] * 100) if stats["total"] > 0 else 0
        db_stats.append({
            "db_id": db_id,
            "total": stats["total"],
            "exec_match": stats["exec_match"],
            "errors": stats["errors"],
            "accuracy": round(accuracy, 1)
        })
    
    report = {
        "version": "baseline_evaluated_locally",
        "total_examples": total,
        "exact_match": exact_match,
        "exact_match_pct": round(exact_match_pct, 2),
        "execution_match": execution_match,
        "execution_accuracy": round(execution_accuracy, 2),
        "execution_errors": execution_errors,
        "execution_error_pct": round(execution_error_pct, 2),
        "by_database": db_stats
    }
    
    return report, predictions

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_local.py <predictions.json> <db_dir> [output_report.json]")
        print()
        print("Example:")
        print("  # First, copy databases from RunPod:")
        print("  scp -r root@runpod:/workspace/lora-train/bird_eval/dev_20240627/dev_databases ./bird_eval/")
        print()
        print("  # Then evaluate:")
        print("  python3 evaluation/evaluate_local.py \\")
        print("    results/qwen2.5-7b/v4-usest8/bird_evaluation_t8/predictions_baseline.json \\")
        print("    bird_eval/dev_databases \\")
        print("    results/qwen2.5-7b/v4-usest8/bird_evaluation_t8/bird_eval_report_baseline.json")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    db_dir = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else pred_file.replace("predictions_", "bird_eval_report_")
    
    if not Path(pred_file).exists():
        print(f"❌ Error: Predictions file not found: {pred_file}")
        sys.exit(1)
    
    if not Path(db_dir).exists():
        print(f"❌ Error: Database directory not found: {db_dir}")
        print()
        print("Download databases from RunPod:")
        print("  scp -r root@YOUR_RUNPOD_IP:/workspace/lora-train/bird_eval/dev_20240627/dev_databases ./bird_eval/")
        sys.exit(1)
    
    print(f"Loading predictions from: {pred_file}")
    print(f"Using databases from: {db_dir}")
    print()
    
    report, updated_predictions = evaluate_predictions(pred_file, db_dir)
    
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples:       {report['total_examples']}")
    print(f"Execution accuracy:   {report['execution_accuracy']:.2f}%")
    print(f"Exact match:          {report['exact_match_pct']:.2f}%")
    print(f"Execution errors:     {report['execution_error_pct']:.2f}%")
    print()
    
    print(f"Saving report to: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save updated predictions with execution_match flags
    updated_pred_file = pred_file.replace(".json", "_evaluated.json")
    print(f"Saving updated predictions to: {updated_pred_file}")
    with open(updated_pred_file, 'w') as f:
        json.dump(updated_predictions, f, indent=2)
    
    print()
    print("✓ Done!")

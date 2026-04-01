#!/usr/bin/env python3
"""
Evaluate BIRD predictions locally (without needing SQLite databases)
This compares prediction JSON against the existing report to generate baseline report
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def load_predictions(pred_file):
    """Load predictions JSON"""
    with open(pred_file) as f:
        return json.load(f)

def calculate_metrics(predictions):
    """Calculate execution match metrics from predictions"""
    total = len(predictions)
    exact_match = 0
    execution_match = 0
    execution_errors = 0
    
    by_db = defaultdict(lambda: {"total": 0, "exec_match": 0, "errors": 0})
    
    for pred in predictions:
        db_id = pred.get("db_id", "unknown")
        by_db[db_id]["total"] += 1
        
        if pred.get("exact_match", False):
            exact_match += 1
        
        if pred.get("execution_match", False):
            execution_match += 1
            by_db[db_id]["exec_match"] += 1
        
        if pred.get("error", False):
            execution_errors += 1
            by_db[db_id]["errors"] += 1
    
    # Calculate percentages
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
    
    return {
        "version": "baseline_no_lora",
        "total_examples": total,
        "exact_match": exact_match,
        "exact_match_pct": round(exact_match_pct, 2),
        "execution_match": execution_match,
        "execution_accuracy": round(execution_accuracy, 2),
        "execution_errors": execution_errors,
        "execution_error_pct": round(execution_error_pct, 2),
        "by_database": db_stats,
        "note": "Generated locally from predictions file"
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_baseline_report.py <predictions_baseline.json> [output.json]")
        print()
        print("Example:")
        print("  python3 evaluation/generate_baseline_report.py \\")
        print("    results/qwen2.5-7b/v4-usest8/bird_evaluation_t8/predictions_baseline.json \\")
        print("    results/qwen2.5-7b/v4-usest8/bird_evaluation_t8/bird_eval_report_baseline.json")
        sys.exit(1)
    
    pred_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else pred_file.replace("predictions_", "bird_eval_report_").replace(".json", ".json")
    
    if not Path(pred_file).exists():
        print(f"❌ Error: Predictions file not found: {pred_file}")
        sys.exit(1)
    
    print(f"Loading predictions from: {pred_file}")
    predictions = load_predictions(pred_file)
    
    print(f"Calculating metrics for {len(predictions)} predictions...")
    report = calculate_metrics(predictions)
    
    print(f"Saving report to: {output_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print()
    print("=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples:       {report['total_examples']}")
    print(f"Execution accuracy:   {report['execution_accuracy']:.2f}%")
    print(f"Exact match:          {report['exact_match_pct']:.2f}%")
    print(f"Execution errors:     {report['execution_error_pct']:.2f}%")
    print()
    print("✓ Report saved!")

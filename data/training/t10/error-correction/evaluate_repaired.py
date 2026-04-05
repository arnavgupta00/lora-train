#!/usr/bin/env python3
"""
Evaluate Repaired Predictions and Generate Comparison Report

Runs BIRD evaluation on repaired predictions and compares against original baseline.

Usage:
    python evaluate_repaired.py \
        --repaired_predictions data/training/t10/error-correction/repaired_predictions_t10.jsonl \
        --original_eval runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/eval_report_t10.json \
        --prompts data/training/t10/bird_dev_t10.jsonl \
        --db_dir data/bird_eval_datasets/dev_databases \
        --output_dir data/training/t10/error-correction

Outputs:
    - repair_eval_report_t10.json: Full evaluation report on repaired predictions
    - repair_eval_summary_t10.md: Comparison summary markdown
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t10_utils import find_database, normalize_sql


# =============================================================================
# SQL Execution (copied from evaluate_t10.py for standalone use)
# =============================================================================

import sqlite3

def execute_sql(db_path: str, sql: str, timeout: int = 30):
    """Execute SQL on a SQLite database."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def results_match(gold_results, pred_results) -> bool:
    """Compare two SQL result sets using set-based comparison."""
    if gold_results is None or pred_results is None:
        return False
    def to_set(results):
        return set(tuple(row) for row in results)
    return to_set(gold_results) == to_set(pred_results)


def categorize_error(error_message: str) -> str:
    """Categorize SQL error message."""
    if not error_message:
        return "unknown"
    error_lower = error_message.lower()
    if "no such column" in error_lower:
        return "column_error"
    elif "no such table" in error_lower:
        return "table_error"
    elif "syntax error" in error_lower:
        return "syntax_error"
    elif "aggregate" in error_lower:
        return "aggregate_error"
    elif "ambiguous column" in error_lower:
        return "ambiguous_column"
    elif "near" in error_lower:
        return "syntax_error"
    else:
        return "other_error"


# =============================================================================
# Evaluation
# =============================================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_predictions(
    predictions: List[Dict],
    db_dir: str,
    timeout: int = 30,
) -> List[Dict]:
    """Evaluate all predictions."""
    results = []
    
    for i, pred in enumerate(predictions):
        qid = pred["question_id"]
        db_id = pred["db_id"]
        gold_sql = normalize_sql(pred.get("gold_sql", ""))
        pred_sql = normalize_sql(pred.get("predicted_sql", ""))
        difficulty = pred.get("difficulty", "unknown")
        repaired = pred.get("repaired", False)
        
        result = {
            "question_id": qid,
            "db_id": db_id,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
            "predicted_sql": pred_sql,
            "repaired": repaired,
            "correct": False,
            "exact_match": False,
            "exec_failed": False,
            "wrong_result": False,
            "pred_error": None,
            "error_category": None,
        }
        
        # Exact match
        if gold_sql.lower().strip() == pred_sql.lower().strip():
            result["exact_match"] = True
        
        # Find database
        db_path = find_database(db_dir, db_id)
        if not db_path:
            result["pred_error"] = f"Database not found: {db_id}"
            result["exec_failed"] = True
            result["error_category"] = "db_not_found"
            results.append(result)
            continue
        
        # Execute gold SQL
        gold_ok, gold_results = execute_sql(db_path, gold_sql, timeout)
        if not gold_ok:
            result["gold_error"] = str(gold_results)[:200]
            results.append(result)
            continue
        
        # Execute predicted SQL
        if not pred_sql:
            result["pred_error"] = "Empty prediction"
            result["exec_failed"] = True
            result["error_category"] = "empty_prediction"
            results.append(result)
            continue
        
        pred_ok, pred_results = execute_sql(db_path, pred_sql, timeout)
        
        if not pred_ok:
            result["pred_error"] = str(pred_results)[:200]
            result["exec_failed"] = True
            result["error_category"] = categorize_error(str(pred_results))
        else:
            if results_match(gold_results, pred_results):
                result["correct"] = True
            else:
                result["wrong_result"] = True
        
        results.append(result)
        
        if (i + 1) % 200 == 0:
            correct_so_far = sum(1 for r in results if r.get("correct"))
            print(f"  [{i + 1}/{len(predictions)}] EX: {correct_so_far}/{i + 1} ({100*correct_so_far/(i+1):.1f}%)")
    
    return results


def generate_report(results: List[Dict], predictions_file: str) -> Dict:
    """Generate evaluation report."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    exact_match = sum(1 for r in results if r.get("exact_match"))
    exec_failed = sum(1 for r in results if r.get("exec_failed"))
    wrong_result = sum(1 for r in results if r.get("wrong_result"))
    
    # Repaired vs non-repaired breakdown
    repaired_results = [r for r in results if r.get("repaired")]
    non_repaired_results = [r for r in results if not r.get("repaired")]
    
    repaired_correct = sum(1 for r in repaired_results if r.get("correct"))
    repaired_exec_fail = sum(1 for r in repaired_results if r.get("exec_failed"))
    
    # Per-difficulty
    difficulty_stats = {}
    for level in ["simple", "moderate", "challenging"]:
        level_results = [r for r in results if r.get("difficulty") == level]
        level_total = len(level_results)
        level_correct = sum(1 for r in level_results if r.get("correct"))
        level_exec_failed = sum(1 for r in level_results if r.get("exec_failed"))
        level_wrong_result = sum(1 for r in level_results if r.get("wrong_result"))
        difficulty_stats[level] = {
            "total": level_total,
            "correct": level_correct,
            "exec_failed": level_exec_failed,
            "wrong_result": level_wrong_result,
            "accuracy": round(100 * level_correct / level_total, 2) if level_total > 0 else 0,
        }
    
    # Error categories
    from collections import Counter
    error_categories = Counter()
    for r in results:
        if r.get("error_category"):
            error_categories[r["error_category"]] += 1
    
    report = {
        "predictions_file": predictions_file,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total_examples": total,
            "execution_accuracy": round(100 * correct / total, 2) if total > 0 else 0,
            "execution_correct": correct,
            "exact_match_accuracy": round(100 * exact_match / total, 2) if total > 0 else 0,
            "exact_match_count": exact_match,
            "exec_fail_count": exec_failed,
            "exec_fail_rate": round(100 * exec_failed / total, 2) if total > 0 else 0,
            "wrong_result_count": wrong_result,
            "wrong_result_rate": round(100 * wrong_result / total, 2) if total > 0 else 0,
        },
        "repaired_breakdown": {
            "total_repaired": len(repaired_results),
            "repaired_correct": repaired_correct,
            "repaired_exec_fail": repaired_exec_fail,
            "repaired_wrong_result": len(repaired_results) - repaired_correct - repaired_exec_fail,
        },
        "per_difficulty": difficulty_stats,
        "error_categories": dict(error_categories.most_common()),
    }
    
    return report


def generate_comparison_summary(
    repaired_report: Dict,
    original_report: Dict,
    repair_summary: Dict,
    repair_log: List[Dict],
) -> str:
    """Generate markdown comparison summary."""
    orig_s = original_report.get("summary", {})
    new_s = repaired_report["summary"]
    
    orig_ex = orig_s.get("execution_accuracy", 0)
    new_ex = new_s["execution_accuracy"]
    delta_ex = new_ex - orig_ex
    
    orig_exec_fail = orig_s.get("exec_fail_count", 0)
    new_exec_fail = new_s["exec_fail_count"]
    
    orig_wrong = orig_s.get("wrong_result_count", 0)
    new_wrong = new_s["wrong_result_count"]
    
    # Repair breakdown
    rb = repaired_report.get("repaired_breakdown", {})
    total_repaired = rb.get("total_repaired", 0)
    repaired_correct = rb.get("repaired_correct", 0)
    repaired_exec_fail = rb.get("repaired_exec_fail", 0)
    repaired_wrong = rb.get("repaired_wrong_result", 0)
    
    # Calculate repair success rate
    repair_success_rate = (100 * repaired_correct / total_repaired) if total_repaired > 0 else 0
    
    # Gain by failure type
    gain_by_type = []
    if repair_summary:
        for ftype, counts in repair_summary.get("by_failure_type", {}).items():
            # Find repaired examples of this type that are now correct
            type_repairs = [r for r in repair_log if r.get("failure_type") == ftype and r.get("final_accepted")]
            gain_by_type.append({
                "type": ftype,
                "attempted": counts.get("attempted", 0),
                "accepted": counts.get("accepted", 0),
            })
    
    # Sample successful repairs
    successful_samples = []
    for log_entry in repair_log[:20]:  # Check first 20
        if log_entry.get("final_accepted"):
            successful_samples.append({
                "question_id": log_entry["question_id"],
                "db_id": log_entry["db_id"],
                "failure_type": log_entry["failure_type"],
                "original_sql": (log_entry.get("original_predicted_sql", ""))[:150],
                "repaired_sql": (log_entry.get("final_sql", ""))[:150],
            })
            if len(successful_samples) >= 5:
                break
    
    md = f"""# T10 Error-Correction Evaluation Summary

**Generated:** {repaired_report['timestamp']}

## Overall Results

| Metric | Original | Repaired | Delta |
|--------|----------|----------|-------|
| **Execution Accuracy** | {orig_ex:.2f}% | {new_ex:.2f}% | **{delta_ex:+.2f}%** |
| Exec Failures | {orig_exec_fail} | {new_exec_fail} | {new_exec_fail - orig_exec_fail:+d} |
| Wrong Results | {orig_wrong} | {new_wrong} | {new_wrong - orig_wrong:+d} |

## Repair Effectiveness

| Metric | Value |
|--------|-------|
| Total repairs accepted | {total_repaired} |
| Repairs producing correct result | {repaired_correct} |
| Repairs still failing execution | {repaired_exec_fail} |
| Repairs producing wrong result | {repaired_wrong} |
| **Repair success rate** | {repair_success_rate:.1f}% |

## Per-Difficulty Breakdown

| Difficulty | Original EX | Repaired EX | Delta |
|------------|-------------|-------------|-------|
"""
    
    for level in ["simple", "moderate", "challenging"]:
        orig_d = original_report.get("per_difficulty", {}).get(level, {})
        new_d = repaired_report.get("per_difficulty", {}).get(level, {})
        orig_acc = orig_d.get("accuracy", 0)
        new_acc = new_d.get("accuracy", 0)
        delta = new_acc - orig_acc
        md += f"| {level.capitalize()} | {orig_acc:.1f}% | {new_acc:.1f}% | {delta:+.1f}% |\n"
    
    md += """
## Repairs by Failure Type

| Failure Type | Attempted | Accepted |
|--------------|-----------|----------|
"""
    for entry in gain_by_type:
        md += f"| {entry['type']} | {entry['attempted']} | {entry['accepted']} |\n"
    
    if successful_samples:
        md += """
## Sample Successful Repairs

"""
        for sample in successful_samples:
            md += f"""### Question {sample['question_id']} ({sample['db_id']})

**Failure type:** {sample['failure_type']}

**Original SQL:**
```sql
{sample['original_sql']}...
```

**Repaired SQL:**
```sql
{sample['repaired_sql']}...
```

---

"""
    
    md += """
## Notes

- Repairs are only attempted on SQL that fails to execute
- SQL that executes but produces wrong results is NOT repaired (no way to verify correctness)
- Repair success rate = repairs producing correct result / total accepted repairs
- A repair may execute successfully but still produce wrong result
"""
    
    return md


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate repaired predictions")
    
    parser.add_argument("--repaired_predictions", required=True,
                        help="Path to repaired_predictions_t10.jsonl")
    parser.add_argument("--original_eval", required=True,
                        help="Path to original eval_report_t10.json")
    parser.add_argument("--prompts", required=True,
                        help="Path to bird_dev_t10.jsonl")
    parser.add_argument("--db_dir", required=True,
                        help="Path to database directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--repair_log", default=None,
                        help="Path to repair_log_t10.jsonl (optional, for detailed summary)")
    parser.add_argument("--repair_summary", default=None,
                        help="Path to repair_summary_t10.json (optional)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for repair log and summary if not specified
    repair_log_path = args.repair_log or (output_dir / "repair_log_t10.jsonl")
    repair_summary_path = args.repair_summary or (output_dir / "repair_summary_t10.json")
    
    print("=" * 60)
    print("T10 Error-Correction Evaluation")
    print("=" * 60)
    print(f"Repaired predictions: {args.repaired_predictions}")
    print(f"Original eval: {args.original_eval}")
    print()
    
    # Load data
    print("Loading data...")
    predictions = load_jsonl(args.repaired_predictions)
    print(f"  Loaded {len(predictions)} repaired predictions")
    
    with open(args.original_eval, 'r') as f:
        original_report = json.load(f)
    
    repair_log = []
    if Path(repair_log_path).exists():
        repair_log = load_jsonl(str(repair_log_path))
    
    repair_summary = {}
    if Path(repair_summary_path).exists():
        with open(repair_summary_path, 'r') as f:
            repair_summary = json.load(f)
    
    # Run evaluation
    print("\nEvaluating repaired predictions...")
    results = evaluate_predictions(predictions, args.db_dir)
    
    # Generate report
    repaired_report = generate_report(results, args.repaired_predictions)
    
    # Write per-example results
    per_example_path = output_dir / "repair_per_example_results.jsonl"
    with open(per_example_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Per-example results: {per_example_path}")
    
    # Write eval report
    report_path = output_dir / "repair_eval_report_t10.json"
    with open(report_path, 'w') as f:
        json.dump(repaired_report, f, indent=2)
    print(f"Eval report: {report_path}")
    
    # Generate comparison summary
    summary_md = generate_comparison_summary(
        repaired_report=repaired_report,
        original_report=original_report,
        repair_summary=repair_summary,
        repair_log=repair_log,
    )
    
    summary_path = output_dir / "repair_eval_summary_t10.md"
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print(f"Comparison summary: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    orig_s = original_report.get("summary", {})
    new_s = repaired_report["summary"]
    
    orig_ex = orig_s.get("execution_accuracy", 0)
    new_ex = new_s["execution_accuracy"]
    
    print(f"Original EX: {orig_ex:.2f}%")
    print(f"Repaired EX: {new_ex:.2f}%")
    print(f"Delta: {new_ex - orig_ex:+.2f}%")
    print()
    print(f"Original exec failures: {orig_s.get('exec_fail_count', 0)}")
    print(f"Repaired exec failures: {new_s['exec_fail_count']}")
    print()
    
    rb = repaired_report.get("repaired_breakdown", {})
    print(f"Total repairs accepted: {rb.get('total_repaired', 0)}")
    print(f"Repairs producing correct result: {rb.get('repaired_correct', 0)}")
    print(f"Repairs producing wrong result: {rb.get('repaired_wrong_result', 0)}")


if __name__ == "__main__":
    main()

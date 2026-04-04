#!/usr/bin/env python3
"""
T11.1 Evaluation Script

Compute metrics for SQL predictions against the BIRD gold standard.

Usage:
    python data/training/t11_1/evaluate_t11_1.py \
        --predictions_file ./results/t11_1/predictions_t11_1.jsonl \
        --output_dir ./results/t11_1/eval
"""

import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
BIRD_EVAL_DIR = REPO_ROOT / "data" / "bird_eval_datasets"
DEFAULT_DB_DIR = BIRD_EVAL_DIR / "dev_databases"
DEFAULT_DEV_JSON = BIRD_EVAL_DIR / "dev.json"
DEFAULT_TIED_APPEND = BIRD_EVAL_DIR / "dev_tied_append.json"


def normalize_sql(sql: str) -> str:
    """Clean up SQL for execution."""
    if not sql:
        return ""
    
    s = sql.strip()
    
    # Strip Qwen3 thinking tags
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    
    # Strip markdown code fences
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    
    # Keep only first statement
    s = s.split(";")[0].strip()
    
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    return s


def find_database(db_dir: str, db_id: str) -> Optional[str]:
    """Find SQLite database file."""
    db_file = Path(db_dir) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    for pattern in [f"*/{db_id}.sqlite", f"*/{db_id}.db"]:
        matches = list(Path(db_dir).glob(pattern))
        if matches:
            return str(matches[0])
    return None


def execute_sql(
    db_path: str,
    sql: str,
    timeout: int = 30,
) -> Tuple[bool, Any]:
    """Execute SQL on a SQLite database and return results."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def results_match(gold_results: Any, pred_results: Any) -> bool:
    """Compare two SQL result sets using set-based comparison (order-independent)."""
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
    elif "aggregate" in error_lower or "misuse of aggregate" in error_lower:
        return "aggregate_error"
    elif "ambiguous column" in error_lower:
        return "ambiguous_column"
    elif "near" in error_lower:
        return "syntax_error"
    else:
        return "other_error"


def load_predictions(predictions_file: str) -> List[Dict]:
    """Load predictions from JSONL or JSON file."""
    pred_path = Path(predictions_file)
    if pred_path.suffix == ".jsonl":
        predictions = []
        with open(pred_path, 'r') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        return predictions

    with open(pred_path, 'r') as f:
        return json.load(f)


def load_dev_data(dev_json_path: str) -> List[Dict]:
    """Load BIRD dev.json file."""
    with open(dev_json_path, 'r') as f:
        return json.load(f)


def load_tied_append(tied_append_path: str) -> Dict[int, Dict]:
    """Load tied-append alternatives."""
    if not Path(tied_append_path).exists():
        return {}
    
    with open(tied_append_path, 'r') as f:
        tied_data = json.load(f)
    
    tied_map = {}
    for entry in tied_data:
        qid = entry.get("question_id")
        if qid is not None:
            tied_map[qid] = entry
    
    return tied_map


def evaluate_single(
    prediction: Dict,
    dev_example: Dict,
    db_dir: str,
    timeout: int,
    tied_entry: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Evaluate a single prediction against the gold SQL."""
    db_id = prediction.get("db_id", "")
    question_id = prediction.get("question_id", -1)
    difficulty = prediction.get("difficulty", dev_example.get("difficulty", "unknown"))
    
    # Get SQLs
    pred_sql = normalize_sql(prediction.get("predicted_sql", ""))
    gold_sql_raw = prediction.get("gold_sql", dev_example.get("SQL", ""))
    gold_sql = normalize_sql(gold_sql_raw)

    result = {
        "question_id": question_id,
        "db_id": db_id,
        "difficulty": difficulty,
        "gold_sql": gold_sql,
        "predicted_sql": pred_sql,
        "correct": False,
        "exact_match": False,
        "exec_failed": False,
        "wrong_result": False,
        "pred_error": None,
        "gold_error": None,
        "error_category": None,
    }

    # Check exact match (case-insensitive)
    if gold_sql.lower().strip() == pred_sql.lower().strip():
        result["exact_match"] = True

    # Find database
    db_path = find_database(db_dir, db_id)
    if not db_path:
        result["pred_error"] = f"Database not found: {db_id}"
        result["exec_failed"] = True
        result["error_category"] = "db_not_found"
        return result

    # Execute gold SQL
    gold_ok, gold_results = execute_sql(db_path, gold_sql, timeout)
    if not gold_ok:
        result["gold_error"] = str(gold_results)[:200]
        return result

    # Execute predicted SQL
    if not pred_sql:
        result["pred_error"] = "Empty prediction"
        result["exec_failed"] = True
        result["error_category"] = "empty_prediction"
        return result

    pred_ok, pred_results = execute_sql(db_path, pred_sql, timeout)
    
    if not pred_ok:
        result["pred_error"] = str(pred_results)[:200]
        result["exec_failed"] = True
        result["error_category"] = categorize_error(str(pred_results))
    else:
        # Compare results (set-based, order-independent)
        if results_match(gold_results, pred_results):
            result["correct"] = True
        else:
            result["wrong_result"] = True

    # Tied-append: if main gold didn't match, try the alternative gold SQL
    if not result["correct"] and tied_entry is not None and not result["exec_failed"]:
        tied_gold_raw = tied_entry.get("SQL", "")
        tied_gold = normalize_sql(tied_gold_raw)
        if tied_gold:
            tied_ok, tied_results = execute_sql(db_path, tied_gold, timeout)
            if tied_ok and results_match(tied_results, pred_results):
                result["correct"] = True
                result["wrong_result"] = False
                result["matched_via"] = "tied_append"

    return result


def evaluate_predictions(
    predictions: List[Dict],
    dev_data: List[Dict],
    db_dir: str,
    timeout: int = 30,
    max_workers: int = 4,
    tied_map: Optional[Dict[int, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all predictions against the gold standard."""
    if len(predictions) != len(dev_data):
        print(
            f"⚠ Warning: {len(predictions)} predictions but {len(dev_data)} examples in dev data. "
            f"Evaluating min({len(predictions)}, {len(dev_data)}) examples."
        )

    # Build lookup for dev data by question_id
    dev_lookup = {ex.get("question_id"): ex for ex in dev_data}

    n = min(len(predictions), len(dev_data))
    results = [None] * n

    if tied_map is None:
        tied_map = {}

    def worker(i: int) -> Tuple[int, Dict]:
        pred = predictions[i]
        qid = pred.get("question_id")
        dev_example = dev_lookup.get(qid) if qid is not None else None
        if dev_example is None:
            dev_example = dev_data[i]
        tied_entry = tied_map.get(qid)
        
        return i, evaluate_single(pred, dev_example, db_dir, timeout, tied_entry)

    print(f"\n🔍 Evaluating {n} predictions (timeout={timeout}s, workers={max_workers})...\n")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, i): i for i in range(n)}

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1

            if completed % 200 == 0 or completed == n:
                correct_so_far = sum(1 for r in results if r and r.get("correct"))
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{n}] EX: {correct_so_far}/{completed} "
                    f"({100 * correct_so_far / completed:.1f}%) - "
                    f"{rate:.0f} ex/s"
                )

    return results


def generate_report(
    results: List[Dict],
    predictions_file: str,
) -> Dict[str, Any]:
    """Generate a comprehensive evaluation report."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    exact_match = sum(1 for r in results if r.get("exact_match"))
    exec_failed = sum(1 for r in results if r.get("exec_failed"))
    wrong_result = sum(1 for r in results if r.get("wrong_result"))
    gold_errors = sum(1 for r in results if r.get("gold_error"))
    tied_matches = sum(1 for r in results if r.get("matched_via") == "tied_append")

    ex_accuracy = 100 * correct / total if total > 0 else 0
    em_accuracy = 100 * exact_match / total if total > 0 else 0

    # Per-difficulty breakdown
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

    # Per-database breakdown
    db_stats = defaultdict(lambda: {"total": 0, "correct": 0, "exec_failed": 0, "wrong_result": 0})
    for r in results:
        db_id = r.get("db_id", "unknown")
        db_stats[db_id]["total"] += 1
        if r.get("correct"):
            db_stats[db_id]["correct"] += 1
        if r.get("exec_failed"):
            db_stats[db_id]["exec_failed"] += 1
        if r.get("wrong_result"):
            db_stats[db_id]["wrong_result"] += 1

    db_breakdown = []
    for db_id in sorted(db_stats.keys()):
        s = db_stats[db_id]
        db_breakdown.append({
            "db_id": db_id,
            "total": s["total"],
            "correct": s["correct"],
            "exec_failed": s["exec_failed"],
            "wrong_result": s["wrong_result"],
            "accuracy": round(100 * s["correct"] / s["total"], 2) if s["total"] > 0 else 0,
        })

    # Error categorization
    error_categories = Counter()
    for r in results:
        if r.get("error_category"):
            error_categories[r["error_category"]] += 1

    # Sample errors
    sample_exec_errors = []
    sample_wrong_results = []
    for r in results:
        if r.get("exec_failed") and len(sample_exec_errors) < 10:
            sample_exec_errors.append({
                "question_id": r["question_id"],
                "db_id": r["db_id"],
                "difficulty": r["difficulty"],
                "predicted_sql": (r.get("predicted_sql", ""))[:300],
                "error": (r.get("pred_error", ""))[:200],
                "error_category": r.get("error_category"),
            })
        if r.get("wrong_result") and len(sample_wrong_results) < 10:
            sample_wrong_results.append({
                "question_id": r["question_id"],
                "db_id": r["db_id"],
                "difficulty": r["difficulty"],
                "predicted_sql": (r.get("predicted_sql", ""))[:300],
                "gold_sql": (r.get("gold_sql", ""))[:300],
            })

    report = {
        "predictions_file": predictions_file,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total_examples": total,
            "execution_accuracy": round(ex_accuracy, 2),
            "execution_correct": correct,
            "exact_match_accuracy": round(em_accuracy, 2),
            "exact_match_count": exact_match,
            "exec_fail_count": exec_failed,
            "exec_fail_rate": round(100 * exec_failed / total, 2) if total > 0 else 0,
            "wrong_result_count": wrong_result,
            "wrong_result_rate": round(100 * wrong_result / total, 2) if total > 0 else 0,
            "gold_sql_errors": gold_errors,
            "tied_append_matches": tied_matches,
        },
        "per_difficulty": difficulty_stats,
        "per_database": db_breakdown,
        "error_categories": dict(error_categories.most_common()),
        "sample_exec_errors": sample_exec_errors,
        "sample_wrong_results": sample_wrong_results,
    }

    return report


def generate_run_manifest(
    predictions_file: str,
    report: Dict[str, Any],
    generation_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate run manifest for reproducibility."""
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "predictions_file": predictions_file,
        "summary": report["summary"],
    }

    if generation_config_path and os.path.exists(generation_config_path):
        with open(generation_config_path, 'r') as f:
            manifest["generation_config"] = json.load(f)

    return manifest


def print_report(report: Dict) -> None:
    """Print a human-readable summary."""
    s = report["summary"]

    print("\n" + "=" * 60)
    print("T11.1 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Predictions file:      {report['predictions_file']}")
    print(f"  Timestamp:             {report['timestamp']}")
    print()
    print(f"  Total examples:        {s['total_examples']}")
    print(f"  Execution Accuracy:    {s['execution_correct']}/{s['total_examples']} ({s['execution_accuracy']}%)")
    print(f"  Exact Match:           {s['exact_match_count']}/{s['total_examples']} ({s['exact_match_accuracy']}%)")
    print()
    print("  --- Failure Breakdown ---")
    print(f"  Exec failures:         {s['exec_fail_count']} ({s['exec_fail_rate']}%)")
    print(f"  Wrong results:         {s['wrong_result_count']} ({s['wrong_result_rate']}%)")
    if s['gold_sql_errors'] > 0:
        print(f"  Gold SQL errors:       {s['gold_sql_errors']} (benchmark issues)")
    if s['tied_append_matches'] > 0:
        print(f"  Tied-append matches:   {s['tied_append_matches']}")

    # Per-difficulty
    print()
    print("-" * 60)
    print("PER-DIFFICULTY BREAKDOWN")
    print("-" * 60)
    for level, stats in report["per_difficulty"].items():
        print(f"  {level:15s}: {stats['correct']:4d}/{stats['total']:4d} ({stats['accuracy']:5.1f}%) "
              f"[exec_fail: {stats['exec_failed']}, wrong: {stats['wrong_result']}]")

    # Per-database
    print()
    print("-" * 60)
    print("PER-DATABASE BREAKDOWN")
    print("-" * 60)
    sorted_dbs = sorted(report["per_database"], key=lambda x: -x["accuracy"])
    for db in sorted_dbs:
        bar = "█" * int(db["accuracy"] / 5)
        print(f"  {db['db_id']:30s}: {db['correct']:3d}/{db['total']:3d} ({db['accuracy']:5.1f}%) {bar}")

    # Error categories
    if report["error_categories"]:
        print()
        print("-" * 60)
        print("ERROR CATEGORIES")
        print("-" * 60)
        for cat, count in report["error_categories"].items():
            print(f"  {cat:20s}: {count}")

    print()
    print("=" * 60)


def generate_markdown_summary(report: Dict) -> str:
    """Generate markdown summary."""
    s = report["summary"]
    md = f"""# T11.1 Evaluation Summary

**Generated:** {report['timestamp']}  
**Predictions:** `{report['predictions_file']}`

## Summary

| Metric | Count | Rate |
|--------|-------|------|
| **Execution Accuracy** | {s['execution_correct']}/{s['total_examples']} | **{s['execution_accuracy']}%** |
| Exact Match | {s['exact_match_count']}/{s['total_examples']} | {s['exact_match_accuracy']}% |
| Exec Failures | {s['exec_fail_count']} | {s['exec_fail_rate']}% |
| Wrong Results | {s['wrong_result_count']} | {s['wrong_result_rate']}% |

## Per-Difficulty Breakdown

| Difficulty | Correct | Total | Accuracy | Exec Fail | Wrong |
|------------|---------|-------|----------|-----------|-------|
"""
    for level in ["simple", "moderate", "challenging"]:
        if level in report["per_difficulty"]:
            stats = report["per_difficulty"][level]
            md += f"| {level} | {stats['correct']} | {stats['total']} | {stats['accuracy']}% | {stats['exec_failed']} | {stats['wrong_result']} |\n"

    md += """
## Per-Database Breakdown

| Database | Correct | Total | Accuracy |
|----------|---------|-------|----------|
"""
    for db in sorted(report["per_database"], key=lambda x: -x["accuracy"]):
        md += f"| {db['db_id']} | {db['correct']} | {db['total']} | {db['accuracy']}% |\n"

    if report["error_categories"]:
        md += """
## Error Categories

| Category | Count |
|----------|-------|
"""
        for cat, count in report["error_categories"].items():
            md += f"| {cat} | {count} |\n"

    return md


def main():
    parser = argparse.ArgumentParser(
        description="T11.1 Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions_file",
        required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--db_dir",
        default=str(DEFAULT_DB_DIR),
        help="Path to dev_databases directory",
    )
    parser.add_argument(
        "--dev_json",
        default=str(DEFAULT_DEV_JSON),
        help="Path to dev.json file",
    )
    parser.add_argument(
        "--tied_append",
        default=str(DEFAULT_TIED_APPEND),
        help="Path to tied-append alternatives",
    )
    parser.add_argument(
        "--no_tied_append",
        action="store_true",
        help="Disable tied-append alternative gold SQL matching",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SQL execution timeout in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--generation_config",
        default=None,
        help="Path to generation_config.json for run manifest",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("T11.1 Evaluation Script")
    print("=" * 60)
    print(f"Predictions: {args.predictions_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"DB dir: {args.db_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    predictions = load_predictions(args.predictions_file)
    print(f"  Loaded {len(predictions)} predictions")
    
    dev_data = load_dev_data(args.dev_json)
    print(f"  Loaded {len(dev_data)} dev examples")
    
    tied_map = {}
    if not args.no_tied_append:
        tied_map = load_tied_append(args.tied_append)
        print(f"  Loaded {len(tied_map)} tied-append alternatives")
    else:
        print("  Tied-append disabled")
    
    # Evaluate
    results = evaluate_predictions(
        predictions=predictions,
        dev_data=dev_data,
        db_dir=args.db_dir,
        timeout=args.timeout,
        max_workers=args.workers,
        tied_map=tied_map,
    )
    
    # Generate report
    report = generate_report(results, args.predictions_file)
    
    # Save outputs
    report_path = os.path.join(args.output_dir, "eval_report_t11_1.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    summary_path = os.path.join(args.output_dir, "eval_summary_t11_1.md")
    with open(summary_path, 'w') as f:
        f.write(generate_markdown_summary(report))
    print(f"Summary saved to: {summary_path}")

    manifest = generate_run_manifest(
        predictions_file=args.predictions_file,
        report=report,
        generation_config_path=args.generation_config,
    )
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest saved to: {manifest_path}")
    
    # Save per-example results
    results_path = os.path.join(args.output_dir, "per_example_results.jsonl")
    with open(results_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Per-example results saved to: {results_path}")
    
    # Print summary
    print_report(report)
    
    print(f"\nOutputs:")
    print(f"  - {report_path}")
    print(f"  - {summary_path}")
    print(f"  - {manifest_path}")
    print(f"  - {results_path}")


if __name__ == "__main__":
    main()

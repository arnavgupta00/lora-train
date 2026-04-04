#!/usr/bin/env python3
"""
Compare Full vs Compact Schema Results

Compare evaluation results between full-schema and compact-schema modes.

Outputs:
- comparison_report.json - structured comparison
- comparison_summary.md - human-readable summary

Usage:
    python evaluation/bird_eval/compare_results.py \
        --full_eval ./results/full/eval/eval_report_full.json \
        --compact_eval ./results/compact/eval/eval_report_compact.json \
        --output_dir ./results/comparison
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


def load_report(report_path: str) -> Dict[str, Any]:
    """Load evaluation report from JSON file."""
    with open(report_path, 'r') as f:
        return json.load(f)


def load_per_example_results(results_path: str) -> Dict[int, Dict]:
    """Load per-example results and index by question_id."""
    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                qid = r.get("question_id")
                if qid is not None:
                    results[qid] = r
    return results


def compare_reports(
    full_report: Dict[str, Any],
    compact_report: Dict[str, Any],
    full_results: Optional[Dict[int, Dict]] = None,
    compact_results: Optional[Dict[int, Dict]] = None,
) -> Dict[str, Any]:
    """Compare full-schema and compact-schema evaluation reports."""
    full_summary = full_report["summary"]
    compact_summary = compact_report["summary"]
    
    # Overall comparison
    ex_delta = compact_summary["execution_accuracy"] - full_summary["execution_accuracy"]
    em_delta = compact_summary["exact_match_accuracy"] - full_summary["exact_match_accuracy"]
    exec_fail_delta = compact_summary["exec_fail_rate"] - full_summary["exec_fail_rate"]
    wrong_result_delta = compact_summary["wrong_result_rate"] - full_summary["wrong_result_rate"]
    
    # Per-difficulty comparison
    difficulty_comparison = {}
    for level in ["simple", "moderate", "challenging"]:
        full_stats = full_report["per_difficulty"].get(level, {})
        compact_stats = compact_report["per_difficulty"].get(level, {})
        
        full_acc = full_stats.get("accuracy", 0)
        compact_acc = compact_stats.get("accuracy", 0)
        
        difficulty_comparison[level] = {
            "full_accuracy": full_acc,
            "compact_accuracy": compact_acc,
            "delta": round(compact_acc - full_acc, 2),
            "full_total": full_stats.get("total", 0),
            "compact_total": compact_stats.get("total", 0),
        }
    
    # Per-database comparison
    db_comparison = []
    full_dbs = {db["db_id"]: db for db in full_report["per_database"]}
    compact_dbs = {db["db_id"]: db for db in compact_report["per_database"]}
    
    all_dbs = set(full_dbs.keys()) | set(compact_dbs.keys())
    for db_id in sorted(all_dbs):
        full_db = full_dbs.get(db_id, {})
        compact_db = compact_dbs.get(db_id, {})
        
        full_acc = full_db.get("accuracy", 0)
        compact_acc = compact_db.get("accuracy", 0)
        
        db_comparison.append({
            "db_id": db_id,
            "full_accuracy": full_acc,
            "compact_accuracy": compact_acc,
            "delta": round(compact_acc - full_acc, 2),
            "full_total": full_db.get("total", 0),
            "compact_total": compact_db.get("total", 0),
        })
    
    # Sort by delta (most improved first)
    db_comparison.sort(key=lambda x: -x["delta"])
    
    # Per-example analysis (if results available)
    example_analysis = None
    if full_results and compact_results:
        compact_wins = []  # Compact correct, full wrong
        full_wins = []  # Full correct, compact wrong
        both_correct = 0
        both_wrong = 0
        
        all_qids = set(full_results.keys()) | set(compact_results.keys())
        
        for qid in all_qids:
            full_r = full_results.get(qid, {})
            compact_r = compact_results.get(qid, {})
            
            full_correct = full_r.get("correct", False)
            compact_correct = compact_r.get("correct", False)
            
            if full_correct and compact_correct:
                both_correct += 1
            elif not full_correct and not compact_correct:
                both_wrong += 1
            elif compact_correct and not full_correct:
                compact_wins.append({
                    "question_id": qid,
                    "db_id": full_r.get("db_id") or compact_r.get("db_id"),
                    "difficulty": full_r.get("difficulty") or compact_r.get("difficulty"),
                    "full_error": full_r.get("pred_error") or ("wrong_result" if full_r.get("wrong_result") else None),
                    "compact_correct": True,
                })
            elif full_correct and not compact_correct:
                full_wins.append({
                    "question_id": qid,
                    "db_id": full_r.get("db_id") or compact_r.get("db_id"),
                    "difficulty": full_r.get("difficulty") or compact_r.get("difficulty"),
                    "compact_error": compact_r.get("pred_error") or ("wrong_result" if compact_r.get("wrong_result") else None),
                    "full_correct": True,
                })
        
        example_analysis = {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "compact_wins_count": len(compact_wins),
            "full_wins_count": len(full_wins),
            "net_delta": len(compact_wins) - len(full_wins),
            "compact_wins_sample": compact_wins[:10],
            "full_wins_sample": full_wins[:10],
        }
    
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "full_report_path": full_report.get("predictions_file", "unknown"),
        "compact_report_path": compact_report.get("predictions_file", "unknown"),
        "overall": {
            "full_execution_accuracy": full_summary["execution_accuracy"],
            "compact_execution_accuracy": compact_summary["execution_accuracy"],
            "execution_accuracy_delta": round(ex_delta, 2),
            "full_exact_match_accuracy": full_summary["exact_match_accuracy"],
            "compact_exact_match_accuracy": compact_summary["exact_match_accuracy"],
            "exact_match_delta": round(em_delta, 2),
            "full_exec_fail_rate": full_summary["exec_fail_rate"],
            "compact_exec_fail_rate": compact_summary["exec_fail_rate"],
            "exec_fail_delta": round(exec_fail_delta, 2),
            "full_wrong_result_rate": full_summary["wrong_result_rate"],
            "compact_wrong_result_rate": compact_summary["wrong_result_rate"],
            "wrong_result_delta": round(wrong_result_delta, 2),
        },
        "per_difficulty": difficulty_comparison,
        "per_database": db_comparison,
        "example_analysis": example_analysis,
    }
    
    return comparison


def generate_markdown_summary(comparison: Dict[str, Any]) -> str:
    """Generate human-readable markdown summary."""
    overall = comparison["overall"]
    
    # Determine overall winner
    ex_delta = overall["execution_accuracy_delta"]
    if abs(ex_delta) < 0.5:
        winner = "≈ Tie (within 0.5%)"
    elif ex_delta > 0:
        winner = f"✅ Compact schema wins (+{ex_delta}%)"
    else:
        winner = f"❌ Full schema wins ({ex_delta}%)"
    
    md = f"""# Full vs Compact Schema Comparison

**Generated:** {comparison['timestamp']}

## Overall Result: {winner}

### Summary

| Metric | Full Schema | Compact Schema | Delta |
|--------|-------------|----------------|-------|
| **Execution Accuracy** | {overall['full_execution_accuracy']}% | {overall['compact_execution_accuracy']}% | **{overall['execution_accuracy_delta']:+.2f}%** |
| Exact Match | {overall['full_exact_match_accuracy']}% | {overall['compact_exact_match_accuracy']}% | {overall['exact_match_delta']:+.2f}% |
| Exec Fail Rate | {overall['full_exec_fail_rate']}% | {overall['compact_exec_fail_rate']}% | {overall['exec_fail_delta']:+.2f}% |
| Wrong Result Rate | {overall['full_wrong_result_rate']}% | {overall['compact_wrong_result_rate']}% | {overall['wrong_result_delta']:+.2f}% |

## Per-Difficulty Breakdown

| Difficulty | Full | Compact | Delta |
|------------|------|---------|-------|
"""
    
    for level in ["simple", "moderate", "challenging"]:
        if level in comparison["per_difficulty"]:
            stats = comparison["per_difficulty"][level]
            delta_str = f"{stats['delta']:+.2f}%" if stats['delta'] != 0 else "0%"
            md += f"| {level} | {stats['full_accuracy']}% | {stats['compact_accuracy']}% | {delta_str} |\n"
    
    md += """
## Per-Database Breakdown (Sorted by Delta)

| Database | Full | Compact | Delta |
|----------|------|---------|-------|
"""
    
    for db in comparison["per_database"]:
        delta_str = f"{db['delta']:+.2f}%" if db['delta'] != 0 else "0%"
        md += f"| {db['db_id']} | {db['full_accuracy']}% | {db['compact_accuracy']}% | {delta_str} |\n"
    
    if comparison.get("example_analysis"):
        analysis = comparison["example_analysis"]
        md += f"""
## Per-Example Analysis

| Category | Count |
|----------|-------|
| Both Correct | {analysis['both_correct']} |
| Both Wrong | {analysis['both_wrong']} |
| Compact Wins (compact correct, full wrong) | {analysis['compact_wins_count']} |
| Full Wins (full correct, compact wrong) | {analysis['full_wins_count']} |
| **Net Delta** | **{analysis['net_delta']:+d}** |

### Sample: Compact Wins

"""
        if analysis['compact_wins_sample']:
            md += "| Question ID | Database | Difficulty | Full Error |\n"
            md += "|-------------|----------|------------|------------|\n"
            for ex in analysis['compact_wins_sample'][:5]:
                error = str(ex.get('full_error', ''))[:50]
                md += f"| {ex['question_id']} | {ex['db_id']} | {ex['difficulty']} | {error} |\n"
        else:
            md += "*No examples where compact wins.*\n"
        
        md += """
### Sample: Full Wins

"""
        if analysis['full_wins_sample']:
            md += "| Question ID | Database | Difficulty | Compact Error |\n"
            md += "|-------------|----------|------------|---------------|\n"
            for ex in analysis['full_wins_sample'][:5]:
                error = str(ex.get('compact_error', ''))[:50]
                md += f"| {ex['question_id']} | {ex['db_id']} | {ex['difficulty']} | {error} |\n"
        else:
            md += "*No examples where full wins.*\n"
    
    md += """
## Interpretation

- **Positive delta** = Compact schema performs better
- **Negative delta** = Full schema performs better
- **Exec Fail Delta**: Lower is better for compact (fewer syntax/column errors)
- **Wrong Result Delta**: Lower is better for compact (fewer logic errors)

### Key Observations

"""
    
    # Auto-generate observations
    if ex_delta > 1:
        md += "- ✅ Compact schema shows meaningful improvement over full schema\n"
    elif ex_delta < -1:
        md += "- ⚠️ Full schema outperforms compact schema - compaction may be too aggressive\n"
    else:
        md += "- ≈ Results are comparable between modes\n"
    
    if overall["exec_fail_delta"] < -1:
        md += "- ✅ Compact schema reduces execution failures (better column/table references)\n"
    elif overall["exec_fail_delta"] > 1:
        md += "- ⚠️ Compact schema increases execution failures (may be removing needed schema info)\n"
    
    if overall["wrong_result_delta"] < -1:
        md += "- ✅ Compact schema reduces wrong results (better query logic)\n"
    elif overall["wrong_result_delta"] > 1:
        md += "- ⚠️ Compact schema increases wrong results (may be missing context for correct joins/filters)\n"
    
    return md


def print_summary(comparison: Dict[str, Any]) -> None:
    """Print human-readable summary to console."""
    overall = comparison["overall"]
    
    print("\n" + "=" * 60)
    print("FULL vs COMPACT SCHEMA COMPARISON")
    print("=" * 60)
    
    ex_delta = overall["execution_accuracy_delta"]
    if abs(ex_delta) < 0.5:
        print("Result: ≈ Tie (within 0.5%)")
    elif ex_delta > 0:
        print(f"Result: ✅ Compact schema wins (+{ex_delta}%)")
    else:
        print(f"Result: ❌ Full schema wins ({ex_delta}%)")
    
    print()
    print(f"{'Metric':<25} {'Full':>10} {'Compact':>10} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Execution Accuracy':<25} {overall['full_execution_accuracy']:>9.1f}% {overall['compact_execution_accuracy']:>9.1f}% {overall['execution_accuracy_delta']:>+9.2f}%")
    print(f"{'Exact Match':<25} {overall['full_exact_match_accuracy']:>9.1f}% {overall['compact_exact_match_accuracy']:>9.1f}% {overall['exact_match_delta']:>+9.2f}%")
    print(f"{'Exec Fail Rate':<25} {overall['full_exec_fail_rate']:>9.1f}% {overall['compact_exec_fail_rate']:>9.1f}% {overall['exec_fail_delta']:>+9.2f}%")
    print(f"{'Wrong Result Rate':<25} {overall['full_wrong_result_rate']:>9.1f}% {overall['compact_wrong_result_rate']:>9.1f}% {overall['wrong_result_delta']:>+9.2f}%")
    
    print()
    print("-" * 60)
    print("PER-DIFFICULTY BREAKDOWN")
    print("-" * 60)
    for level in ["simple", "moderate", "challenging"]:
        if level in comparison["per_difficulty"]:
            stats = comparison["per_difficulty"][level]
            print(f"  {level:<15} Full: {stats['full_accuracy']:5.1f}%  Compact: {stats['compact_accuracy']:5.1f}%  Delta: {stats['delta']:+5.2f}%")
    
    if comparison.get("example_analysis"):
        analysis = comparison["example_analysis"]
        print()
        print("-" * 60)
        print("PER-EXAMPLE ANALYSIS")
        print("-" * 60)
        print(f"  Both correct:   {analysis['both_correct']}")
        print(f"  Both wrong:     {analysis['both_wrong']}")
        print(f"  Compact wins:   {analysis['compact_wins_count']}")
        print(f"  Full wins:      {analysis['full_wins_count']}")
        print(f"  Net delta:      {analysis['net_delta']:+d}")
    
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Full vs Compact Schema Evaluation Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full_eval",
        required=True,
        help="Path to full-schema eval report (eval_report_full.json)",
    )
    parser.add_argument(
        "--compact_eval",
        required=True,
        help="Path to compact-schema eval report (eval_report_compact.json)",
    )
    parser.add_argument(
        "--full_results",
        default="",
        help="Path to full-schema per_example_results.jsonl (optional, for detailed analysis)",
    )
    parser.add_argument(
        "--compact_results",
        default="",
        help="Path to compact-schema per_example_results.jsonl (optional, for detailed analysis)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for comparison results",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Full vs Compact Schema Comparison")
    print("=" * 60)
    print(f"Full eval: {args.full_eval}")
    print(f"Compact eval: {args.compact_eval}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reports
    print("Loading evaluation reports...")
    full_report = load_report(args.full_eval)
    compact_report = load_report(args.compact_eval)
    print(f"  Full: {full_report['summary']['total_examples']} examples, {full_report['summary']['execution_accuracy']}% EX")
    print(f"  Compact: {compact_report['summary']['total_examples']} examples, {compact_report['summary']['execution_accuracy']}% EX")
    
    # Load per-example results if available
    full_results = None
    compact_results = None
    
    if args.full_results and Path(args.full_results).exists():
        full_results = load_per_example_results(args.full_results)
        print(f"  Loaded {len(full_results)} full per-example results")
    else:
        # Try auto-detecting from report path
        full_dir = Path(args.full_eval).parent
        auto_path = full_dir / "per_example_results.jsonl"
        if auto_path.exists():
            full_results = load_per_example_results(str(auto_path))
            print(f"  Auto-loaded {len(full_results)} full per-example results")
    
    if args.compact_results and Path(args.compact_results).exists():
        compact_results = load_per_example_results(args.compact_results)
        print(f"  Loaded {len(compact_results)} compact per-example results")
    else:
        # Try auto-detecting from report path
        compact_dir = Path(args.compact_eval).parent
        auto_path = compact_dir / "per_example_results.jsonl"
        if auto_path.exists():
            compact_results = load_per_example_results(str(auto_path))
            print(f"  Auto-loaded {len(compact_results)} compact per-example results")
    
    # Compare
    print("\nComparing results...")
    comparison = compare_reports(full_report, compact_report, full_results, compact_results)
    
    # Save outputs
    report_path = os.path.join(args.output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison report saved to: {report_path}")
    
    summary_path = os.path.join(args.output_dir, "comparison_summary.md")
    with open(summary_path, 'w') as f:
        f.write(generate_markdown_summary(comparison))
    print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print_summary(comparison)
    
    print(f"\nOutputs:")
    print(f"  - {report_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()

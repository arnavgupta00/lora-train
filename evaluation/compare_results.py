#!/usr/bin/env python3
"""
Compare baseline vs LoRA fine-tuned model performance on BIRD benchmark
"""
import json
import sys
from pathlib import Path

def load_report(path):
    """Load evaluation report JSON"""
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)

def print_comparison(baseline_path, finetuned_path):
    """Print side-by-side comparison of results"""
    
    baseline = load_report(baseline_path)
    finetuned = load_report(finetuned_path)
    
    if not baseline:
        print(f"❌ Baseline report not found: {baseline_path}")
        print("   Run: nohup bash evaluation/run_bird_eval_baseline.sh > bird_eval_baseline.log 2>&1 &")
        return
    
    if not finetuned:
        print(f"❌ Fine-tuned report not found: {finetuned_path}")
        print("   Run: nohup bash evaluation/run_bird_eval.sh > bird_eval_t8.log 2>&1 &")
        return
    
    # Overall comparison
    print("=" * 80)
    print("BIRD BENCHMARK RESULTS - BASELINE vs FINE-TUNED (T8)")
    print("=" * 80)
    print()
    
    baseline_acc = baseline.get("overall_accuracy", 0)
    finetuned_acc = finetuned.get("overall_accuracy", 0)
    improvement = finetuned_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    
    print(f"{'Metric':<30} {'Baseline':<15} {'Fine-tuned (T8)':<15} {'Δ':<15}")
    print("-" * 80)
    print(f"{'Execution Accuracy':<30} {baseline_acc:>6.2f}%        {finetuned_acc:>6.2f}%        {improvement:>+6.2f}% ({improvement_pct:+.1f}%)")
    print(f"{'Correct':<30} {baseline.get('correct', 0):>6d}         {finetuned.get('correct', 0):>6d}         {finetuned.get('correct', 0) - baseline.get('correct', 0):>+6d}")
    print(f"{'Total':<30} {baseline.get('total', 0):>6d}         {finetuned.get('total', 0):>6d}")
    print()
    
    # Per-database comparison
    print("=" * 80)
    print("PER-DATABASE BREAKDOWN")
    print("=" * 80)
    print()
    print(f"{'Database':<30} {'Baseline':<15} {'Fine-tuned':<15} {'Δ':<15}")
    print("-" * 80)
    
    baseline_dbs = baseline.get("per_database", {})
    finetuned_dbs = finetuned.get("per_database", {})
    
    all_dbs = sorted(set(baseline_dbs.keys()) | set(finetuned_dbs.keys()))
    
    for db in all_dbs:
        base_acc = baseline_dbs.get(db, {}).get("accuracy", 0)
        ft_acc = finetuned_dbs.get(db, {}).get("accuracy", 0)
        delta = ft_acc - base_acc
        
        base_str = f"{base_acc:.1f}%"
        ft_str = f"{ft_acc:.1f}%"
        delta_str = f"{delta:+.1f}%"
        
        print(f"{db:<30} {base_str:<15} {ft_str:<15} {delta_str:<15}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if improvement > 0:
        print(f"✅ Fine-tuning IMPROVED performance by {improvement:.2f}% ({improvement_pct:+.1f}%)")
        print(f"   Baseline:    {baseline_acc:.2f}%")
        print(f"   Fine-tuned:  {finetuned_acc:.2f}%")
    elif improvement < 0:
        print(f"❌ Fine-tuning DEGRADED performance by {improvement:.2f}%")
        print(f"   This suggests overfitting or training issues")
    else:
        print(f"⚠️  No change in performance")
    
    print()
    
    # Context: Compare to public benchmarks
    print("=" * 80)
    print("COMPARISON TO PUBLIC BENCHMARKS")
    print("=" * 80)
    print()
    print("BIRD Dev Set Leaderboard (approximate):")
    print("  GPT-4 Turbo:          54.89%")
    print("  Claude Opus 4.6:      ~60-65% (estimated)")
    print("  GPT-3.5 Turbo:        40.08%")
    print("  Qwen2.5-7B-Instruct:  " + f"{baseline_acc:.2f}% ← Our baseline")
    print("  Qwen2.5-7B + T8 LoRA: " + f"{finetuned_acc:.2f}% ← Our fine-tuned")
    print()
    
    if finetuned_acc > 54.89:
        print("🎉 BEATS GPT-4 Turbo (54.89%)! Great for LinkedIn post!")
    elif finetuned_acc > 40.08:
        print("✅ BEATS GPT-3.5 Turbo (40.08%)!")
    
    if finetuned_acc > baseline_acc:
        print(f"✅ BEATS base model by {improvement:.2f}% - shows fine-tuning works!")
    
    print()

if __name__ == "__main__":
    baseline_path = "/workspace/lora-train/outputs/bird_evaluation/bird_eval_report_baseline.json"
    finetuned_path = "/workspace/lora-train/outputs/bird_evaluation/bird_eval_report_v3.json"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        baseline_path = sys.argv[1]
    if len(sys.argv) > 2:
        finetuned_path = sys.argv[2]
    
    print_comparison(baseline_path, finetuned_path)

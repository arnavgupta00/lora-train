#!/usr/bin/env python3
"""Merge T9 v2 with v3 batches and validate pattern distribution."""

import json
import re
from pathlib import Path
from collections import defaultdict

T9_DIR = Path("/Users/arnav/programming/lm/data/training/t9")

BATCH_FILES = [
    "batch1_join_order_limit.jsonl",
    "batch2_join_distinct_order.jsonl",
    "batch3_join_subquery_distinct.jsonl",
    "batch4_join_case_subquery.jsonl",
    "batch5_join_cte_order.jsonl",
    "batch6_join_window_order.jsonl",
]

PATTERN_REGEXES = {
    "JOIN": re.compile(r"\bJOIN\b", re.IGNORECASE),
    "ORDER_BY": re.compile(r"\bORDER\s+BY\b", re.IGNORECASE),
    "DISTINCT": re.compile(r"\bDISTINCT\b", re.IGNORECASE),
    "LIMIT": re.compile(r"\bLIMIT\b", re.IGNORECASE),
    "SUBQUERY": re.compile(r"\(\s*SELECT\b", re.IGNORECASE),
    "CASE": re.compile(r"\bCASE\s+WHEN\b", re.IGNORECASE),
    "GROUP_BY": re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
    "CTE": re.compile(r"\bWITH\s+\w+\s+AS\s*\(", re.IGNORECASE),
    "WINDOW": re.compile(r"\bOVER\s*\(", re.IGNORECASE),
}

TARGETS = {
    "JOIN": (72, 75),
    "ORDER_BY": (25, 30),
    "DISTINCT": (20, 22),
    "LIMIT": (20, 25),
    "SUBQUERY": (14, 16),
    "CASE": (12, 15),
    "GROUP_BY": (12, 14),
    "CTE": (5, 7),
    "WINDOW": (4, 5),
}


def extract_sql(example):
    """Extract SQL from ChatML format."""
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def detect_patterns(sql):
    """Detect all patterns in SQL."""
    patterns = []
    for name, regex in PATTERN_REGEXES.items():
        if regex.search(sql):
            patterns.append(name)
    return patterns


def main():
    # Load v2 base
    v2_path = T9_DIR / "train_v2.jsonl"
    examples = []
    
    print("Loading train_v2.jsonl...")
    with open(v2_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples from v2")
    
    # Load all batches
    for batch_file in BATCH_FILES:
        batch_path = T9_DIR / batch_file
        count = 0
        with open(batch_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                    count += 1
        print(f"  Added {count} examples from {batch_file}")
    
    print(f"\nTotal: {len(examples)} examples")
    
    # Analyze pattern distribution
    pattern_counts = defaultdict(int)
    for ex in examples:
        sql = extract_sql(ex)
        for pattern in detect_patterns(sql):
            pattern_counts[pattern] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("PATTERN DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"{'Pattern':<12} {'Count':>8} {'%':>8} {'Target':>12} {'Status':>10}")
    print("-" * 60)
    
    total = len(examples)
    all_good = True
    
    for pattern in ["JOIN", "ORDER_BY", "DISTINCT", "LIMIT", "SUBQUERY", 
                    "CASE", "GROUP_BY", "CTE", "WINDOW"]:
        count = pattern_counts[pattern]
        pct = (count / total) * 100
        target_lo, target_hi = TARGETS[pattern]
        
        if pct >= target_lo and pct <= target_hi:
            status = "✅"
        elif pct >= target_lo - 2 and pct <= target_hi + 2:
            status = "~✅"
        else:
            status = "❌"
            all_good = False
        
        print(f"{pattern:<12} {count:>8} {pct:>7.1f}% {target_lo}-{target_hi}%{'':<4} {status:>10}")
    
    print("-" * 60)
    
    # Check db_id distribution
    with_db = sum(1 for ex in examples if ex.get("db_id"))
    without_db = total - with_db
    print(f"\nWith db_id: {with_db} ({with_db/total*100:.1f}%)")
    print(f"Without db_id: {without_db} ({without_db/total*100:.1f}%)")
    
    # Write merged file
    output_path = T9_DIR / "train_v3.jsonl"
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(examples)} examples to {output_path}")
    
    # Create dev split (5% of original dev + some from new batches)
    dev_v2_path = T9_DIR / "dev_v2.jsonl"
    dev_examples = []
    with open(dev_v2_path) as f:
        for line in f:
            if line.strip():
                dev_examples.append(json.loads(line))
    
    dev_output = T9_DIR / "dev_v3.jsonl"
    with open(dev_output, "w") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(dev_examples)} examples to {dev_output}")
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All patterns within or near target range!")
    else:
        print("⚠️  Some patterns outside target range")
    print("=" * 60)


if __name__ == "__main__":
    main()

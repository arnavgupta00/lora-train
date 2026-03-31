#!/usr/bin/env python3
"""
Create t7 dataset by combining:
1. t3 existing data (23K examples)
2. BIRD train (9K examples)
3. Complex patterns (9.5K examples)

Total: ~42K examples
"""

import json
import random
from pathlib import Path
from collections import Counter
import hashlib

def dedupe_examples(examples):
    """Remove duplicates based on SQL content."""
    seen = set()
    unique = []
    for ex in examples:
        # Get SQL content
        sql = ex['messages'][2]['content'].strip().lower()
        sql_hash = hashlib.md5(sql.encode()).hexdigest()
        if sql_hash not in seen:
            seen.add(sql_hash)
            unique.append(ex)
    return unique


def load_jsonl(path):
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main():
    random.seed(42)
    
    print("Creating t7 dataset...")
    
    # Load t3 train data
    t3_path = Path("dataset/t3_test1000_rebalanced/all-all-train.qwen.jsonl")
    print(f"\n1. Loading t3 train from {t3_path}...")
    t3_examples = load_jsonl(t3_path)
    print(f"   Loaded {len(t3_examples)} examples")
    
    # Load BIRD train (ChatML format)
    bird_path = Path("dataset/bird_train_chatml.jsonl")
    print(f"\n2. Loading BIRD train from {bird_path}...")
    bird_examples = load_jsonl(bird_path)
    print(f"   Loaded {len(bird_examples)} examples")
    
    # Load complex patterns
    complex_path = Path("dataset/complex_patterns.jsonl")
    print(f"\n3. Loading complex patterns from {complex_path}...")
    complex_examples = load_jsonl(complex_path)
    print(f"   Loaded {len(complex_examples)} examples")
    
    # Combine all
    print("\n4. Combining datasets...")
    all_examples = t3_examples + bird_examples + complex_examples
    print(f"   Total before dedup: {len(all_examples)}")
    
    # Dedupe
    all_examples = dedupe_examples(all_examples)
    print(f"   Total after dedup: {len(all_examples)}")
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Create output directory
    output_dir = Path("dataset/t7")
    output_dir.mkdir(exist_ok=True)
    
    # Split: 90% train, 5% dev, 5% test
    n = len(all_examples)
    train_end = int(n * 0.9)
    dev_end = int(n * 0.95)
    
    train = all_examples[:train_end]
    dev = all_examples[train_end:dev_end]
    test = all_examples[dev_end:]
    
    # Save
    print("\n5. Saving splits...")
    
    with open(output_dir / "train.jsonl", 'w') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')
    print(f"   Train: {len(train)} examples")
    
    with open(output_dir / "dev.jsonl", 'w') as f:
        for ex in dev:
            f.write(json.dumps(ex) + '\n')
    print(f"   Dev: {len(dev)} examples")
    
    with open(output_dir / "test.jsonl", 'w') as f:
        for ex in test:
            f.write(json.dumps(ex) + '\n')
    print(f"   Test: {len(test)} examples")
    
    # Analyze pattern distribution in combined dataset
    print("\n6. SQL pattern analysis...")
    
    pattern_counts = {
        'cte': 0,
        'window': 0,
        'subquery': 0,
        'case': 0,
        'join': 0,
        'agg': 0,
        'like': 0,
        'having': 0
    }
    
    for ex in train:
        sql = ex['messages'][2]['content'].upper()
        if 'WITH ' in sql: pattern_counts['cte'] += 1
        if ' OVER(' in sql or ' OVER (' in sql: pattern_counts['window'] += 1
        if sql.count('SELECT') > 1: pattern_counts['subquery'] += 1
        if 'CASE ' in sql: pattern_counts['case'] += 1
        if 'JOIN' in sql: pattern_counts['join'] += 1
        if any(x in sql for x in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(']): pattern_counts['agg'] += 1
        if 'LIKE ' in sql: pattern_counts['like'] += 1
        if 'HAVING ' in sql: pattern_counts['having'] += 1
    
    total = len(train)
    print(f"   Pattern distribution (train, n={total}):")
    for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"      {k:10s}: {v:5d} ({100*v/total:5.1f}%)")
    
    # Save metadata
    metadata = {
        "sources": {
            "t3_train": len(t3_examples),
            "bird_train": len(bird_examples),
            "complex_patterns": len(complex_examples)
        },
        "splits": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test)
        },
        "total": len(all_examples),
        "patterns": {k: {"count": v, "pct": round(100*v/total, 2)} for k, v in pattern_counts.items()}
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ t7 dataset created in {output_dir}")
    print(f"   Total: {len(all_examples)} examples")


if __name__ == "__main__":
    main()

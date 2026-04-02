#!/usr/bin/env python3
"""
Merge T9 dataset components, deduplicate, and create final splits.
"""

import json
import hashlib
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

def load_jsonl(path):
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def save_jsonl(examples, path):
    """Save examples to JSONL."""
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

def extract_sql(example):
    """Extract SQL from various formats."""
    if 'messages' in example:
        for msg in example['messages']:
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    if 'SQL' in example:
        return example['SQL']
    if 'sql' in example:
        return example['sql']
    return ''

def sql_hash(sql):
    """Create hash of normalized SQL for deduplication."""
    normalized = ' '.join(sql.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()

def analyze_patterns(sql):
    """Analyze SQL patterns."""
    sql_upper = sql.upper()
    return {
        'join': bool(re.search(r'\bJOIN\b', sql_upper)),
        'order_by': bool(re.search(r'\bORDER\s+BY\b', sql_upper)),
        'distinct': bool(re.search(r'\bDISTINCT\b', sql_upper)),
        'limit': bool(re.search(r'\bLIMIT\b', sql_upper)),
        'subquery': sql_upper.count('SELECT') > 1,
        'case': bool(re.search(r'\bCASE\b', sql_upper)),
        'group_by': bool(re.search(r'\bGROUP\s+BY\b', sql_upper)),
        'cte': bool(re.search(r'\bWITH\b', sql_upper)),
        'window': bool(re.search(r'\bOVER\s*\(', sql_upper)),
        'backtick': '`' in sql,
        'aggregation': bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper)),
    }

def validate_example(example):
    """Basic validation of example format."""
    if 'messages' not in example:
        return False
    messages = example['messages']
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    
    has_user = any(m.get('role') == 'user' for m in messages)
    has_assistant = any(m.get('role') == 'assistant' for m in messages)
    
    sql = extract_sql(example)
    if not sql or len(sql.strip()) < 10:
        return False
    
    return has_user and has_assistant

def main():
    t9_dir = Path('/Users/arnav/programming/lm/data/training/t9')
    
    print("=" * 70)
    print("MERGING T9 DATASET")
    print("=" * 70)
    
    # Load all components
    print("\n1. Loading components...")
    
    core = load_jsonl(t9_dir / 'core_t9.jsonl')
    print(f"   Core: {len(core)} examples")
    
    backticks = load_jsonl(t9_dir / 'augment_backticks.jsonl')
    print(f"   Backticks: {len(backticks)} examples")
    
    subqueries = load_jsonl(t9_dir / 'augment_subqueries.jsonl')
    print(f"   Subqueries: {len(subqueries)} examples")
    
    # Merge all
    print("\n2. Merging and deduplicating...")
    all_examples = core + backticks + subqueries
    
    # Deduplicate
    seen_hashes = set()
    unique_examples = []
    
    for ex in all_examples:
        if not validate_example(ex):
            continue
        sql = extract_sql(ex)
        h = sql_hash(sql)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_examples.append(ex)
    
    print(f"   Before dedup: {len(all_examples)}")
    print(f"   After dedup: {len(unique_examples)}")
    
    # Analyze distribution
    print("\n3. Analyzing final distribution...")
    pattern_counts = Counter()
    for ex in unique_examples:
        sql = extract_sql(ex)
        patterns = analyze_patterns(sql)
        for pattern, present in patterns.items():
            if present:
                pattern_counts[pattern] += 1
    
    total = len(unique_examples)
    targets = {
        'join': (72, 75), 'order_by': (25, 30), 'distinct': (20, 22),
        'limit': (20, 25), 'subquery': (14, 16), 'case': (12, 15),
        'group_by': (12, 14), 'cte': (5, 7), 'window': (4, 5)
    }
    
    print(f"\n   Pattern Distribution ({total} total examples):")
    print(f"   {'Pattern':<12} {'Count':>6} {'Actual':>7} {'Target':>12} {'Status':>8}")
    print("   " + "-" * 50)
    
    for pattern in ['join', 'order_by', 'distinct', 'limit', 'subquery', 'case', 'group_by', 'cte', 'window', 'backtick', 'aggregation']:
        pct = 100 * pattern_counts[pattern] / total
        if pattern in targets:
            lo, hi = targets[pattern]
            if lo <= pct <= hi:
                status = "✅"
            elif pct < lo:
                status = "⬇️"
            else:
                status = "⬆️"
            target_str = f"{lo}-{hi}%"
        else:
            status = ""
            target_str = "-"
        print(f"   {pattern:<12} {pattern_counts[pattern]:>6} {pct:>6.1f}% {target_str:>12} {status:>8}")
    
    # Shuffle and split
    print("\n4. Creating train/dev splits...")
    random.shuffle(unique_examples)
    
    dev_size = min(800, int(len(unique_examples) * 0.06))
    dev = unique_examples[:dev_size]
    train = unique_examples[dev_size:]
    
    print(f"   Train: {len(train)} examples")
    print(f"   Dev: {len(dev)} examples")
    
    # Save final datasets
    print("\n5. Saving final datasets...")
    save_jsonl(train, t9_dir / 'train.jsonl')
    save_jsonl(dev, t9_dir / 'dev.jsonl')
    
    # Save metadata
    metadata = {
        'version': 'T9',
        'total_examples': len(unique_examples),
        'train_examples': len(train),
        'dev_examples': len(dev),
        'pattern_distribution': {
            pattern: {'count': count, 'pct': round(100 * count / total, 2)}
            for pattern, count in pattern_counts.items()
        },
        'sources': {
            'core': len(core),
            'backticks_augment': len(backticks),
            'subqueries_augment': len(subqueries),
        },
        'targets': targets,
    }
    
    with open(t9_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ T9 DATASET COMPLETE")
    print("=" * 70)
    print(f"\n   Total: {len(unique_examples)} examples")
    print(f"   Train: {len(train)} | Dev: {len(dev)}")
    print(f"\n   Key Metrics:")
    print(f"   - JOIN: {100*pattern_counts['join']/total:.1f}% (target: 72-75%)")
    print(f"   - CASE: {100*pattern_counts['case']/total:.1f}% (target: 12-15%)")
    print(f"   - Backtick: {pattern_counts['backtick']} examples")
    print(f"\n   Files saved to: {t9_dir}")

if __name__ == '__main__':
    main()

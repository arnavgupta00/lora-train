#!/usr/bin/env python3
"""
Merge all T9 v2 components into final dataset.
"""

import json
import hashlib
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def save_jsonl(examples, path):
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

def extract_sql(example):
    if 'messages' in example:
        for msg in example['messages']:
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    return ''

def sql_hash(sql):
    normalized = ' '.join(sql.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()

def analyze_patterns(sql):
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
    print("CREATING FINAL T9 v2 DATASET")
    print("=" * 70)
    
    # Load cleaned base
    print("\n1. Loading cleaned base...")
    base_train = load_jsonl(t9_dir / 'train_cleaned.jsonl')
    base_dev = load_jsonl(t9_dir / 'dev_cleaned.jsonl')
    print(f"   Base train: {len(base_train)}")
    print(f"   Base dev: {len(base_dev)}")
    
    # Load all augmentation files
    print("\n2. Loading augmentation files...")
    augments = {}
    augment_files = [
        'augment_cte.jsonl',
        'augment_window.jsonl',
        'augment_mixed_patterns.jsonl',
        'augment_backtick_domains.jsonl',
        'augment_domains.jsonl',
    ]
    
    for fname in augment_files:
        fpath = t9_dir / fname
        if fpath.exists():
            data = load_jsonl(fpath)
            augments[fname] = data
            print(f"   {fname}: {len(data)} examples")
    
    # Merge all
    print("\n3. Merging all examples...")
    all_examples = base_train + base_dev
    for name, data in augments.items():
        all_examples.extend(data)
    
    print(f"   Total before dedup: {len(all_examples)}")
    
    # Deduplicate
    print("\n4. Deduplicating...")
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
    
    print(f"   Unique valid: {len(unique_examples)}")
    
    # Analyze distribution
    print("\n5. Analyzing pattern distribution...")
    pattern_counts = Counter()
    for ex in unique_examples:
        sql = extract_sql(ex)
        patterns = analyze_patterns(sql)
        for pattern, present in patterns.items():
            if present:
                pattern_counts[pattern] += 1
    
    total = len(unique_examples)
    targets = {
        'join': (72, 75), 'order_by': (24, 30), 'distinct': (18, 22),
        'limit': (18, 25), 'subquery': (12, 16), 'case': (10, 15),
        'group_by': (10, 14), 'cte': (4, 7), 'window': (3, 5)
    }
    
    print(f"\n   Pattern Distribution ({total} examples):")
    print(f"   {'Pattern':<12} {'Count':>6} {'Actual':>7} {'Target':>12} {'Status'}")
    print("   " + "-" * 55)
    
    for pattern in ['join', 'order_by', 'distinct', 'limit', 'subquery', 'case', 'group_by', 'cte', 'window', 'backtick']:
        pct = 100 * pattern_counts[pattern] / total
        if pattern in targets:
            lo, hi = targets[pattern]
            if lo <= pct <= hi:
                status = "✅ OK"
            elif pct < lo:
                status = f"⬇️ LOW ({lo-pct:.1f}% gap)"
            else:
                status = f"⬆️ HIGH (+{pct-hi:.1f}%)"
            target_str = f"{lo}-{hi}%"
        else:
            status = ""
            target_str = "-"
        print(f"   {pattern:<12} {pattern_counts[pattern]:>6} {pct:>6.1f}% {target_str:>12} {status}")
    
    # Shuffle and split
    print("\n6. Creating train/dev splits...")
    random.shuffle(unique_examples)
    
    dev_size = min(700, int(len(unique_examples) * 0.06))
    dev = unique_examples[:dev_size]
    train = unique_examples[dev_size:]
    
    print(f"   Train: {len(train)}")
    print(f"   Dev: {len(dev)}")
    
    # Save
    print("\n7. Saving final dataset...")
    save_jsonl(train, t9_dir / 'train_v2.jsonl')
    save_jsonl(dev, t9_dir / 'dev_v2.jsonl')
    
    # Metadata
    metadata = {
        'version': 'T9_v2',
        'total_examples': len(unique_examples),
        'train_examples': len(train),
        'dev_examples': len(dev),
        'sources': {
            'cleaned_base': len(base_train) + len(base_dev),
            **{k: len(v) for k, v in augments.items()}
        },
        'pattern_distribution': {
            p: {'count': c, 'pct': round(100 * c / total, 2)}
            for p, c in pattern_counts.items()
        },
        'targets': targets
    }
    
    with open(t9_dir / 'metadata_v2.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ T9 v2 DATASET COMPLETE")
    print("=" * 70)
    print(f"\n   Total: {len(unique_examples)}")
    print(f"   Train: {len(train)} | Dev: {len(dev)}")
    print(f"\n   Key Improvements vs Original T9:")
    print(f"   - CTE: {pattern_counts['cte']} examples ({100*pattern_counts['cte']/total:.1f}%)")
    print(f"   - WINDOW: {pattern_counts['window']} examples ({100*pattern_counts['window']/total:.1f}%)")
    print(f"   - DISTINCT: {pattern_counts['distinct']} examples ({100*pattern_counts['distinct']/total:.1f}%)")
    print(f"   - SUBQUERY: {pattern_counts['subquery']} examples ({100*pattern_counts['subquery']/total:.1f}%)")
    print(f"   - Backticks: {pattern_counts['backtick']} examples")

if __name__ == '__main__':
    main()

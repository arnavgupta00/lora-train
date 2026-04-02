#!/usr/bin/env python3
"""
Create T9 core dataset from BIRD train + filtered examples.
Target: ~8,000 high-quality examples with proper distribution.
"""

import json
import hashlib
import random
from pathlib import Path
from collections import Counter
import re

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
    # Normalize: lowercase, remove extra whitespace
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

def main():
    base_path = Path('/Users/arnav/programming/lm/data')
    output_dir = base_path / 'training/t9'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CREATING T9 CORE DATASET")
    print("=" * 70)
    
    # Load BIRD train (ChatML format - already formatted)
    print("\n1. Loading BIRD train...")
    bird_train = load_jsonl(base_path / 'processed/bird_train_chatml.jsonl')
    print(f"   Loaded {len(bird_train)} BIRD examples")
    
    # Load Complex patterns for additional examples
    print("\n2. Loading Complex patterns...")
    complex_patterns = load_jsonl(base_path / 'processed/complex_patterns.jsonl')
    print(f"   Loaded {len(complex_patterns)} complex pattern examples")
    
    # Load T8 for reference (we'll filter from it)
    print("\n3. Loading T8 train...")
    t8_train = load_jsonl(base_path / 'training/t8/training/train.jsonl')
    print(f"   Loaded {len(t8_train)} T8 examples")
    
    # === STEP 1: Start with all BIRD examples (these ARE the benchmark) ===
    print("\n4. Building core from BIRD...")
    core_examples = []
    seen_hashes = set()
    
    for ex in bird_train:
        sql = extract_sql(ex)
        h = sql_hash(sql)
        if h not in seen_hashes and sql.strip():
            seen_hashes.add(h)
            patterns = analyze_patterns(sql)
            ex['_patterns'] = patterns
            ex['_source'] = 'bird'
            core_examples.append(ex)
    
    print(f"   BIRD unique examples: {len(core_examples)}")
    
    # === STEP 2: Add JOIN-heavy examples from T8 that aren't in BIRD ===
    print("\n5. Adding JOIN-heavy examples from T8...")
    join_added = 0
    
    for ex in t8_train:
        sql = extract_sql(ex)
        h = sql_hash(sql)
        if h not in seen_hashes and sql.strip():
            patterns = analyze_patterns(sql)
            # Only add if it has JOIN and doesn't have excessive CASE
            if patterns['join'] and not patterns['case']:
                seen_hashes.add(h)
                ex['_patterns'] = patterns
                ex['_source'] = 't8_join'
                core_examples.append(ex)
                join_added += 1
                if join_added >= 3000:  # Cap at 3000 additional JOINs
                    break
    
    print(f"   Added {join_added} JOIN examples from T8")
    
    # === STEP 3: Add subquery/DISTINCT examples from Complex patterns ===
    print("\n6. Adding subquery examples from Complex patterns...")
    subquery_added = 0
    
    for ex in complex_patterns:
        sql = extract_sql(ex)
        h = sql_hash(sql)
        if h not in seen_hashes and sql.strip():
            patterns = analyze_patterns(sql)
            # Add subquery examples
            if patterns['subquery'] and not patterns['case']:
                seen_hashes.add(h)
                ex['_patterns'] = patterns
                ex['_source'] = 'complex_subquery'
                core_examples.append(ex)
                subquery_added += 1
                if subquery_added >= 1000:
                    break
    
    print(f"   Added {subquery_added} subquery examples")
    
    # === STEP 4: Analyze current distribution ===
    print("\n7. Analyzing current distribution...")
    pattern_counts = Counter()
    for ex in core_examples:
        for pattern, present in ex.get('_patterns', {}).items():
            if present:
                pattern_counts[pattern] += 1
    
    total = len(core_examples)
    print(f"\n   Current distribution ({total} examples):")
    for pattern in ['join', 'order_by', 'distinct', 'limit', 'subquery', 'case', 'group_by', 'cte', 'window', 'backtick']:
        pct = 100 * pattern_counts[pattern] / total
        print(f"      {pattern:<12}: {pattern_counts[pattern]:>5} ({pct:>5.1f}%)")
    
    # === STEP 5: Rebalance - undersample CASE and ORDER BY ===
    print("\n8. Rebalancing distribution...")
    
    # Separate by patterns
    case_only = [ex for ex in core_examples if ex['_patterns'].get('case') and not ex['_patterns'].get('join')]
    order_only = [ex for ex in core_examples if ex['_patterns'].get('order_by') and not ex['_patterns'].get('join') and not ex['_patterns'].get('case')]
    other = [ex for ex in core_examples if ex not in case_only and ex not in order_only]
    
    print(f"   CASE-only examples: {len(case_only)} -> keeping 500")
    print(f"   ORDER-only examples: {len(order_only)} -> keeping 1000")
    print(f"   Other examples: {len(other)} -> keeping all")
    
    # Undersample
    random.shuffle(case_only)
    random.shuffle(order_only)
    
    rebalanced = other + case_only[:500] + order_only[:1000]
    random.shuffle(rebalanced)
    
    print(f"   Rebalanced total: {len(rebalanced)}")
    
    # === STEP 6: Final distribution check ===
    print("\n9. Final distribution:")
    pattern_counts = Counter()
    for ex in rebalanced:
        for pattern, present in ex.get('_patterns', {}).items():
            if present:
                pattern_counts[pattern] += 1
    
    total = len(rebalanced)
    for pattern in ['join', 'order_by', 'distinct', 'limit', 'subquery', 'case', 'group_by']:
        pct = 100 * pattern_counts[pattern] / total
        print(f"      {pattern:<12}: {pattern_counts[pattern]:>5} ({pct:>5.1f}%)")
    
    # === STEP 7: Save core dataset ===
    print("\n10. Saving core dataset...")
    
    # Remove internal fields before saving
    clean_examples = []
    for ex in rebalanced:
        clean_ex = {k: v for k, v in ex.items() if not k.startswith('_')}
        clean_examples.append(clean_ex)
    
    output_path = output_dir / 'core_t9.jsonl'
    save_jsonl(clean_examples, output_path)
    print(f"   Saved {len(clean_examples)} examples to {output_path}")
    
    # Save metadata
    metadata = {
        'total_examples': len(clean_examples),
        'sources': {
            'bird': len([ex for ex in rebalanced if ex.get('_source') == 'bird']),
            't8_join': len([ex for ex in rebalanced if ex.get('_source') == 't8_join']),
            'complex_subquery': len([ex for ex in rebalanced if ex.get('_source') == 'complex_subquery']),
        },
        'pattern_distribution': {
            pattern: {'count': count, 'pct': round(100 * count / total, 2)}
            for pattern, count in pattern_counts.items()
        }
    }
    
    with open(output_dir / 'core_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Core T9 dataset created: {len(clean_examples)} examples")
    print(f"   JOIN percentage: {100 * pattern_counts['join'] / total:.1f}%")
    print(f"   CASE percentage: {100 * pattern_counts['case'] / total:.1f}%")

if __name__ == '__main__':
    main()

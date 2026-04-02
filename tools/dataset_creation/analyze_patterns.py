#!/usr/bin/env python3
"""
Analyze SQL pattern distributions across all data sources.
Compare against BIRD benchmark targets for T9 creation.
"""

import json
import re
from pathlib import Path
from collections import Counter

def load_jsonl(path):
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def extract_sql(example):
    """Extract SQL from various formats."""
    # ChatML format
    if 'messages' in example:
        for msg in example['messages']:
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    # Raw BIRD format
    if 'SQL' in example:
        return example['SQL']
    if 'sql' in example:
        return example['sql']
    return ''

def analyze_patterns(sql):
    """Analyze SQL patterns in a query."""
    sql_upper = sql.upper()
    patterns = {
        'join': bool(re.search(r'\bJOIN\b', sql_upper)),
        'inner_join': bool(re.search(r'\bINNER\s+JOIN\b', sql_upper)),
        'left_join': bool(re.search(r'\bLEFT\s+(OUTER\s+)?JOIN\b', sql_upper)),
        'order_by': bool(re.search(r'\bORDER\s+BY\b', sql_upper)),
        'distinct': bool(re.search(r'\bDISTINCT\b', sql_upper)),
        'limit': bool(re.search(r'\bLIMIT\b', sql_upper)),
        'subquery': sql_upper.count('SELECT') > 1,
        'case': bool(re.search(r'\bCASE\b', sql_upper)),
        'group_by': bool(re.search(r'\bGROUP\s+BY\b', sql_upper)),
        'having': bool(re.search(r'\bHAVING\b', sql_upper)),
        'cte': bool(re.search(r'\bWITH\b', sql_upper)),
        'window': bool(re.search(r'\bOVER\s*\(', sql_upper)),
        'union': bool(re.search(r'\bUNION\b', sql_upper)),
        'like': bool(re.search(r'\bLIKE\b', sql_upper)),
        'in_clause': bool(re.search(r'\bIN\s*\(', sql_upper)),
        'exists': bool(re.search(r'\bEXISTS\b', sql_upper)),
        'backtick': '`' in sql,
        'aggregation': bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper)),
    }
    return patterns

def analyze_dataset(examples, name):
    """Analyze pattern distribution in a dataset."""
    pattern_counts = Counter()
    total = 0
    
    for ex in examples:
        sql = extract_sql(ex)
        if not sql:
            continue
        total += 1
        patterns = analyze_patterns(sql)
        for pattern, present in patterns.items():
            if present:
                pattern_counts[pattern] += 1
    
    return {
        'name': name,
        'total': total,
        'patterns': {k: {'count': v, 'pct': round(100 * v / total, 2) if total > 0 else 0} 
                     for k, v in pattern_counts.items()}
    }

def main():
    base_path = Path('/Users/arnav/programming/lm/data')
    
    # Define data sources to analyze
    sources = [
        ('BIRD Train (raw)', base_path / 'raw/bird_train.jsonl'),
        ('BIRD Train (ChatML)', base_path / 'processed/bird_train_chatml.jsonl'),
        ('Spider+BIRD', base_path / 'processed/bird_spider_train.jsonl'),
        ('Complex Patterns', base_path / 'processed/complex_patterns.jsonl'),
        ('T8 Train', base_path / 'training/t8/training/train.jsonl'),
    ]
    
    # BIRD benchmark targets (from T9_SPECIFICATION.md)
    targets = {
        'join': 74.3,
        'order_by': 24.3,
        'distinct': 22.3,
        'limit': 18.5,
        'subquery': 15.2,
        'case': 13.3,
        'group_by': 11.9,
        'cte': 6.6,
        'window': 4.4,
    }
    
    results = []
    
    print("=" * 80)
    print("T9 DATASET PATTERN ANALYSIS")
    print("=" * 80)
    
    for name, path in sources:
        if path.exists():
            print(f"\nLoading {name}...")
            examples = load_jsonl(path)
            result = analyze_dataset(examples, name)
            results.append(result)
            print(f"  Total examples: {result['total']}")
        else:
            print(f"\n[SKIP] {name} - file not found: {path}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("PATTERN DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    # Key patterns to compare
    key_patterns = ['join', 'order_by', 'distinct', 'limit', 'subquery', 
                    'case', 'group_by', 'cte', 'window', 'backtick', 'aggregation']
    
    # Header
    header = f"{'Pattern':<15} {'BIRD Target':<12}"
    for r in results:
        header += f" {r['name'][:12]:<12}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for pattern in key_patterns:
        row = f"{pattern:<15}"
        target = targets.get(pattern, '-')
        if isinstance(target, (int, float)):
            row += f" {target:>10.1f}%"
        else:
            row += f" {target:>11}"
        
        for r in results:
            pct = r['patterns'].get(pattern, {}).get('pct', 0)
            row += f" {pct:>10.1f}%"
        print(row)
    
    # Print gaps analysis
    print("\n" + "=" * 80)
    print("GAP ANALYSIS vs BIRD TARGETS")
    print("=" * 80)
    
    # Use T8 as reference
    t8_result = next((r for r in results if 'T8' in r['name']), None)
    if t8_result:
        print(f"\nT8 Dataset Gaps:")
        for pattern, target in targets.items():
            t8_pct = t8_result['patterns'].get(pattern, {}).get('pct', 0)
            gap = t8_pct - target
            status = "✅" if abs(gap) < 3 else ("⬆️ over" if gap > 0 else "⬇️ under")
            print(f"  {pattern:<15}: T8={t8_pct:>5.1f}% vs target={target:>5.1f}%  gap={gap:>+6.1f}%  {status}")
    
    # Save results
    output_path = base_path / 'training/t9_pattern_analysis.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'targets': targets,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Analysis saved to {output_path}")
    
    return results

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Dataset Distribution Analysis Tool

Analyzes training data distribution and compares against BIRD benchmark patterns.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def count_patterns(sql: str) -> dict:
    """Count SQL patterns in a query."""
    sql_upper = sql.upper()
    patterns = {
        'cte': 1 if ('WITH ' in sql_upper and 'AS (' in sql_upper) else 0,
        'window': 1 if ('OVER (' in sql_upper or 'OVER(' in sql_upper) else 0,
        'case': 1 if ('CASE ' in sql_upper or 'CASE\n' in sql_upper) else 0,
        'subquery': 1 if sql_upper.count('SELECT') > 1 else 0,
        'join': 1 if ' JOIN ' in sql_upper else 0,
        'group_by': 1 if 'GROUP BY' in sql_upper else 0,
        'having': 1 if 'HAVING' in sql_upper else 0,
        'union': 1 if 'UNION' in sql_upper else 0,
        'distinct': 1 if 'DISTINCT' in sql_upper else 0,
        'limit': 1 if 'LIMIT' in sql_upper else 0,
        'order': 1 if 'ORDER BY' in sql_upper else 0,
        'between': 1 if 'BETWEEN' in sql_upper else 0,
    }
    return patterns


def analyze_dataset(filepath: str) -> dict:
    """Analyze a JSONL dataset file."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    
    totals = defaultdict(int)
    total_examples = 0
    schema_ddl = 0
    backtick_cols = 0
    
    with open(filepath) as f:
        for line in f:
            total_examples += 1
            data = json.loads(line)
            
            # Extract SQL and schema from messages
            messages = data.get('messages', [])
            sql = ''
            schema = ''
            for msg in messages:
                if msg['role'] == 'assistant':
                    sql = msg['content']
                if msg['role'] == 'user':
                    schema = msg['content']
            
            # Count patterns
            patterns = count_patterns(sql)
            for k, v in patterns.items():
                totals[k] += v
            
            # Schema analysis
            if 'CREATE TABLE' in schema:
                schema_ddl += 1
            if '`' in schema:
                backtick_cols += 1
    
    return {
        'filepath': str(filepath),
        'total_examples': total_examples,
        'patterns': {k: {'count': v, 'pct': v / total_examples * 100} 
                     for k, v in totals.items()},
        'schema_ddl_pct': schema_ddl / total_examples * 100,
        'backtick_cols_pct': backtick_cols / total_examples * 100,
    }


# BIRD benchmark reference distribution
BIRD_REFERENCE = {
    'join': 74.3,
    'order': 24.3,
    'distinct': 22.3,
    'limit': 18.5,
    'subquery': 15.2,
    'case': 13.3,
    'group_by': 11.9,
    'cte': 6.6,
    'between': 5.5,
    'window': 4.4,
    'having': 1.9,
    'union': 0.3,
}


def compare_to_bird(analysis: dict) -> dict:
    """Compare dataset distribution to BIRD benchmark."""
    comparisons = {}
    for pattern, bird_pct in BIRD_REFERENCE.items():
        dataset_pct = analysis['patterns'].get(pattern, {}).get('pct', 0)
        diff = dataset_pct - bird_pct
        comparisons[pattern] = {
            'bird_pct': bird_pct,
            'dataset_pct': round(dataset_pct, 1),
            'diff': round(diff, 1),
            'status': '✓' if abs(diff) <= 5 else ('↑' if diff > 0 else '↓'),
        }
    return comparisons


def print_analysis(analysis: dict, comparison: dict = None):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"Dataset: {analysis['filepath']}")
    print(f"Total Examples: {analysis['total_examples']:,}")
    print(f"DDL Schema: {analysis['schema_ddl_pct']:.1f}%")
    print(f"Backtick Columns: {analysis['backtick_cols_pct']:.1f}%")
    print(f"{'='*60}")
    
    if comparison:
        print(f"\n{'Pattern':<15} {'Dataset %':<12} {'BIRD %':<12} {'Diff':<10} {'Status'}")
        print("-" * 55)
        for pattern in sorted(BIRD_REFERENCE.keys(), 
                              key=lambda x: -BIRD_REFERENCE[x]):
            c = comparison[pattern]
            print(f"{pattern:<15} {c['dataset_pct']:<12.1f} {c['bird_pct']:<12.1f} "
                  f"{c['diff']:>+8.1f}   {c['status']}")
    else:
        print(f"\n{'Pattern':<15} {'Count':<12} {'Percentage'}")
        print("-" * 40)
        for pattern, data in sorted(analysis['patterns'].items(), 
                                     key=lambda x: -x[1]['pct']):
            print(f"{pattern:<15} {data['count']:<12,} {data['pct']:.1f}%")


def main():
    if len(sys.argv) < 2:
        # Default: analyze common paths
        paths = [
            'data/training/t7/train.jsonl',
            'data/training/t8/training/train.jsonl',
            'data/processed/bird_dev_chatml.jsonl',
        ]
    else:
        paths = sys.argv[1:]
    
    for path in paths:
        analysis = analyze_dataset(path)
        if analysis:
            comparison = compare_to_bird(analysis)
            print_analysis(analysis, comparison)


if __name__ == '__main__':
    main()

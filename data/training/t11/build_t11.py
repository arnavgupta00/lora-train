#!/usr/bin/env python3
"""
T11 Dataset Builder

Builds T11 mixed full/compact schema dataset from T10.

Usage:
    python build_t11.py [--t10-dir PATH] [--output-dir PATH] [--dry-run]

T11 = T10 + 50% compact-schema examples
- Per-db_id alternating assignment (deterministic)
- Conservative programmatic compacting
- Fallback to full schema when uncertain
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "t10"))

from t11_utils import (
    assign_schema_modes_per_db,
    build_compact_schema,
    build_t11_example,
    validate_gold_sql_unchanged,
    validate_no_invention,
    validate_sql_coverage,
    parse_schema,
    get_all_schema_identifiers,
    _extract_question,
    _extract_schema,
    _extract_hints,
    _extract_gold_sql,
)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[Dict], path: Path) -> None:
    """Save examples to JSONL file."""
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def process_dataset(
    t10_examples: List[Dict],
    dataset_name: str,
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Process a T10 dataset into T11 format.
    
    Returns:
        (t11_examples, stats)
    """
    stats = {
        'total': len(t10_examples),
        'full_schema': 0,
        'compact_schema': 0,
        'compaction_success': 0,
        'compaction_fallback_parse_error': 0,
        'compaction_fallback_empty_result': 0,
        'compaction_fallback_sql_coverage': 0,
        'total_original_schema_len': 0,
        'total_compact_schema_len': 0,
        'per_db_stats': defaultdict(lambda: {'full': 0, 'compact': 0}),
    }
    
    # Assign schema modes per-db
    print(f"  Assigning schema modes for {dataset_name}...")
    assignments = assign_schema_modes_per_db(t10_examples)
    
    # Process each example
    t11_examples = []
    
    for i, t10_ex in enumerate(t10_examples):
        if verbose and i % 1000 == 0:
            print(f"  Processing example {i}/{len(t10_examples)}...")
        
        db_id = t10_ex.get('db_id', 'unknown')
        question = _extract_question(t10_ex)
        schema_mode = assignments.get((db_id, question), 'full')
        
        stats['per_db_stats'][db_id][schema_mode] += 1
        
        if schema_mode == 'full':
            # Keep example unchanged
            t11_ex = build_t11_example(t10_ex, 'full')
            stats['full_schema'] += 1
        else:
            # Attempt compaction
            full_schema = _extract_schema(t10_ex)
            gold_sql = _extract_gold_sql(t10_ex)
            hints = _extract_hints(t10_ex)
            
            compaction = build_compact_schema(full_schema, gold_sql, question, hints)
            
            stats['total_original_schema_len'] += compaction.original_len
            stats['total_compact_schema_len'] += compaction.compact_len
            
            # Additional validation: check SQL coverage
            compaction_valid = compaction.status == 'success'
            if compaction_valid:
                # Verify all SQL-referenced identifiers are in compact schema
                full_info = parse_schema(full_schema)
                full_tables, full_columns = get_all_schema_identifiers(full_info)
                
                valid, missing = validate_sql_coverage(
                    compaction.compact_schema, gold_sql, full_tables, full_columns
                )
                if not valid:
                    # SQL coverage failure - fallback to full schema
                    compaction_valid = False
                    compaction = compaction._replace(
                        status="fallback_sql_coverage",
                        fallback_reason=f"Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                    ) if hasattr(compaction, '_replace') else None
                    # If CompactionResult is not a namedtuple, handle differently
                    if compaction is None:
                        from t11_utils import CompactionResult
                        compaction = CompactionResult(
                            compact_schema=full_schema,
                            original_len=len(full_schema),
                            compact_len=len(full_schema),
                            status="fallback_sql_coverage",
                            fallback_reason=f"Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                        )
            
            if compaction_valid:
                stats['compaction_success'] += 1
                stats['compact_schema'] += 1
                t11_ex = build_t11_example(t10_ex, 'compact', compaction)
            else:
                # Fallback to full schema
                if 'parse_error' in compaction.status:
                    stats['compaction_fallback_parse_error'] += 1
                elif 'sql_coverage' in compaction.status:
                    stats['compaction_fallback_sql_coverage'] += 1
                else:
                    stats['compaction_fallback_empty_result'] += 1
                stats['full_schema'] += 1
                
                # Build as full but record the failed attempt
                t11_ex = build_t11_example(t10_ex, 'full')
                t11_ex['compaction_attempted'] = True
                t11_ex['compaction_status'] = compaction.status
                t11_ex['compaction_fallback_reason'] = compaction.fallback_reason
        
        t11_examples.append(t11_ex)
    
    return t11_examples, stats


def validate_examples(
    t10_examples: List[Dict],
    t11_examples: List[Dict],
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate T11 examples against T10.
    
    Returns:
        (all_valid, list_of_errors)
    """
    errors = []
    
    if len(t10_examples) != len(t11_examples):
        errors.append(f"Count mismatch: T10={len(t10_examples)}, T11={len(t11_examples)}")
        return False, errors
    
    for i, (t10_ex, t11_ex) in enumerate(zip(t10_examples, t11_examples)):
        # Validate gold SQL unchanged
        if not validate_gold_sql_unchanged(t10_ex, t11_ex):
            errors.append(f"Example {i}: Gold SQL changed")
        
        # Validate system prompt
        t10_system = t10_ex['messages'][0]['content']
        t11_system = t11_ex['messages'][0]['content']
        if t10_system != t11_system:
            errors.append(f"Example {i}: System prompt changed")
        
        # For compact examples, validate no invention
        if t11_ex.get('schema_mode') == 'compact' and t11_ex.get('compaction_status') == 'success':
            full_schema = _extract_schema(t10_ex)
            compact_schema = _extract_schema(t11_ex)
            
            valid, msg = validate_no_invention(full_schema, compact_schema)
            if not valid:
                errors.append(f"Example {i}: Invention detected - {msg}")
            
            # Validate SQL coverage
            gold_sql = _extract_gold_sql(t10_ex)
            full_info = parse_schema(full_schema)
            full_tables, full_columns = get_all_schema_identifiers(full_info)
            
            valid, missing = validate_sql_coverage(compact_schema, gold_sql, full_tables, full_columns)
            if not valid:
                errors.append(f"Example {i}: SQL coverage failure - missing {missing}")
        
        if verbose and errors and len(errors) <= 10:
            print(f"  Validation error at example {i}")
    
    return len(errors) == 0, errors


def find_example_for_summary(t11_examples: List[Dict]) -> Optional[Dict]:
    """Find a good example to show in the summary."""
    for ex in t11_examples:
        if ex.get('schema_mode') == 'compact' and ex.get('compaction_status') == 'success':
            original_len = ex.get('original_schema_len', 0)
            compact_len = ex.get('compact_schema_len', 0)
            if original_len > 500 and compact_len < original_len * 0.5:
                return ex
    return None


def generate_summary(
    train_stats: Dict,
    dev_stats: Dict,
    train_examples: List[Dict],
    dev_examples: List[Dict],
    t10_train: List[Dict],
    output_dir: Path
) -> Dict:
    """Generate build summary."""
    # Calculate averages
    total_compact_attempts = (
        train_stats['compaction_success'] + 
        train_stats['compaction_fallback_parse_error'] + 
        train_stats['compaction_fallback_empty_result'] +
        dev_stats['compaction_success'] + 
        dev_stats['compaction_fallback_parse_error'] + 
        dev_stats['compaction_fallback_empty_result']
    )
    
    total_original_len = train_stats['total_original_schema_len'] + dev_stats['total_original_schema_len']
    total_compact_len = train_stats['total_compact_schema_len'] + dev_stats['total_compact_schema_len']
    
    avg_original = total_original_len / total_compact_attempts if total_compact_attempts > 0 else 0
    avg_compact = total_compact_len / total_compact_attempts if total_compact_attempts > 0 else 0
    avg_reduction = (1 - avg_compact / avg_original) * 100 if avg_original > 0 else 0
    
    # Find example for before/after
    example_ex = find_example_for_summary(train_examples) or find_example_for_summary(dev_examples)
    
    example_before_after = None
    if example_ex:
        # Find matching T10 example
        db_id = example_ex.get('db_id')
        question = _extract_question(example_ex)
        for t10_ex in t10_train:
            if t10_ex.get('db_id') == db_id and _extract_question(t10_ex) == question:
                example_before_after = {
                    'db_id': db_id,
                    'question': question,
                    'full_schema_snippet': _extract_schema(t10_ex)[:500] + '...' if len(_extract_schema(t10_ex)) > 500 else _extract_schema(t10_ex),
                    'compact_schema_snippet': _extract_schema(example_ex),
                    'gold_sql': _extract_gold_sql(example_ex),
                    'original_len': example_ex.get('original_schema_len'),
                    'compact_len': example_ex.get('compact_schema_len'),
                }
                break
    
    summary = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'train_count': train_stats['total'],
        'dev_count': dev_stats['total'],
        'full_schema_count': train_stats['full_schema'] + dev_stats['full_schema'],
        'compact_schema_count': train_stats['compact_schema'] + dev_stats['compact_schema'],
        'compact_ratio': (train_stats['compact_schema'] + dev_stats['compact_schema']) / (train_stats['total'] + dev_stats['total']),
        'compaction_status_breakdown': {
            'success': train_stats['compaction_success'] + dev_stats['compaction_success'],
            'fallback_parse_error': train_stats['compaction_fallback_parse_error'] + dev_stats['compaction_fallback_parse_error'],
            'fallback_empty_result': train_stats['compaction_fallback_empty_result'] + dev_stats['compaction_fallback_empty_result'],
            'fallback_sql_coverage': train_stats.get('compaction_fallback_sql_coverage', 0) + dev_stats.get('compaction_fallback_sql_coverage', 0),
            'not_applicable': train_stats['full_schema'] + dev_stats['full_schema'],
        },
        'avg_full_schema_len': round(avg_original, 1),
        'avg_compact_schema_len': round(avg_compact, 1),
        'avg_reduction_pct': round(avg_reduction, 1),
        'train_stats': {
            'total': train_stats['total'],
            'full_schema': train_stats['full_schema'],
            'compact_schema': train_stats['compact_schema'],
            'compaction_success': train_stats['compaction_success'],
            'fallback_sql_coverage': train_stats.get('compaction_fallback_sql_coverage', 0),
        },
        'dev_stats': {
            'total': dev_stats['total'],
            'full_schema': dev_stats['full_schema'],
            'compact_schema': dev_stats['compact_schema'],
            'compaction_success': dev_stats['compaction_success'],
            'fallback_sql_coverage': dev_stats.get('compaction_fallback_sql_coverage', 0),
        },
        'example_before_after': example_before_after,
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Build T11 dataset from T10')
    parser.add_argument('--t10-dir', type=str, default='data/training/t10',
                        help='Directory containing T10 dataset')
    parser.add_argument('--output-dir', type=str, default='data/training/t11',
                        help='Output directory for T11 dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process but do not write output files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate output (default: True)')
    parser.add_argument('--no-validate', action='store_false', dest='validate',
                        help='Skip validation')
    args = parser.parse_args()
    
    t10_dir = Path(args.t10_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("T11 Dataset Builder")
    print("=" * 60)
    print(f"T10 directory: {t10_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load T10 datasets
    print("Loading T10 datasets...")
    t10_train = load_jsonl(t10_dir / 'train_t10.jsonl')
    t10_dev = load_jsonl(t10_dir / 'dev_t10.jsonl')
    print(f"  Loaded {len(t10_train)} train examples")
    print(f"  Loaded {len(t10_dev)} dev examples")
    print()
    
    # Process train dataset
    print("Processing train dataset...")
    t11_train, train_stats = process_dataset(t10_train, 'train', verbose=args.verbose)
    print(f"  Full schema: {train_stats['full_schema']}")
    print(f"  Compact schema: {train_stats['compact_schema']}")
    print(f"  Compaction success: {train_stats['compaction_success']}")
    print(f"  Fallback (parse error): {train_stats['compaction_fallback_parse_error']}")
    print(f"  Fallback (empty result): {train_stats['compaction_fallback_empty_result']}")
    print()
    
    # Process dev dataset
    print("Processing dev dataset...")
    t11_dev, dev_stats = process_dataset(t10_dev, 'dev', verbose=args.verbose)
    print(f"  Full schema: {dev_stats['full_schema']}")
    print(f"  Compact schema: {dev_stats['compact_schema']}")
    print(f"  Compaction success: {dev_stats['compaction_success']}")
    print(f"  Fallback (parse error): {dev_stats['compaction_fallback_parse_error']}")
    print(f"  Fallback (empty result): {dev_stats['compaction_fallback_empty_result']}")
    print()
    
    # Validate
    if args.validate:
        print("Validating output...")
        train_valid, train_errors = validate_examples(t10_train, t11_train, verbose=args.verbose)
        dev_valid, dev_errors = validate_examples(t10_dev, t11_dev, verbose=args.verbose)
        
        if train_valid:
            print("  Train validation: PASSED")
        else:
            print(f"  Train validation: FAILED ({len(train_errors)} errors)")
            for err in train_errors[:5]:
                print(f"    - {err}")
            if len(train_errors) > 5:
                print(f"    ... and {len(train_errors) - 5} more")
        
        if dev_valid:
            print("  Dev validation: PASSED")
        else:
            print(f"  Dev validation: FAILED ({len(dev_errors)} errors)")
            for err in dev_errors[:5]:
                print(f"    - {err}")
            if len(dev_errors) > 5:
                print(f"    ... and {len(dev_errors) - 5} more")
        print()
    
    # Generate summary
    print("Generating summary...")
    summary = generate_summary(train_stats, dev_stats, t11_train, t11_dev, t10_train, output_dir)
    
    # Print summary
    print()
    print("=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    print(f"Train examples: {summary['train_count']}")
    print(f"Dev examples: {summary['dev_count']}")
    print(f"Full schema examples: {summary['full_schema_count']}")
    print(f"Compact schema examples: {summary['compact_schema_count']}")
    print(f"Compact ratio: {summary['compact_ratio']:.1%}")
    print(f"Average full schema length: {summary['avg_full_schema_len']:.0f} chars")
    print(f"Average compact schema length: {summary['avg_compact_schema_len']:.0f} chars")
    print(f"Average reduction: {summary['avg_reduction_pct']:.1f}%")
    print()
    print("Compaction status breakdown:")
    for status, count in summary['compaction_status_breakdown'].items():
        print(f"  {status}: {count}")
    
    if summary['example_before_after']:
        ex = summary['example_before_after']
        print()
        print("-" * 60)
        print("EXAMPLE BEFORE/AFTER")
        print("-" * 60)
        print(f"DB: {ex['db_id']}")
        print(f"Question: {ex['question'][:100]}...")
        print(f"Original schema ({ex['original_len']} chars):")
        print(ex['full_schema_snippet'][:300] + '...' if len(ex['full_schema_snippet']) > 300 else ex['full_schema_snippet'])
        print()
        print(f"Compact schema ({ex['compact_len']} chars):")
        print(ex['compact_schema_snippet'][:300] + '...' if len(ex['compact_schema_snippet']) > 300 else ex['compact_schema_snippet'])
        print()
        print(f"SQL: {ex['gold_sql']}")
    
    # Save outputs
    if not args.dry_run:
        print()
        print("Saving outputs...")
        
        save_jsonl(t11_train, output_dir / 'train_t11.jsonl')
        print(f"  Saved {output_dir / 'train_t11.jsonl'}")
        
        save_jsonl(t11_dev, output_dir / 'dev_t11.jsonl')
        print(f"  Saved {output_dir / 'dev_t11.jsonl'}")
        
        with open(output_dir / 'build_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved {output_dir / 'build_summary.json'}")
    else:
        print()
        print("Dry run - no files saved")
    
    print()
    print("Done!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

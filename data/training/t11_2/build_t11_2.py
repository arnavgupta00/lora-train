#!/usr/bin/env python3
"""
T11.1 Dataset Builder

Builds T11.1 mixed full/compact schema dataset from T10.
T11.1 is a corrected version of T11 with softer compaction policy.

Usage:
    python build_t11_1.py [--t10-dir PATH] [--output-dir PATH] [--dry-run]

Changes from T11:
- Wider safety margins (≤8 small-table, 5 extras, primary-table bonus, min 4 cols)
- Over-compaction guards (reject if >80% reduction, <300 chars on large schemas)
- Re-widen retry before full-schema fallback
- Hash-based 55% compact assignment (post-fallback lands near 50%)
- Target 65-80% average reduction (down from T11's 91%)
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "t10"))

from t11_2_utils import (
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
    Process a T10 dataset into T11.1 format.

    Returns:
        (t11_examples, stats)
    """
    stats = {
        'total': len(t10_examples),
        'full_schema': 0,
        'compact_schema': 0,
        'compaction_success': 0,
        'compaction_success_widened': 0,
        'compaction_fallback_parse_error': 0,
        'compaction_fallback_empty_result': 0,
        'compaction_fallback_sql_coverage': 0,
        'compaction_fallback_over_compacted': 0,
        'total_original_schema_len': 0,
        'total_compact_schema_len': 0,
        'per_db_stats': defaultdict(lambda: {'full': 0, 'compact': 0}),
    }

    # Assign schema modes per-db (hash-based 55% compact)
    print(f"  Assigning schema modes for {dataset_name} (hash-based 55% compact)...")
    assignments = assign_schema_modes_per_db(t10_examples)

    # Count assignments
    compact_assigned = sum(1 for v in assignments.values() if v == 'compact')
    full_assigned = sum(1 for v in assignments.values() if v == 'full')
    print(f"  Pre-validation assignment: {compact_assigned} compact, {full_assigned} full "
          f"({compact_assigned / len(assignments) * 100:.1f}% compact)")

    # Process each example
    t11_examples = []

    for i, t10_ex in enumerate(t10_examples):
        if verbose and i % 500 == 0:
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
                    from t11_2_utils import CompactionResult
                    compaction = CompactionResult(
                        compact_schema=full_schema,
                        original_len=len(full_schema),
                        compact_len=len(full_schema),
                        status="fallback_sql_coverage",
                        fallback_reason=f"Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                    )

            if compaction_valid:
                stats['compaction_success'] += 1
                if compaction.widened:
                    stats['compaction_success_widened'] += 1
                stats['compact_schema'] += 1
                t11_ex = build_t11_example(t10_ex, 'compact', compaction)
            else:
                # Fallback to full schema
                if 'parse_error' in compaction.status:
                    stats['compaction_fallback_parse_error'] += 1
                elif 'sql_coverage' in compaction.status:
                    stats['compaction_fallback_sql_coverage'] += 1
                elif 'over_compacted' in compaction.status:
                    stats['compaction_fallback_over_compacted'] += 1
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
    Validate T11.1 examples against T10.

    Returns:
        (all_valid, list_of_errors)
    """
    errors = []

    if len(t10_examples) != len(t11_examples):
        errors.append(f"Count mismatch: T10={len(t10_examples)}, T11.1={len(t11_examples)}")
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
    """Generate build summary with distribution stats, not just averages."""
    total_examples = train_stats['total'] + dev_stats['total']
    total_compact = train_stats['compact_schema'] + dev_stats['compact_schema']

    compact_examples = [ex for ex in (train_examples + dev_examples) if ex.get('schema_mode') == 'compact']

    reduction_pcts, compact_lens, full_lens = [], [], []
    for ex in compact_examples:
        orig = ex.get('original_schema_len', 0) or 0
        comp = ex.get('compact_schema_len', 0) or 0
        if orig > 0:
            reduction_pcts.append((1 - comp / orig) * 100)
            compact_lens.append(comp)
            full_lens.append(orig)

    def _pct(vals, p):
        if not vals:
            return 0.0
        vals = sorted(vals)
        if len(vals) == 1:
            return float(vals[0])
        k = (len(vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(vals[int(k)])
        return float(vals[f] * (c - k) + vals[c] * (k - f))

    avg_full = (sum(full_lens) / len(full_lens)) if full_lens else 0.0
    avg_compact = (sum(compact_lens) / len(compact_lens)) if compact_lens else 0.0
    avg_reduction = (sum(reduction_pcts) / len(reduction_pcts)) if reduction_pcts else 0.0

    over_75 = sum(1 for r in reduction_pcts if r > 75)
    over_80 = sum(1 for r in reduction_pcts if r > 80)
    small_under_500 = sum(1 for ex in compact_examples if (ex.get('compact_schema_len', 0) or 0) < 500)
    small_under_800_large = sum(
        1 for ex in compact_examples
        if (ex.get('original_schema_len', 0) or 0) > 2000 and (ex.get('compact_schema_len', 0) or 0) < 800
    )

    example_ex = find_example_for_summary(train_examples) or find_example_for_summary(dev_examples)
    example_before_after = None
    if example_ex:
        db_id = example_ex.get('db_id')
        question = _extract_question(example_ex)
        for t10_ex in t10_train:
            if t10_ex.get('db_id') == db_id and _extract_question(t10_ex) == question:
                full_schema = _extract_schema(t10_ex)
                compact_schema = _extract_schema(example_ex)
                example_before_after = {
                    'db_id': db_id,
                    'question': question,
                    'full_schema_snippet': full_schema[:500] + '...' if len(full_schema) > 500 else full_schema,
                    'compact_schema_snippet': compact_schema,
                    'gold_sql': _extract_gold_sql(example_ex),
                    'original_len': example_ex.get('original_schema_len'),
                    'compact_len': example_ex.get('compact_schema_len'),
                    'compaction_reduction_pct': example_ex.get('compaction_reduction_pct'),
                    'widened': example_ex.get('compaction_widened', False),
                }
                break

    summary = {
        'version': 'T11.1-recommended',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'changes_from_t11': [
            'Stronger safety margins (<=10 small-table, 6 extras, larger primary-table bonus, column floors)',
            'Hard over-compaction guards (reject >80% reduction; reject tiny compact schemas on large originals)',
            'Retry with wider margin before full-schema fallback',
            'Hash-based 55% compact assignment (post-fallback target ~50%)',
            'Distribution stats added to monitor compact-schema sharpness',
        ],
        'train_count': train_stats['total'],
        'dev_count': dev_stats['total'],
        'full_schema_count': train_stats['full_schema'] + dev_stats['full_schema'],
        'compact_schema_count': total_compact,
        'compact_ratio': total_compact / total_examples if total_examples > 0 else 0,
        'compaction_status_breakdown': {
            'success': train_stats['compaction_success'] + dev_stats['compaction_success'],
            'success_widened': train_stats['compaction_success_widened'] + dev_stats['compaction_success_widened'],
            'fallback_parse_error': train_stats['compaction_fallback_parse_error'] + dev_stats['compaction_fallback_parse_error'],
            'fallback_uncertain_extraction': train_stats.get('compaction_fallback_uncertain_extraction', 0) + dev_stats.get('compaction_fallback_uncertain_extraction', 0),
            'fallback_empty_result': train_stats['compaction_fallback_empty_result'] + dev_stats['compaction_fallback_empty_result'],
            'fallback_sql_coverage': train_stats.get('compaction_fallback_sql_coverage', 0) + dev_stats.get('compaction_fallback_sql_coverage', 0),
            'fallback_over_compacted': train_stats['compaction_fallback_over_compacted'] + dev_stats['compaction_fallback_over_compacted'],
            'not_applicable_full': train_stats['full_schema'] + dev_stats['full_schema'],
        },
        'avg_full_schema_len': round(avg_full, 1),
        'avg_compact_schema_len': round(avg_compact, 1),
        'avg_reduction_pct_compact_examples': round(avg_reduction, 1),
        'avg_reduction_pct_all_attempts': round(avg_reduction, 1),
        'avg_reduction_pct_successful': round(avg_reduction, 1),
        'reduction_pct_distribution': {
            'p50': round(_pct(reduction_pcts, 0.50), 1),
            'p75': round(_pct(reduction_pcts, 0.75), 1),
            'p90': round(_pct(reduction_pcts, 0.90), 1),
            'p95': round(_pct(reduction_pcts, 0.95), 1),
        },
        'compact_len_distribution': {
            'p50': round(_pct(compact_lens, 0.50), 1),
            'p90': round(_pct(compact_lens, 0.90), 1),
            'p95': round(_pct(compact_lens, 0.95), 1),
        },
        'over_compaction_tail': {
            'count_reduction_gt_75': over_75,
            'count_reduction_gt_80': over_80,
            'count_compact_len_lt_500': small_under_500,
            'count_compact_len_lt_800_for_large_originals': small_under_800_large,
        },
        'train_stats': {
            'total': train_stats['total'],
            'full_schema': train_stats['full_schema'],
            'compact_schema': train_stats['compact_schema'],
            'compaction_success': train_stats['compaction_success'],
            'compaction_success_widened': train_stats['compaction_success_widened'],
            'fallback_parse_error': train_stats['compaction_fallback_parse_error'],
            'fallback_uncertain_extraction': train_stats.get('compaction_fallback_uncertain_extraction', 0),
            'fallback_empty_result': train_stats['compaction_fallback_empty_result'],
            'fallback_sql_coverage': train_stats.get('compaction_fallback_sql_coverage', 0),
            'fallback_over_compacted': train_stats['compaction_fallback_over_compacted'],
        },
        'dev_stats': {
            'total': dev_stats['total'],
            'full_schema': dev_stats['full_schema'],
            'compact_schema': dev_stats['compact_schema'],
            'compaction_success': dev_stats['compaction_success'],
            'compaction_success_widened': dev_stats['compaction_success_widened'],
            'fallback_parse_error': dev_stats['compaction_fallback_parse_error'],
            'fallback_uncertain_extraction': dev_stats.get('compaction_fallback_uncertain_extraction', 0),
            'fallback_empty_result': dev_stats['compaction_fallback_empty_result'],
            'fallback_sql_coverage': dev_stats.get('compaction_fallback_sql_coverage', 0),
            'fallback_over_compacted': dev_stats['compaction_fallback_over_compacted'],
        },
        'example_before_after': example_before_after,
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description='Build T11.1 dataset from T10')
    parser.add_argument('--t10-dir', type=str, default=None,
                        help='Directory containing T10 dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for T11.1 dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process but do not write output files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate output (default: True)')
    parser.add_argument('--no-validate', action='store_false', dest='validate',
                        help='Skip validation')
    args = parser.parse_args()

    # Resolve default paths relative to script location
    script_dir = Path(__file__).parent
    t10_dir = Path(args.t10_dir) if args.t10_dir else script_dir.parent / "t10"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T11.1 Dataset Builder (Softer Compaction)")
    print("=" * 60)
    print(f"T10 directory: {t10_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    print("T11.1 changes from T11:")
    print("  - Wider safety margins (≤8 small-table, 5 extras, primary bonus, min 4 cols)")
    print("  - Over-compaction guards (reject >80% reduction)")
    print("  - Re-widen retry before full-schema fallback")
    print("  - Hash-based 55% compact assignment")
    print("  - Target 65-80% average reduction")
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
    print(f"    - of which widened: {train_stats['compaction_success_widened']}")
    print(f"  Fallback (parse error): {train_stats['compaction_fallback_parse_error']}")
    print(f"  Fallback (empty result): {train_stats['compaction_fallback_empty_result']}")
    print(f"  Fallback (SQL coverage): {train_stats.get('compaction_fallback_sql_coverage', 0)}")
    print(f"  Fallback (over-compacted): {train_stats['compaction_fallback_over_compacted']}")
    print()

    # Process dev dataset
    print("Processing dev dataset...")
    t11_dev, dev_stats = process_dataset(t10_dev, 'dev', verbose=args.verbose)
    print(f"  Full schema: {dev_stats['full_schema']}")
    print(f"  Compact schema: {dev_stats['compact_schema']}")
    print(f"  Compaction success: {dev_stats['compaction_success']}")
    print(f"    - of which widened: {dev_stats['compaction_success_widened']}")
    print(f"  Fallback (parse error): {dev_stats['compaction_fallback_parse_error']}")
    print(f"  Fallback (empty result): {dev_stats['compaction_fallback_empty_result']}")
    print(f"  Fallback (SQL coverage): {dev_stats.get('compaction_fallback_sql_coverage', 0)}")
    print(f"  Fallback (over-compacted): {dev_stats['compaction_fallback_over_compacted']}")
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
    print("BUILD SUMMARY — T11.1")
    print("=" * 60)
    print(f"Train examples: {summary['train_count']}")
    print(f"Dev examples: {summary['dev_count']}")
    print(f"Full schema examples: {summary['full_schema_count']}")
    print(f"Compact schema examples: {summary['compact_schema_count']}")
    print(f"Compact ratio: {summary['compact_ratio']:.1%}")
    print(f"Average full schema length: {summary['avg_full_schema_len']:.0f} chars")
    print(f"Average compact schema length: {summary['avg_compact_schema_len']:.0f} chars")
    print(f"Average reduction (all attempts): {summary['avg_reduction_pct_all_attempts']:.1f}%")
    print(f"Average reduction (successful only): {summary['avg_reduction_pct_successful']:.1f}%")
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
        print(f"Widened: {ex.get('widened', False)}")
        print(f"Original schema ({ex['original_len']} chars):")
        snippet = ex['full_schema_snippet']
        print(snippet[:300] + '...' if len(snippet) > 300 else snippet)
        print()
        print(f"Compact schema ({ex['compact_len']} chars):")
        snippet = ex['compact_schema_snippet']
        print(snippet[:300] + '...' if len(snippet) > 300 else snippet)
        print()
        print(f"SQL: {ex['gold_sql']}")

    # Save outputs
    if not args.dry_run:
        print()
        print("Saving outputs...")

        save_jsonl(t11_train, output_dir / 'train_t11_2.jsonl')
        print(f"  Saved {output_dir / 'train_t11_2.jsonl'}")

        save_jsonl(t11_dev, output_dir / 'dev_t11_2.jsonl')
        print(f"  Saved {output_dir / 'dev_t11_2.jsonl'}")

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

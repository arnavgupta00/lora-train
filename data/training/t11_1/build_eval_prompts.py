#!/usr/bin/env python3
"""
Build T11.1 eval prompts from existing full-schema prompts.

Input:
    data/training/t10/bird_dev_t10.jsonl

Outputs:
    data/training/t11_1/bird_dev_t11_1.jsonl
    data/training/t11_1/bird_dev_t11_1.stats.json

Usage:
    python data/training/t11_1/build_eval_prompts.py \
        --full_prompts_file data/training/t10/bird_dev_t10.jsonl \
        --output data/training/t11_1/bird_dev_t11_1.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data" / "training" / "t10"))

from compact_schema import (
    compact_schema,
    extract_schema_from_t10_prompt,
    extract_hints_from_t10_prompt,
    extract_question_from_t10_prompt,
    replace_schema_in_t10_prompt,
    CompactionMetadata,
)

# Import T10 utilities
try:
    from t10_utils import (
        T10_SYSTEM_PROMPT,
        validate_t10_prompt,
        get_t10_system_prompt_hash,
    )
except ImportError:
    # Fallback if t10_utils not available
    T10_SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Copy table names and column names exactly from the schema.
Never invent normalized identifiers.
If an identifier contains spaces, punctuation, %, hyphens, slashes, or parentheses, use it exactly and wrap it in backticks.
Use only the tables and columns that exist in the schema.
Only output the SQL query, nothing else."""
    
    def validate_t10_prompt(system, user, strict=False):
        return True, []
    
    def get_t10_system_prompt_hash():
        import hashlib
        return hashlib.sha256(T10_SYSTEM_PROMPT.encode()).hexdigest()[:16]


def load_full_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    """Load full-schema prompts from JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def build_compact_prompt(
    full_prompt: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a compact-schema prompt from a full-schema prompt.
    
    Returns the prompt record with compaction_metadata included.
    """
    # Extract components from the full prompt
    t10_prompt = full_prompt.get('t10_prompt', {})
    user_content = t10_prompt.get('user', '')
    
    full_schema = extract_schema_from_t10_prompt(user_content)
    hints = extract_hints_from_t10_prompt(user_content)
    question = extract_question_from_t10_prompt(user_content)
    
    if not full_schema or not question:
        # Can't compact, return as-is with fallback metadata
        return {
            **full_prompt,
            'compaction_metadata': {
                'original_schema_length': len(full_schema),
                'compact_schema_length': len(full_schema),
                'reduction_percent': 0.0,
                'primary_table': None,
                'tables_kept': [],
                'tables_dropped': [],
                'compaction_status': 'fallback',
                'fallback_reason': 'Could not extract schema or question from prompt',
                'pass_number': 1,
            }
        }
    
    # Apply compaction
    result = compact_schema(full_schema, question, hints)
    
    # Build new prompt with compact schema
    if result.metadata.compaction_status in ('success', 'widened'):
        new_user_content = replace_schema_in_t10_prompt(user_content, result.compact_schema)
    else:
        # Fallback - use original schema
        new_user_content = user_content
    
    # Build output record
    compact_prompt = {
        'question_id': full_prompt.get('question_id'),
        'db_id': full_prompt.get('db_id'),
        'question': full_prompt.get('question'),
        'gold_sql': full_prompt.get('gold_sql'),
        'difficulty': full_prompt.get('difficulty'),
        'evidence': full_prompt.get('evidence'),
        't10_prompt': {
            'system': t10_prompt.get('system', T10_SYSTEM_PROMPT),
            'user': new_user_content,
        },
        'compaction_metadata': asdict(result.metadata),
    }
    
    return compact_prompt


def generate_summary(
    compact_prompts: List[Dict[str, Any]],
    full_prompts_file: str,
) -> Dict[str, Any]:
    """Generate aggregate compaction statistics."""
    total = len(compact_prompts)
    
    # Status counts
    status_counts = Counter()
    fallback_reasons = Counter()
    
    # Reduction statistics
    reductions = []
    successful_reductions = []
    
    # Per-difficulty breakdown
    difficulty_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'total': 0, 'success': 0, 'widened': 0, 'fallback': 0,
        'avg_reduction': 0.0, 'reductions': []
    })
    
    # Per-database breakdown
    db_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'total': 0, 'success': 0, 'widened': 0, 'fallback': 0,
        'avg_reduction': 0.0, 'reductions': []
    })
    
    for prompt in compact_prompts:
        meta = prompt.get('compaction_metadata', {})
        status = meta.get('compaction_status', 'unknown')
        difficulty = prompt.get('difficulty', 'unknown')
        db_id = prompt.get('db_id', 'unknown')
        reduction = meta.get('reduction_percent', 0.0)
        
        status_counts[status] += 1
        reductions.append(reduction)
        
        if status in ('success', 'widened'):
            successful_reductions.append(reduction)
        
        if status == 'fallback' and meta.get('fallback_reason'):
            # Simplify fallback reason for counting
            reason = meta['fallback_reason']
            if 'parse error' in reason.lower():
                fallback_reasons['parse_error'] += 1
            elif 'no tables' in reason.lower() or 'no matching' in reason.lower():
                fallback_reasons['no_tables_matched'] += 1
            elif 'over-compacted' in reason.lower():
                fallback_reasons['over_compacted'] += 1
            elif 'not smaller' in reason.lower():
                fallback_reasons['compact_not_smaller'] += 1
            else:
                fallback_reasons['other'] += 1
        
        # Per-difficulty
        difficulty_stats[difficulty]['total'] += 1
        difficulty_stats[difficulty][status] = difficulty_stats[difficulty].get(status, 0) + 1
        difficulty_stats[difficulty]['reductions'].append(reduction)
        
        # Per-database
        db_stats[db_id]['total'] += 1
        db_stats[db_id][status] = db_stats[db_id].get(status, 0) + 1
        db_stats[db_id]['reductions'].append(reduction)
    
    # Calculate averages
    avg_reduction = sum(reductions) / len(reductions) if reductions else 0.0
    avg_successful_reduction = (
        sum(successful_reductions) / len(successful_reductions) 
        if successful_reductions else 0.0
    )
    
    # Calculate per-difficulty averages
    for diff, stats in difficulty_stats.items():
        if stats['reductions']:
            stats['avg_reduction'] = round(sum(stats['reductions']) / len(stats['reductions']), 1)
        del stats['reductions']
    
    # Calculate per-database averages
    for db_id, stats in db_stats.items():
        if stats['reductions']:
            stats['avg_reduction'] = round(sum(stats['reductions']) / len(stats['reductions']), 1)
        del stats['reductions']
    
    # Reduction distribution (buckets)
    reduction_buckets = {
        '0-20%': 0,
        '20-40%': 0,
        '40-60%': 0,
        '60-80%': 0,
        '80-100%': 0,
    }
    for r in reductions:
        if r < 20:
            reduction_buckets['0-20%'] += 1
        elif r < 40:
            reduction_buckets['20-40%'] += 1
        elif r < 60:
            reduction_buckets['40-60%'] += 1
        elif r < 80:
            reduction_buckets['60-80%'] += 1
        else:
            reduction_buckets['80-100%'] += 1
    
    # Sample examples
    sample_success = []
    sample_fallback = []
    for prompt in compact_prompts:
        meta = prompt.get('compaction_metadata', {})
        status = meta.get('compaction_status', 'unknown')
        
        if status in ('success', 'widened') and len(sample_success) < 3:
            sample_success.append({
                'question_id': prompt.get('question_id'),
                'db_id': prompt.get('db_id'),
                'reduction_percent': meta.get('reduction_percent'),
                'primary_table': meta.get('primary_table'),
                'tables_kept': meta.get('tables_kept'),
                'status': status,
            })
        elif status == 'fallback' and len(sample_fallback) < 3:
            sample_fallback.append({
                'question_id': prompt.get('question_id'),
                'db_id': prompt.get('db_id'),
                'fallback_reason': meta.get('fallback_reason'),
            })
    
    summary = {
        'source_file': full_prompts_file,
        't10_system_prompt_hash': get_t10_system_prompt_hash(),
        'total_examples': total,
        'status_counts': dict(status_counts),
        'success_rate': round(100 * (status_counts['success'] + status_counts['widened']) / total, 1) if total > 0 else 0.0,
        'widened_rate': round(100 * status_counts['widened'] / total, 1) if total > 0 else 0.0,
        'fallback_rate': round(100 * status_counts['fallback'] / total, 1) if total > 0 else 0.0,
        'avg_reduction_all': round(avg_reduction, 1),
        'avg_reduction_successful': round(avg_successful_reduction, 1),
        'reduction_distribution': reduction_buckets,
        'fallback_reasons': dict(fallback_reasons),
        'per_difficulty': dict(difficulty_stats),
        'per_database': dict(db_stats),
        'sample_success': sample_success,
        'sample_fallback': sample_fallback,
    }
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("COMPACT SCHEMA PROMPT BUILD SUMMARY")
    print("=" * 60)
    print(f"Source: {summary['source_file']}")
    print(f"Total examples: {summary['total_examples']}")
    print()
    
    print("--- Status Breakdown ---")
    for status, count in summary['status_counts'].items():
        pct = 100 * count / summary['total_examples']
        print(f"  {status:15s}: {count:5d} ({pct:5.1f}%)")
    print()
    
    print(f"Success rate: {summary['success_rate']}%")
    print(f"Widened rate: {summary['widened_rate']}%")
    print(f"Fallback rate: {summary['fallback_rate']}%")
    print()
    
    print("--- Reduction Statistics ---")
    print(f"Average reduction (all):        {summary['avg_reduction_all']}%")
    print(f"Average reduction (successful): {summary['avg_reduction_successful']}%")
    print()
    
    print("--- Reduction Distribution ---")
    for bucket, count in summary['reduction_distribution'].items():
        bar = "█" * int(count / summary['total_examples'] * 40)
        print(f"  {bucket:10s}: {count:5d} {bar}")
    print()
    
    if summary['fallback_reasons']:
        print("--- Fallback Reasons ---")
        for reason, count in summary['fallback_reasons'].items():
            print(f"  {reason:25s}: {count}")
        print()
    
    print("--- Per-Difficulty ---")
    for diff in ['simple', 'moderate', 'challenging']:
        if diff in summary['per_difficulty']:
            stats = summary['per_difficulty'][diff]
            success_pct = 100 * (stats.get('success', 0) + stats.get('widened', 0)) / stats['total'] if stats['total'] > 0 else 0
            print(f"  {diff:15s}: {stats['total']:4d} examples, {success_pct:5.1f}% success, avg reduction {stats['avg_reduction']}%")
    print()
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build T11.1 eval prompts from existing full-schema prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full_prompts_file",
        default="data/training/t10/bird_dev_t10.jsonl",
        help="Path to full-schema prompts (default: data/training/t10/bird_dev_t10.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/training/t11_1/bird_dev_t11_1.jsonl",
        help="Output JSONL path for T11.1 prompts",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building T11.1 Eval Prompts")
    print("=" * 60)
    print(f"Full prompts: {args.full_prompts_file}")
    print(f"Output file: {args.output}")
    print(f"T10 system prompt hash: {get_t10_system_prompt_hash()}")
    print()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load full prompts
    print("Loading full-schema prompts...")
    full_prompts = load_full_prompts(args.full_prompts_file)
    print(f"  Loaded {len(full_prompts)} prompts")
    print()
    
    # Build compact prompts
    print("Building compact prompts...")
    compact_prompts = []
    
    for i, full_prompt in enumerate(full_prompts):
        compact_prompt = build_compact_prompt(full_prompt)
        compact_prompts.append(compact_prompt)
        
        # Validate T10 compliance
        t10_prompt = compact_prompt.get('t10_prompt', {})
        is_valid, errors = validate_t10_prompt(
            t10_prompt.get('system', ''),
            t10_prompt.get('user', ''),
            strict=False
        )
        if not is_valid:
            print(f"  ⚠ Q{compact_prompt.get('question_id')}: T10 validation failed: {errors}")
        
        # Progress
        if (i + 1) % 200 == 0 or (i + 1) == len(full_prompts):
            success_count = sum(
                1 for p in compact_prompts 
                if p.get('compaction_metadata', {}).get('compaction_status') in ('success', 'widened')
            )
            print(f"  [{i+1}/{len(full_prompts)}] Built {success_count} compact, {i+1-success_count} fallback")
    
    # Generate summary
    summary = generate_summary(compact_prompts, args.full_prompts_file)
    
    # Write outputs
    compact_path = output_path
    print(f"\nWriting compact prompts to {compact_path}...")
    with open(compact_path, 'w') as f:
        for prompt in compact_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    summary_path = output_path.with_suffix(".stats.json")
    print(f"Writing summary to {summary_path}...")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print_summary(summary)
    
    print(f"\nOutputs:")
    print(f"  - {compact_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()

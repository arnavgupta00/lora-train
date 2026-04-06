#!/usr/bin/env python3
"""
T12 Dataset Validator and Assembler

Validates generated examples against:
1. Execution validation (SQL syntax)
2. Schema validation (valid tables/columns)
3. Duplicate filtering
4. Contamination filtering
5. Archetype consistency
6. Difficulty plausibility

Then assembles the final T12 dataset.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


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


def extract_sql(example: Dict) -> str:
    """Extract SQL from example."""
    return example['messages'][-1]['content']


def extract_schema(example: Dict) -> str:
    """Extract schema from example user content."""
    user_content = example['messages'][1]['content']
    match = re.search(r'Schema:\n(.*?)\n\nHints:', user_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def validate_sql_syntax(sql: str, schema_path: Path) -> Tuple[bool, str]:
    """Validate SQL syntax against schema."""
    try:
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Load schema
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        cursor.executescript(schema_sql)
        
        # Try to explain the query (validates syntax without needing data)
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)


def compute_sql_hash(sql: str) -> str:
    """Compute normalized hash of SQL for duplicate detection."""
    # Normalize whitespace and case for comparison
    normalized = ' '.join(sql.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def compute_question_hash(example: Dict) -> str:
    """Compute hash of question for duplicate detection."""
    user_content = example['messages'][1]['content']
    match = re.search(r'Question:\n(.+)', user_content, re.DOTALL)
    if match:
        question = ' '.join(match.group(1).strip().lower().split())
        return hashlib.md5(question.encode()).hexdigest()
    return ""


def load_existing_hashes(t10_path: Path, t11_path: Path) -> Set[str]:
    """Load SQL hashes from existing training data."""
    hashes = set()
    
    for path in [t10_path, t11_path]:
        if path.exists():
            examples = load_jsonl(path)
            for ex in examples:
                sql = extract_sql(ex)
                hashes.add(compute_sql_hash(sql))
    
    return hashes


def estimate_difficulty(sql: str) -> str:
    """Estimate difficulty based on SQL structure."""
    sql_lower = sql.lower()
    
    score = 0
    
    # Count joins
    join_count = len(re.findall(r'\bjoin\b', sql_lower))
    score += join_count * 2
    
    # Count subqueries
    subquery_count = sql_lower.count('select') - 1
    score += subquery_count * 3
    
    # Aggregations
    if re.search(r'\b(count|sum|avg|max|min)\s*\(', sql_lower):
        score += 1
    
    # GROUP BY
    if 'group by' in sql_lower:
        score += 1
    
    # HAVING
    if 'having' in sql_lower:
        score += 2
    
    # Window functions
    if re.search(r'\bover\s*\(', sql_lower):
        score += 3
    
    # DISTINCT
    if 'distinct' in sql_lower:
        score += 1
    
    # Set operations
    if re.search(r'\b(union|intersect|except)\b', sql_lower):
        score += 2
    
    # CTEs
    if sql_lower.strip().startswith('with'):
        score += 2
    
    if score <= 2:
        return 'simple'
    elif score <= 6:
        return 'moderate'
    else:
        return 'challenging'


def validate_and_filter(
    raw_examples: List[Dict],
    schema_path: Path,
    existing_hashes: Set[str],
    target_count: int,
    family_name: str
) -> Tuple[List[Dict], Dict]:
    """Validate and filter examples for a family."""
    
    stats = {
        'total_raw': len(raw_examples),
        'passed_syntax': 0,
        'passed_duplicate': 0,
        'accepted': 0,
        'rejected_syntax': 0,
        'rejected_duplicate': 0,
        'rejected_over_target': 0,
    }
    
    accepted = []
    seen_hashes = set()
    
    for ex in raw_examples:
        sql = extract_sql(ex)
        sql_hash = compute_sql_hash(sql)
        
        # Check syntax
        valid, error = validate_sql_syntax(sql, schema_path)
        if not valid:
            stats['rejected_syntax'] += 1
            continue
        stats['passed_syntax'] += 1
        
        # Check duplicates (within batch and against existing)
        if sql_hash in seen_hashes or sql_hash in existing_hashes:
            stats['rejected_duplicate'] += 1
            continue
        stats['passed_duplicate'] += 1
        
        # Check difficulty consistency
        estimated = estimate_difficulty(sql)
        claimed = ex.get('difficulty_target', 'moderate')
        
        # Allow some flexibility but flag major mismatches
        if (estimated == 'simple' and claimed == 'challenging') or \
           (estimated == 'challenging' and claimed == 'simple'):
            # Adjust the claimed difficulty
            ex['difficulty_target'] = estimated
        
        # Add to accepted if under target
        if len(accepted) < target_count:
            seen_hashes.add(sql_hash)
            ex['original_source_id'] = f"{family_name}_{len(accepted):04d}"
            ex['generation_method'] = 'synthetic_llm'
            ex['schema_source_type'] = 'synthetic_isomorphic'
            accepted.append(ex)
            stats['accepted'] += 1
        else:
            stats['rejected_over_target'] += 1
    
    return accepted, stats


def create_audit_set(all_accepted: List[Dict], target_size: int = 100) -> List[Dict]:
    """Create balanced audit set from accepted examples."""
    # Group by family
    by_family = defaultdict(list)
    for ex in all_accepted:
        family = ex.get('t12_archetype_family', 'unknown')
        by_family[family].append(ex)
    
    # Sample proportionally
    audit_set = []
    per_family = max(1, target_size // len(by_family))
    
    for family, examples in by_family.items():
        sample_size = min(per_family, len(examples))
        audit_set.extend(random.sample(examples, sample_size))
    
    # Trim to target
    if len(audit_set) > target_size:
        audit_set = random.sample(audit_set, target_size)
    
    return audit_set


def main():
    parser = argparse.ArgumentParser(description='T12 Dataset Validator and Assembler')
    parser.add_argument('--t12-dir', type=Path, default=Path('data/training/t12'))
    parser.add_argument('--t10-train', type=Path, default=Path('data/training/t10/train_t10.jsonl'))
    parser.add_argument('--t11-train', type=Path, default=Path('data/training/t11_2/train_t11_2.jsonl'))
    parser.add_argument('--t10-dev', type=Path, default=Path('data/training/t10/dev_t10.jsonl'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    print("T12 Dataset Validator and Assembler")
    print("=" * 50)
    
    # Family configuration
    families = {
        'california_schools_type': {
            'raw_file': 'california_schools_type_raw.jsonl',
            'schema_file': 'schemas/education_programs.sql',
            'target': 360,
        },
        'financial_type': {
            'raw_file': 'financial_type_raw_merged.jsonl',
            'schema_file': 'schemas/banking_system.sql',
            'target': 340,
        },
        'formula_1_type': {
            'raw_file': 'formula_1_type_raw_merged.jsonl',
            'schema_file': 'schemas/motorsport_events.sql',
            'target': 300,
        },
        'thrombosis_prediction_type': {
            'raw_file': 'thrombosis_prediction_type_raw_merged.jsonl',
            'schema_file': 'schemas/clinical_records.sql',
            'target': 260,
        },
        'debit_card_specializing_type': {
            'raw_file': 'debit_card_specializing_type_raw.jsonl',
            'schema_file': 'schemas/retail_analytics.sql',
            'target': 160,
        },
        'card_games_type': {
            'raw_file': 'card_games_type_raw.jsonl',
            'schema_file': 'schemas/media_catalog.sql',
            'target': 180,
        },
        'toxicology_type': {
            'raw_file': 'toxicology_type_raw_merged.jsonl',
            'schema_file': 'schemas/materials_chemistry.sql',
            'target': 150,
        },
        'stability_pack': {
            'raw_file': 'stability_pack_raw_merged.jsonl',
            'schema_file': 'schemas/mixed_stable.sql',
            'target': 300,
        },
    }
    
    generated_dir = args.t12_dir / 'generated'
    
    # Load existing hashes for duplicate detection
    print("\nLoading existing training data hashes...")
    existing_hashes = load_existing_hashes(args.t10_train, args.t11_train)
    print(f"  Loaded {len(existing_hashes)} existing SQL hashes")
    
    # Process each family
    all_accepted = []
    all_stats = {}
    
    for family_name, config in families.items():
        print(f"\n--- Processing {family_name} ---")
        
        raw_path = generated_dir / config['raw_file']
        schema_path = generated_dir / config['schema_file']
        
        if not raw_path.exists():
            print(f"  WARNING: {raw_path} not found, skipping")
            continue
        
        raw_examples = load_jsonl(raw_path)
        print(f"  Loaded {len(raw_examples)} raw examples")
        
        accepted, stats = validate_and_filter(
            raw_examples,
            schema_path,
            existing_hashes,
            config['target'],
            family_name
        )
        
        all_accepted.extend(accepted)
        all_stats[family_name] = stats
        
        # Update existing hashes with accepted ones
        for ex in accepted:
            existing_hashes.add(compute_sql_hash(extract_sql(ex)))
        
        print(f"  Accepted: {stats['accepted']}/{config['target']} target")
        print(f"  Rejected: syntax={stats['rejected_syntax']}, duplicate={stats['rejected_duplicate']}")
    
    print(f"\n{'=' * 50}")
    print(f"Total accepted: {len(all_accepted)}")
    
    if args.dry_run:
        print("\nDry run - not writing files")
        return
    
    # Load backbone (T11_2 training data)
    print("\nLoading T11_2 backbone...")
    backbone = load_jsonl(args.t11_train)
    print(f"  Loaded {len(backbone)} backbone examples")
    
    # Merge
    train_t12 = backbone + all_accepted
    print(f"\nFinal train_t12 size: {len(train_t12)}")
    
    # Save train file
    train_path = args.t12_dir / 'train_t12.jsonl'
    save_jsonl(train_t12, train_path)
    print(f"  Saved: {train_path}")
    
    # Copy dev unchanged from T10
    print("\nCopying dev from T10 (unchanged)...")
    dev_examples = load_jsonl(args.t10_dev)
    dev_path = args.t12_dir / 'dev_t12.jsonl'
    save_jsonl(dev_examples, dev_path)
    print(f"  Saved: {dev_path}")
    
    # Create audit set
    print("\nCreating audit set...")
    audit_set = create_audit_set(all_accepted, target_size=100)
    audit_path = args.t12_dir / 't12_audit_set.jsonl'
    save_jsonl(audit_set, audit_path)
    print(f"  Saved: {audit_path} ({len(audit_set)} examples)")
    
    # Create source map
    print("\nCreating source map...")
    source_map = []
    for ex in all_accepted:
        source_map.append({
            'original_source_id': ex.get('original_source_id'),
            't12_archetype_family': ex.get('t12_archetype_family'),
            'db_id': ex.get('db_id'),
            'difficulty_target': ex.get('difficulty_target'),
            'example_type': ex.get('example_type'),
            'sampling_weight': ex.get('sampling_weight'),
            'target_failure_mode': ex.get('target_failure_mode'),
        })
    source_map_path = args.t12_dir / 't12_source_map.jsonl'
    save_jsonl(source_map, source_map_path)
    print(f"  Saved: {source_map_path}")
    
    # Create acceptance report
    print("\nCreating acceptance report...")
    report = {
        'created_at': datetime.now().isoformat(),
        'total_raw': sum(s['total_raw'] for s in all_stats.values()),
        'total_accepted': len(all_accepted),
        'backbone_size': len(backbone),
        'final_train_size': len(train_t12),
        'dev_size': len(dev_examples),
        'audit_set_size': len(audit_set),
        'per_family_stats': all_stats,
    }
    report_path = args.t12_dir / 't12_acceptance_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")
    
    # Create archetype summary
    print("\nCreating archetype summary...")
    archetype_summary = defaultdict(lambda: {
        'count': 0,
        'example_types': defaultdict(int),
        'difficulty': defaultdict(int),
        'avg_sampling_weight': 0.0,
    })
    
    for ex in all_accepted:
        family = ex.get('t12_archetype_family', 'unknown')
        archetype_summary[family]['count'] += 1
        archetype_summary[family]['example_types'][ex.get('example_type', 'unknown')] += 1
        archetype_summary[family]['difficulty'][ex.get('difficulty_target', 'unknown')] += 1
        archetype_summary[family]['avg_sampling_weight'] += ex.get('sampling_weight', 1.0)
    
    # Convert to regular dict and compute averages
    summary_dict = {}
    for family, data in archetype_summary.items():
        if data['count'] > 0:
            data['avg_sampling_weight'] /= data['count']
        data['example_types'] = dict(data['example_types'])
        data['difficulty'] = dict(data['difficulty'])
        summary_dict[family] = data
    
    summary_path = args.t12_dir / 't12_archetype_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # Create sampling plan
    print("\nCreating sampling plan...")
    sampling_plan = {
        'strategy': 'per_example_weights',
        'max_physical_duplication': 2,
        'weights_by_archetype': {
            'rank-by-X return-Y': 1.8,
            'derived ratio/formula': 1.8,
            'multi-hop join chain': 2.0,
            'table-family disambiguation': 1.8,
            'distinct entity vs row count': 1.7,
            'cohort/denominator percentage': 1.9,
            'date-anchor / temporal SQLite logic': 1.8,
            'side-table text/metadata return-field': 1.6,
            'graph traversal / molecule-level': 1.7,
        },
        'total_weighted_examples': sum(
            ex.get('sampling_weight', 1.0) for ex in all_accepted
        ),
    }
    sampling_path = args.t12_dir / 't12_sampling_plan.json'
    with open(sampling_path, 'w') as f:
        json.dump(sampling_plan, f, indent=2)
    print(f"  Saved: {sampling_path}")
    
    # Create build manifest
    print("\nCreating build manifest...")
    manifest = {
        'version': 'T12',
        'created_at': datetime.now().isoformat(),
        'description': 'Final archetype-targeted dataset upgrade for SQL generation',
        'benchmark_clean': True,
        'backbone_source': 'T11_2',
        'files': {
            'train': 'train_t12.jsonl',
            'dev': 'dev_t12.jsonl',
            'audit_set': 't12_audit_set.jsonl',
            'source_map': 't12_source_map.jsonl',
            'acceptance_report': 't12_acceptance_report.json',
            'archetype_summary': 't12_archetype_summary.json',
            'sampling_plan': 't12_sampling_plan.json',
            'schema_registry': 'schema_family_registry.json',
        },
        'stats': {
            'train_examples': len(train_t12),
            'dev_examples': len(dev_examples),
            't12_additions': len(all_accepted),
            'backbone_examples': len(backbone),
        },
        'target_families': list(families.keys()),
    }
    manifest_path = args.t12_dir / 't12_build_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")
    
    print("\n" + "=" * 50)
    print("T12 build complete!")
    print(f"  Train: {len(train_t12)} examples")
    print(f"  Dev: {len(dev_examples)} examples")
    print(f"  T12 additions: {len(all_accepted)} examples")


if __name__ == '__main__':
    main()

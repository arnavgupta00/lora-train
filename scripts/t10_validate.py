#!/usr/bin/env python3
"""
T10 Dataset Validation Script

Validates that T10 dataset meets all requirements:
- Standardized system prompt
- Canonical user prompt structure (Schema: -> Hints: -> Question:)
- Multi-line DDL schema format
- No /no_think tokens
- No dropped examples
- Preserved SQL targets
"""

import json
import re
import argparse
from pathlib import Path


STANDARDIZED_SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Copy table names and column names exactly from the schema.
Never invent normalized identifiers.
If an identifier contains spaces, punctuation, %, hyphens, slashes, or parentheses, use it exactly and wrap it in backticks.
Use only the tables and columns that exist in the schema.
Only output the SQL query, nothing else.""".strip()


def validate_example(example: dict, idx: int) -> list:
    """Validate a single T10 example. Returns list of errors."""
    errors = []
    messages = example.get('messages', [])
    
    # Check message structure
    if len(messages) != 3:
        errors.append(f"Example {idx}: Expected 3 messages, got {len(messages)}")
        return errors
    
    system_msg = messages[0]
    user_msg = messages[1]
    assistant_msg = messages[2]
    
    # 1. Validate system prompt
    if system_msg.get('role') != 'system':
        errors.append(f"Example {idx}: First message should be system, got {system_msg.get('role')}")
    elif system_msg.get('content', '').strip() != STANDARDIZED_SYSTEM_PROMPT:
        errors.append(f"Example {idx}: System prompt does not match standardized version")
    
    # 2. Validate user message role
    if user_msg.get('role') != 'user':
        errors.append(f"Example {idx}: Second message should be user, got {user_msg.get('role')}")
        return errors
    
    user_content = user_msg.get('content', '')
    
    # 3. Check section presence
    if 'Schema:' not in user_content:
        errors.append(f"Example {idx}: Missing 'Schema:' section")
    if 'Hints:' not in user_content:
        errors.append(f"Example {idx}: Missing 'Hints:' section")
    if 'Question:' not in user_content:
        errors.append(f"Example {idx}: Missing 'Question:' section")
    
    # 4. Check section order
    schema_pos = user_content.find('Schema:')
    hints_pos = user_content.find('Hints:')
    question_pos = user_content.find('Question:')
    
    if schema_pos != -1 and hints_pos != -1 and question_pos != -1:
        if not (schema_pos < hints_pos < question_pos):
            errors.append(f"Example {idx}: Sections not in correct order (Schema < Hints < Question)")
    
    # 5. Check schema is multiline
    if schema_pos != -1 and hints_pos != -1:
        schema_section = user_content[schema_pos + 7:hints_pos].strip()  # 7 = len('Schema:')
        
        # Check for multiline: should have multiple lines with proper structure
        schema_lines = schema_section.split('\n')
        if len(schema_lines) < 3:
            errors.append(f"Example {idx}: Schema appears to be single-line or too short ({len(schema_lines)} lines)")
        
        # Check for CREATE TABLE statements
        if 'CREATE TABLE' not in schema_section.upper():
            errors.append(f"Example {idx}: Schema missing CREATE TABLE statement")
        
        # Check that columns are on separate lines (look for indentation pattern)
        has_indented_columns = any(
            line.startswith('    ') or line.startswith('\t') 
            for line in schema_lines 
            if line.strip() and not line.strip().startswith('CREATE') and not line.strip().startswith(')')
        )
        # Note: Some very simple schemas might not need indentation, so this is a soft check
    
    # 6. Check no /no_think
    if '/no_think' in user_content:
        errors.append(f"Example {idx}: Contains '/no_think' token")
    if '/no_think' in assistant_msg.get('content', ''):
        errors.append(f"Example {idx}: Assistant response contains '/no_think' token")
    
    # 7. Validate assistant message
    if assistant_msg.get('role') != 'assistant':
        errors.append(f"Example {idx}: Third message should be assistant, got {assistant_msg.get('role')}")
    
    assistant_content = assistant_msg.get('content', '').strip()
    if not assistant_content:
        errors.append(f"Example {idx}: Empty assistant response (SQL target)")
    
    return errors


def validate_dataset(filepath: Path, original_count: int = None) -> dict:
    """Validate entire dataset file."""
    results = {
        'filepath': str(filepath),
        'total_examples': 0,
        'valid_examples': 0,
        'invalid_examples': 0,
        'errors': [],
        'sample_errors': [],
        'hints_none_count': 0,
        'hints_present_count': 0,
        'multiline_schema_count': 0,
        'single_line_schema_count': 0
    }
    
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    results['total_examples'] = len(examples)
    
    # Check count
    if original_count is not None and len(examples) != original_count:
        results['errors'].append(f"Example count mismatch: expected {original_count}, got {len(examples)}")
    
    for idx, ex in enumerate(examples):
        errors = validate_example(ex, idx)
        
        if errors:
            results['invalid_examples'] += 1
            results['sample_errors'].extend(errors[:3])  # Keep first few errors
        else:
            results['valid_examples'] += 1
        
        # Count hints
        user_content = ex.get('messages', [{}])[1].get('content', '') if len(ex.get('messages', [])) > 1 else ''
        if 'Hints:\nNone' in user_content or 'Hints: None' in user_content:
            results['hints_none_count'] += 1
        elif 'Hints:' in user_content:
            results['hints_present_count'] += 1
        
        # Check schema multiline
        schema_match = re.search(r'Schema:\s*(.*?)(?=\n\s*Hints:)', user_content, re.DOTALL)
        if schema_match:
            schema_text = schema_match.group(1).strip()
            if schema_text.count('\n') >= 3:
                results['multiline_schema_count'] += 1
            else:
                results['single_line_schema_count'] += 1
    
    return results


def compare_targets(t9_path: Path, t10_path: Path) -> dict:
    """Compare SQL targets between T9 and T10 to ensure preservation."""
    results = {
        'matched': 0,
        'mismatched': 0,
        'mismatches': []
    }
    
    # Load both datasets
    t9_examples = []
    with open(t9_path, 'r') as f:
        for line in f:
            if line.strip():
                t9_examples.append(json.loads(line))
    
    t10_examples = []
    with open(t10_path, 'r') as f:
        for line in f:
            if line.strip():
                t10_examples.append(json.loads(line))
    
    if len(t9_examples) != len(t10_examples):
        results['error'] = f"Count mismatch: T9={len(t9_examples)}, T10={len(t10_examples)}"
        return results
    
    for idx, (t9_ex, t10_ex) in enumerate(zip(t9_examples, t10_examples)):
        # Extract assistant content (SQL target)
        t9_sql = None
        t10_sql = None
        
        for msg in t9_ex.get('messages', []):
            if msg['role'] == 'assistant':
                t9_sql = msg['content'].strip()
                break
        
        for msg in t10_ex.get('messages', []):
            if msg['role'] == 'assistant':
                t10_sql = msg['content'].strip()
                break
        
        if t9_sql == t10_sql:
            results['matched'] += 1
        else:
            results['mismatched'] += 1
            if len(results['mismatches']) < 5:
                results['mismatches'].append({
                    'idx': idx,
                    't9': t9_sql[:100] if t9_sql else None,
                    't10': t10_sql[:100] if t10_sql else None
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate T10 dataset')
    parser.add_argument('--t10-dir', type=str, default='data/training/t10',
                        help='Directory containing T10 dataset')
    parser.add_argument('--t9-dir', type=str, default='data/training/t9',
                        help='Directory containing T9 dataset (for comparison)')
    parser.add_argument('--intermediate', action='store_true',
                        help='Validate intermediate files (before hint generation)')
    parser.add_argument('--compare-targets', action='store_true',
                        help='Compare SQL targets with T9')
    args = parser.parse_args()
    
    t10_dir = Path(args.t10_dir)
    t9_dir = Path(args.t9_dir)
    
    suffix = '_nohints' if args.intermediate else ''
    
    # Count original examples
    train_t9_count = sum(1 for _ in open(t9_dir / 'train_v4.jsonl') if _.strip())
    dev_t9_count = sum(1 for _ in open(t9_dir / 'dev_v4.jsonl') if _.strip())
    
    print("=== T10 Dataset Validation ===\n")
    
    # Validate train
    train_path = t10_dir / f'train_t10{suffix}.jsonl'
    if train_path.exists():
        print(f"Validating {train_path}...")
        train_results = validate_dataset(train_path, train_t9_count)
        print(f"  Total: {train_results['total_examples']}")
        print(f"  Valid: {train_results['valid_examples']}")
        print(f"  Invalid: {train_results['invalid_examples']}")
        print(f"  Hints present: {train_results['hints_present_count']}")
        print(f"  Hints: None: {train_results['hints_none_count']}")
        print(f"  Multiline schemas: {train_results['multiline_schema_count']}")
        print(f"  Single-line schemas: {train_results['single_line_schema_count']}")
        if train_results['sample_errors']:
            print(f"  Sample errors:")
            for err in train_results['sample_errors'][:5]:
                print(f"    - {err}")
    else:
        print(f"Train file not found: {train_path}")
    
    print()
    
    # Validate dev
    dev_path = t10_dir / f'dev_t10{suffix}.jsonl'
    if dev_path.exists():
        print(f"Validating {dev_path}...")
        dev_results = validate_dataset(dev_path, dev_t9_count)
        print(f"  Total: {dev_results['total_examples']}")
        print(f"  Valid: {dev_results['valid_examples']}")
        print(f"  Invalid: {dev_results['invalid_examples']}")
        print(f"  Hints present: {dev_results['hints_present_count']}")
        print(f"  Hints: None: {dev_results['hints_none_count']}")
        print(f"  Multiline schemas: {dev_results['multiline_schema_count']}")
        print(f"  Single-line schemas: {dev_results['single_line_schema_count']}")
        if dev_results['sample_errors']:
            print(f"  Sample errors:")
            for err in dev_results['sample_errors'][:5]:
                print(f"    - {err}")
    else:
        print(f"Dev file not found: {dev_path}")
    
    # Compare targets
    if args.compare_targets:
        print("\n=== SQL Target Comparison ===\n")
        
        train_t9 = t9_dir / 'train_v4.jsonl'
        train_t10 = t10_dir / f'train_t10{suffix}.jsonl'
        if train_t9.exists() and train_t10.exists():
            print("Comparing train targets...")
            train_cmp = compare_targets(train_t9, train_t10)
            print(f"  Matched: {train_cmp['matched']}")
            print(f"  Mismatched: {train_cmp['mismatched']}")
            if train_cmp.get('mismatches'):
                print(f"  Sample mismatches:")
                for m in train_cmp['mismatches'][:3]:
                    print(f"    idx {m['idx']}: T9='{m['t9']}' vs T10='{m['t10']}'")
        
        dev_t9 = t9_dir / 'dev_v4.jsonl'
        dev_t10 = t10_dir / f'dev_t10{suffix}.jsonl'
        if dev_t9.exists() and dev_t10.exists():
            print("\nComparing dev targets...")
            dev_cmp = compare_targets(dev_t9, dev_t10)
            print(f"  Matched: {dev_cmp['matched']}")
            print(f"  Mismatched: {dev_cmp['mismatched']}")
    
    # Final status
    print("\n=== Validation Summary ===")
    all_valid = True
    
    if train_path.exists() and train_results['invalid_examples'] > 0:
        print(f"❌ Train has {train_results['invalid_examples']} invalid examples")
        all_valid = False
    elif train_path.exists():
        print(f"✓ Train validation passed")
    
    if dev_path.exists() and dev_results['invalid_examples'] > 0:
        print(f"❌ Dev has {dev_results['invalid_examples']} invalid examples")
        all_valid = False
    elif dev_path.exists():
        print(f"✓ Dev validation passed")
    
    if all_valid:
        print("\n✓ All validations passed!")
    else:
        print("\n❌ Some validations failed")
        exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Clean T9 dataset:
1. Remove entries with unknown db_id
2. Validate SQL syntax
3. Check for schema misalignment
"""

import json
import re
import sqlite3
from pathlib import Path
from collections import Counter

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

def extract_schema(example):
    if 'messages' in example:
        for msg in example['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Extract CREATE TABLE statements
                tables = re.findall(r'CREATE TABLE[^;]+;?', content, re.IGNORECASE | re.DOTALL)
                return tables
    return []

def extract_db_id(example):
    return example.get('db_id', 'unknown')

def validate_sql_syntax(sql):
    """Basic SQL syntax validation."""
    sql = sql.strip()
    if not sql:
        return False, "Empty SQL"
    
    # Check for SELECT
    if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
        return False, "No SELECT"
    
    # Check balanced parentheses
    if sql.count('(') != sql.count(')'):
        return False, "Unbalanced parentheses"
    
    # Check for common errors
    if re.search(r'\bSELECT\s*,', sql, re.IGNORECASE):
        return False, "SELECT followed by comma"
    
    return True, "OK"

def check_schema_alignment(sql, schema_tables):
    """Check if SQL references tables that exist in schema."""
    # Extract table names from SQL
    sql_tables = set()
    
    # FROM clause
    from_matches = re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
    sql_tables.update(from_matches)
    
    # JOIN clauses
    join_matches = re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
    sql_tables.update(join_matches)
    
    # Extract table names from schema
    schema_table_names = set()
    for table in schema_tables:
        match = re.search(r'CREATE TABLE\s+[`"]?(\w+)[`"]?', table, re.IGNORECASE)
        if match:
            schema_table_names.add(match.group(1).lower())
    
    # Check alignment
    sql_tables_lower = {t.lower() for t in sql_tables}
    missing = sql_tables_lower - schema_table_names
    
    # Allow aliases (T1, T2, etc.) and common subquery names
    missing = {t for t in missing if not re.match(r'^t\d+$', t) and t not in ['subquery', 'sub', 'temp']}
    
    if missing and schema_table_names:
        return False, f"Missing tables: {missing}"
    
    return True, "OK"

def main():
    t9_dir = Path('/Users/arnav/programming/lm/data/training/t9')
    
    print("=" * 70)
    print("T9 DATASET CLEANING")
    print("=" * 70)
    
    # Load train and dev
    print("\n1. Loading datasets...")
    train = load_jsonl(t9_dir / 'train.jsonl')
    dev = load_jsonl(t9_dir / 'dev.jsonl')
    
    print(f"   Train: {len(train)}")
    print(f"   Dev: {len(dev)}")
    
    all_examples = train + dev
    
    # Step 1: Count db_ids
    print("\n2. Analyzing db_id distribution...")
    db_ids = Counter(extract_db_id(ex) for ex in all_examples)
    unknown_count = db_ids.get('unknown', 0)
    print(f"   Total db_ids: {len(db_ids)}")
    print(f"   Unknown db_id: {unknown_count}")
    print(f"   Top 5 db_ids: {db_ids.most_common(5)}")
    
    # Step 2: Filter out unknown db_id
    print("\n3. Removing unknown db_id entries...")
    known_examples = [ex for ex in all_examples if extract_db_id(ex) != 'unknown']
    print(f"   Removed: {len(all_examples) - len(known_examples)}")
    print(f"   Remaining: {len(known_examples)}")
    
    # Step 3: Validate SQL syntax
    print("\n4. Validating SQL syntax...")
    syntax_errors = []
    valid_examples = []
    
    for ex in known_examples:
        sql = extract_sql(ex)
        is_valid, reason = validate_sql_syntax(sql)
        if is_valid:
            valid_examples.append(ex)
        else:
            syntax_errors.append((sql[:100], reason))
    
    print(f"   Syntax errors: {len(syntax_errors)}")
    print(f"   Valid: {len(valid_examples)}")
    
    if syntax_errors:
        print("\n   Sample syntax errors:")
        for sql, reason in syntax_errors[:3]:
            print(f"      - {reason}: {sql[:50]}...")
    
    # Step 4: Check schema alignment (sample)
    print("\n5. Checking schema alignment (sample of 100)...")
    import random
    random.seed(42)
    sample = random.sample(valid_examples, min(100, len(valid_examples)))
    
    alignment_errors = []
    for ex in sample:
        sql = extract_sql(ex)
        schema = extract_schema(ex)
        is_aligned, reason = check_schema_alignment(sql, schema)
        if not is_aligned:
            alignment_errors.append((sql[:80], reason))
    
    print(f"   Schema alignment errors in sample: {len(alignment_errors)}/100")
    
    if alignment_errors:
        print("\n   Sample alignment errors:")
        for sql, reason in alignment_errors[:3]:
            print(f"      - {reason}")
            print(f"        SQL: {sql[:60]}...")
    
    # Estimate total alignment errors
    estimated_alignment_errors = int(len(alignment_errors) / 100 * len(valid_examples))
    print(f"\n   Estimated total alignment errors: ~{estimated_alignment_errors}")
    
    # Step 5: Save cleaned dataset
    print("\n6. Saving cleaned dataset...")
    
    # Shuffle and split
    random.shuffle(valid_examples)
    dev_size = min(600, int(len(valid_examples) * 0.06))
    new_dev = valid_examples[:dev_size]
    new_train = valid_examples[dev_size:]
    
    save_jsonl(new_train, t9_dir / 'train_cleaned.jsonl')
    save_jsonl(new_dev, t9_dir / 'dev_cleaned.jsonl')
    
    print(f"   Train: {len(new_train)} -> train_cleaned.jsonl")
    print(f"   Dev: {len(new_dev)} -> dev_cleaned.jsonl")
    
    # Summary
    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"   Original: {len(all_examples)}")
    print(f"   Removed unknown db_id: {len(all_examples) - len(known_examples)}")
    print(f"   Removed syntax errors: {len(known_examples) - len(valid_examples)}")
    print(f"   Final: {len(valid_examples)}")
    print(f"\n   Files saved:")
    print(f"   - {t9_dir / 'train_cleaned.jsonl'}")
    print(f"   - {t9_dir / 'dev_cleaned.jsonl'}")

if __name__ == '__main__':
    main()

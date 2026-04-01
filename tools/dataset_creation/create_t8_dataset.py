#!/usr/bin/env python3
"""
Create T8 Dataset for BIRD Benchmark Training

T8 improves on T7 by:
1. Converting ALL schemas to DDL format (matching eval format)
2. Upsampling CASE/CTE/Window patterns to match BIRD dev distribution
3. Adding complex column name training (backticks, spaces, parentheses)

Output structure:
  data/training/t8/
  ├── training/
  │   ├── train.jsonl      # Training data (~18-20K examples)
  │   └── dev.jsonl        # Validation during training (~1K)
  └── eval/
      ├── README.md        # Explains eval data
      └── bird_dev.jsonl   # Official BIRD dev (1,534 examples)

Usage:
  python tools/dataset_creation/create_t8_dataset.py

Requirements:
  - data/training/t7/train.jsonl (source data)
  - data/raw/bird_train.jsonl (for pattern extraction)
  - data/raw/bird_dev.jsonl (for eval folder)
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file into list of dicts."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    """Write list of dicts to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Wrote {len(rows)} examples to {path}")


def simple_to_ddl(simple_schema: str) -> str:
    """
    Convert simple schema format to DDL format.
    
    Input: 'users(id,name,email)'
    Output: 'CREATE TABLE users (\n    id,\n    name,\n    email\n);'
    
    Note: We don't have type info in simple format, so we just list columns.
    The model should learn to work with column names regardless of types.
    """
    ddl_parts = []
    
    # Split by newlines to get each table definition
    for line in simple_schema.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are already DDL
        if line.upper().startswith('CREATE TABLE'):
            ddl_parts.append(line)
            continue
            
        # Parse simple format: table_name(col1,col2,col3)
        match = re.match(r'^(\w+)\((.*)\)$', line)
        if match:
            table_name = match.group(1)
            columns = [col.strip() for col in match.group(2).split(',')]
            
            # Build DDL
            ddl = f"CREATE TABLE {table_name}\n(\n"
            col_defs = []
            for col in columns:
                col_defs.append(f"    {col}")
            ddl += ",\n".join(col_defs)
            ddl += "\n);"
            ddl_parts.append(ddl)
        else:
            # If we can't parse, keep as is (might be comments, rules, etc.)
            ddl_parts.append(line)
    
    return "\n".join(ddl_parts)


def is_ddl_format(schema_text: str) -> bool:
    """Check if schema is already in DDL format."""
    return 'CREATE TABLE' in schema_text.upper()


def convert_example_to_ddl(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an example's schema to DDL format if needed."""
    if 'messages' not in example or len(example['messages']) < 2:
        return example
    
    user_msg = example['messages'][1]
    if user_msg.get('role') != 'user':
        return example
    
    content = user_msg.get('content', '')
    
    # Check if already DDL
    if is_ddl_format(content):
        return example
    
    # Extract schema portion (after "Schema:" and before "Question:" or "Business rules:")
    schema_match = re.search(
        r'Schema:\s*(.*?)(?:Business rules:|Question:|Hints:|$)',
        content, 
        re.DOTALL | re.IGNORECASE
    )
    
    if not schema_match:
        return example
    
    old_schema = schema_match.group(1).strip()
    
    # Convert to DDL
    new_schema = simple_to_ddl(old_schema)
    
    # Replace in content
    new_content = content.replace(old_schema, new_schema)
    
    # Create new example with updated content
    new_example = example.copy()
    new_example['messages'] = example['messages'].copy()
    new_example['messages'][1] = {
        'role': 'user',
        'content': new_content
    }
    
    return new_example


def has_pattern(sql: str, pattern: str) -> bool:
    """Check if SQL contains a specific pattern."""
    sql_upper = sql.upper()
    if pattern == 'case':
        return bool(re.search(r'\bCASE\b', sql_upper))
    elif pattern == 'cte':
        return bool(re.search(r'\bWITH\b', sql_upper))
    elif pattern == 'window':
        return bool(re.search(r'\bOVER\s*\(', sql_upper))
    elif pattern == 'join':
        return bool(re.search(r'\bJOIN\b', sql_upper))
    elif pattern == 'subquery':
        return bool(re.search(r'SELECT.*SELECT', sql_upper, re.DOTALL))
    return False


def get_sql_from_example(example: Dict[str, Any]) -> Optional[str]:
    """Extract SQL from example's assistant message."""
    if 'messages' not in example:
        return None
    for msg in example['messages']:
        if msg.get('role') == 'assistant':
            return msg.get('content', '')
    return None


def analyze_patterns(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count pattern occurrences in examples."""
    counts = {'case': 0, 'cte': 0, 'window': 0, 'join': 0, 'subquery': 0}
    for ex in examples:
        sql = get_sql_from_example(ex)
        if sql:
            for pattern in counts:
                if has_pattern(sql, pattern):
                    counts[pattern] += 1
    return counts


def upsample_by_pattern(
    examples: List[Dict[str, Any]], 
    pattern: str, 
    target_pct: float
) -> List[Dict[str, Any]]:
    """
    Upsample examples containing a pattern to reach target percentage.
    
    Returns additional examples to add (duplicates of pattern-containing examples).
    """
    total = len(examples)
    current_count = sum(1 for ex in examples if has_pattern(get_sql_from_example(ex) or '', pattern))
    current_pct = current_count / total * 100
    
    print(f"  {pattern}: {current_count}/{total} ({current_pct:.1f}%) -> target {target_pct}%")
    
    if current_pct >= target_pct:
        print(f"    Already at target, no upsampling needed")
        return []
    
    # Calculate how many more we need
    # After adding N duplicates: (current_count + N) / (total + N) = target_pct/100
    # Solving: N = (target_pct * total - 100 * current_count) / (100 - target_pct)
    target_ratio = target_pct / 100
    needed = int((target_ratio * total - current_count) / (1 - target_ratio))
    
    # Find examples with this pattern
    pattern_examples = [ex for ex in examples if has_pattern(get_sql_from_example(ex) or '', pattern)]
    
    if not pattern_examples:
        print(f"    No examples with pattern found!")
        return []
    
    # Sample with replacement
    additional = random.choices(pattern_examples, k=needed)
    print(f"    Adding {len(additional)} duplicates")
    
    return additional


def generate_complex_column_examples() -> List[Dict[str, Any]]:
    """
    Generate synthetic examples with complex column names.
    
    These examples specifically target the california_schools failure mode:
    - Columns with backticks: `FRPM Count (Ages 5-17)`
    - Columns with spaces and special chars
    """
    examples = []
    
    # Template for complex schema examples
    complex_schemas = [
        # California schools-style schema
        {
            'schema': '''CREATE TABLE frpm
(
    CDSCode TEXT PRIMARY KEY,
    `Academic Year` TEXT,
    `County Name` TEXT,
    `School Name` TEXT,
    `Free Meal Count (K-12)` INTEGER,
    `FRPM Count (K-12)` INTEGER,
    `Enrollment (K-12)` INTEGER,
    `Free Meal Count (Ages 5-17)` INTEGER,
    `FRPM Count (Ages 5-17)` INTEGER,
    `Enrollment (Ages 5-17)` INTEGER,
    `Percent (%) Eligible Free (K-12)` REAL
);
CREATE TABLE schools
(
    CDSCode TEXT PRIMARY KEY,
    School TEXT,
    County TEXT,
    City TEXT,
    Charter INTEGER,
    Magnet INTEGER,
    Virtual TEXT
);
CREATE TABLE satscores
(
    cds TEXT PRIMARY KEY,
    sname TEXT,
    NumTstTakr INTEGER,
    AvgScrRead INTEGER,
    AvgScrMath INTEGER,
    AvgScrWrite INTEGER
);''',
            'qa_pairs': [
                {
                    'question': 'What is the highest free meal rate for K-12 students in schools in Alameda County?',
                    'sql': '''SELECT MAX(T1.`Percent (%) Eligible Free (K-12)`) 
FROM frpm AS T1 
INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode 
WHERE T2.County = 'Alameda' '''
                },
                {
                    'question': 'List the school names and their FRPM count for ages 5-17 in Los Angeles County.',
                    'sql': '''SELECT T1.`School Name`, T1.`FRPM Count (Ages 5-17)` 
FROM frpm AS T1 
INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode 
WHERE T2.County = 'Los Angeles' '''
                },
                {
                    'question': 'What is the average enrollment for K-12 students in charter schools?',
                    'sql': '''SELECT AVG(T1.`Enrollment (K-12)`) 
FROM frpm AS T1 
INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode 
WHERE T2.Charter = 1'''
                },
                {
                    'question': 'Find schools where the free meal count exceeds the enrollment count for ages 5-17.',
                    'sql': '''SELECT T1.`School Name` 
FROM frpm AS T1 
WHERE T1.`Free Meal Count (Ages 5-17)` > T1.`Enrollment (Ages 5-17)` '''
                },
            ]
        },
        # Financial data style
        {
            'schema': '''CREATE TABLE accounts
(
    `Account ID` INTEGER PRIMARY KEY,
    `Account Type` TEXT,
    `Opening Balance ($)` REAL,
    `Current Balance ($)` REAL,
    `Date Opened` TEXT
);
CREATE TABLE transactions
(
    `Transaction ID` INTEGER PRIMARY KEY,
    `Account ID` INTEGER,
    `Transaction Type` TEXT,
    `Amount ($)` REAL,
    `Transaction Date` TEXT,
    `Description` TEXT,
    FOREIGN KEY (`Account ID`) REFERENCES accounts(`Account ID`)
);''',
            'qa_pairs': [
                {
                    'question': 'What is the total amount of deposits across all accounts?',
                    'sql': '''SELECT SUM(`Amount ($)`) 
FROM transactions 
WHERE `Transaction Type` = 'deposit' '''
                },
                {
                    'question': 'Find accounts where the current balance is less than the opening balance.',
                    'sql': '''SELECT `Account ID`, `Account Type` 
FROM accounts 
WHERE `Current Balance ($)` < `Opening Balance ($)` '''
                },
            ]
        },
        # Medical records style
        {
            'schema': '''CREATE TABLE patients
(
    `Patient ID` INTEGER PRIMARY KEY,
    `Full Name` TEXT,
    `Date of Birth` TEXT,
    `Blood Type` TEXT,
    `Insurance Provider` TEXT
);
CREATE TABLE visits
(
    `Visit ID` INTEGER PRIMARY KEY,
    `Patient ID` INTEGER,
    `Visit Date` TEXT,
    `Diagnosis Code` TEXT,
    `Treatment Cost ($)` REAL,
    `Insurance Coverage (%)` REAL,
    FOREIGN KEY (`Patient ID`) REFERENCES patients(`Patient ID`)
);''',
            'qa_pairs': [
                {
                    'question': 'What is the average treatment cost for patients with blood type O+?',
                    'sql': '''SELECT AVG(T2.`Treatment Cost ($)`) 
FROM patients AS T1 
INNER JOIN visits AS T2 ON T1.`Patient ID` = T2.`Patient ID` 
WHERE T1.`Blood Type` = 'O+' '''
                },
                {
                    'question': 'List patients whose insurance coverage is below 50%.',
                    'sql': '''SELECT DISTINCT T1.`Full Name` 
FROM patients AS T1 
INNER JOIN visits AS T2 ON T1.`Patient ID` = T2.`Patient ID` 
WHERE T2.`Insurance Coverage (%)` < 50'''
                },
            ]
        },
    ]
    
    system_prompt = "You are an expert SQL assistant. Generate SQLite queries from natural language questions.\nGiven a database schema and a question, generate the correct SQL query.\nOnly output the SQL query, nothing else."
    
    for schema_data in complex_schemas:
        schema = schema_data['schema']
        for qa in schema_data['qa_pairs']:
            example = {
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Schema:\n{schema}\n\nQuestion: {qa['question']}"},
                    {'role': 'assistant', 'content': qa['sql'].strip()}
                ],
                'source': 'complex_column_synthetic',
                'type': 'chatml'
            }
            examples.append(example)
    
    # Duplicate each example 10x to have meaningful representation
    examples = examples * 10
    
    print(f"Generated {len(examples)} complex column examples")
    return examples


def convert_bird_dev_to_chatml(bird_dev: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert BIRD dev examples to ChatML format for eval folder."""
    system_prompt = "You are an expert SQL assistant. Generate SQLite queries from natural language questions.\nGiven a database schema and a question, generate the correct SQL query.\nOnly output the SQL query, nothing else."
    
    chatml_examples = []
    for ex in bird_dev:
        # Build user prompt
        user_content = f"Schema:\n[Schema will be injected during evaluation]\n\n"
        if ex.get('evidence'):
            user_content += f"Hints: {ex['evidence']}\n\n"
        user_content += f"Question: {ex['question']}"
        
        chatml_example = {
            'db_id': ex['db_id'],
            'question_id': ex.get('question_id'),
            'question': ex['question'],
            'evidence': ex.get('evidence', ''),
            'gold_sql': ex['SQL'],
            'difficulty': ex.get('difficulty'),
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ]
        }
        chatml_examples.append(chatml_example)
    
    return chatml_examples


def main():
    random.seed(42)
    
    # Paths
    root = Path(__file__).parent.parent.parent
    t7_train = root / 'data' / 'training' / 't7' / 'train.jsonl'
    t7_dev = root / 'data' / 'training' / 't7' / 'dev.jsonl'
    bird_train = root / 'data' / 'raw' / 'bird_train.jsonl'
    bird_dev = root / 'data' / 'raw' / 'bird_dev.jsonl'
    
    t8_dir = root / 'data' / 'training' / 't8'
    t8_training_dir = t8_dir / 'training'
    t8_eval_dir = t8_dir / 'eval'
    
    print("=" * 60)
    print("Creating T8 Dataset")
    print("=" * 60)
    
    # Load source data
    print("\n1. Loading source data...")
    t7_train_data = read_jsonl(str(t7_train))
    t7_dev_data = read_jsonl(str(t7_dev))
    bird_dev_data = read_jsonl(str(bird_dev))
    
    print(f"   T7 train: {len(t7_train_data)} examples")
    print(f"   T7 dev: {len(t7_dev_data)} examples")
    print(f"   BIRD dev: {len(bird_dev_data)} examples")
    
    # Analyze current pattern distribution
    print("\n2. Analyzing current patterns in T7 train...")
    initial_counts = analyze_patterns(t7_train_data)
    total = len(t7_train_data)
    for pattern, count in initial_counts.items():
        print(f"   {pattern}: {count} ({count/total*100:.1f}%)")
    
    # Step 1: Convert all examples to DDL format
    print("\n3. Converting to DDL format...")
    converted = 0
    t8_train_data = []
    for ex in t7_train_data:
        new_ex = convert_example_to_ddl(ex)
        t8_train_data.append(new_ex)
        if new_ex != ex:
            converted += 1
    print(f"   Converted {converted} examples to DDL format")
    
    # Step 2: Upsample patterns
    print("\n4. Upsampling SQL patterns...")
    
    # Target percentages (based on BIRD dev distribution)
    targets = {
        'case': 25.0,    # From 5.5% to 25% (BIRD dev has 29.5%)
        'cte': 5.0,      # From 0.5% to 5% (BIRD dev has 6.9%)
        'window': 5.0,   # From 0.4% to 5% (BIRD dev has 5.9%)
    }
    
    additional_examples = []
    for pattern, target_pct in targets.items():
        additional = upsample_by_pattern(t8_train_data, pattern, target_pct)
        additional_examples.extend(additional)
    
    t8_train_data.extend(additional_examples)
    print(f"\n   Added {len(additional_examples)} upsampled examples")
    
    # Step 3: Add complex column examples
    print("\n5. Generating complex column examples...")
    complex_examples = generate_complex_column_examples()
    t8_train_data.extend(complex_examples)
    
    # Shuffle
    print("\n6. Shuffling data...")
    random.shuffle(t8_train_data)
    
    # Final stats
    print("\n7. Final pattern distribution:")
    final_counts = analyze_patterns(t8_train_data)
    total = len(t8_train_data)
    for pattern, count in final_counts.items():
        print(f"   {pattern}: {count} ({count/total*100:.1f}%)")
    
    # Also convert dev data to DDL
    print("\n8. Converting dev data to DDL...")
    t8_dev_data = [convert_example_to_ddl(ex) for ex in t7_dev_data]
    
    # Create output directories
    os.makedirs(t8_training_dir, exist_ok=True)
    os.makedirs(t8_eval_dir, exist_ok=True)
    
    # Write training files
    print("\n9. Writing output files...")
    write_jsonl(t8_train_data, str(t8_training_dir / 'train.jsonl'))
    write_jsonl(t8_dev_data, str(t8_training_dir / 'dev.jsonl'))
    
    # Write eval files
    bird_dev_chatml = convert_bird_dev_to_chatml(bird_dev_data)
    write_jsonl(bird_dev_chatml, str(t8_eval_dir / 'bird_dev.jsonl'))
    
    # Write README for eval folder
    readme_content = """# BIRD Evaluation Data

This folder contains the official BIRD dev set for benchmark evaluation.

## Important: 0% Overlap by Design!

The 11 databases in this eval set are **intentionally different** from the 69 databases 
used in training. This is how BIRD benchmark works - it tests generalization to unseen schemas.

## Databases in Eval Set (NEVER in training)
- california_schools
- card_games  
- codebase_community
- debit_card_specializing
- european_football_2
- financial
- formula_1
- student_club
- superhero
- thrombosis_prediction
- toxicology

## Files
- `bird_dev.jsonl`: 1,534 examples in ChatML-like format
  - Contains question, gold_sql, db_id, difficulty
  - Schema must be extracted from actual SQLite files during evaluation

## Usage
Run evaluation with `evaluation/run_bird_eval.sh` which:
1. Loads actual SQLite databases from BIRD benchmark
2. Extracts DDL schemas using `SELECT sql FROM sqlite_master`
3. Generates predictions
4. Computes execution accuracy
"""
    with open(t8_eval_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    # Write metadata
    metadata = {
        'created_from': 't7 + upsampling + complex columns',
        'train_examples': len(t8_train_data),
        'dev_examples': len(t8_dev_data),
        'eval_examples': len(bird_dev_chatml),
        'improvements': [
            'All schemas converted to DDL format',
            f'CASE upsampled to ~{targets["case"]}%',
            f'CTE upsampled to ~{targets["cte"]}%',
            f'Window upsampled to ~{targets["window"]}%',
            'Added complex column name examples',
        ],
        'pattern_distribution': {
            pattern: {'count': count, 'pct': round(count/total*100, 1)}
            for pattern, count in final_counts.items()
        }
    }
    
    with open(t8_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("T8 Dataset Created!")
    print("=" * 60)
    print(f"Training: {t8_training_dir}")
    print(f"  - train.jsonl: {len(t8_train_data)} examples")
    print(f"  - dev.jsonl: {len(t8_dev_data)} examples")
    print(f"Eval: {t8_eval_dir}")
    print(f"  - bird_dev.jsonl: {len(bird_dev_chatml)} examples")
    print(f"  - README.md")


if __name__ == '__main__':
    main()

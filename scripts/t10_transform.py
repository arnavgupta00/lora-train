#!/usr/bin/env python3
"""
T10 Dataset Transform Script

Transforms T9 dataset to T10 format:
- Standardizes system prompt
- Reformats user prompt to canonical structure (Schema: -> Hints: -> Question:)
- Ensures schema is proper multi-line DDL
- Adds 'Hints: None' for examples without hints
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


def format_schema_multiline(schema_text: str) -> str:
    """
    Ensure schema is formatted as proper multi-line DDL.
    - Each CREATE TABLE on its own line
    - Column definitions indented
    - Preserve exact identifiers, quoting, spaces, parentheses, punctuation
    """
    # If already multiline with proper structure, return as-is
    lines = schema_text.strip().split('\n')
    if len(lines) > 5:
        # Check if it's already well-formatted (has indented columns)
        has_indented = any(line.startswith('    ') or line.startswith('\t') for line in lines)
        if has_indented:
            return schema_text.strip()
    
    # Handle single-line or poorly formatted schemas
    # Split on CREATE TABLE boundaries
    result_lines = []
    
    # Pattern to split CREATE TABLE statements while preserving them
    # This handles: CREATE TABLE, CREATE TABLE IF NOT EXISTS, etc.
    parts = re.split(r'(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)', schema_text, flags=re.IGNORECASE)
    
    statements = []
    i = 1
    while i < len(parts):
        if i + 1 < len(parts):
            stmt = parts[i] + parts[i + 1]
            statements.append(stmt.strip())
        i += 2
    
    if not statements:
        # Fallback: schema might be already split or different format
        return schema_text.strip()
    
    for stmt in statements:
        formatted = format_single_create_table(stmt)
        result_lines.append(formatted)
    
    return '\n'.join(result_lines)


def format_single_create_table(stmt: str) -> str:
    """Format a single CREATE TABLE statement with proper indentation."""
    # Check if already multiline
    if stmt.count('\n') > 2:
        return stmt
    
    # Find the opening parenthesis for columns
    paren_match = re.search(r'\(\s*', stmt)
    if not paren_match:
        return stmt
    
    start_idx = paren_match.end()
    
    # Find matching closing parenthesis (handle nested parens)
    depth = 1
    end_idx = start_idx
    for i in range(start_idx, len(stmt)):
        if stmt[i] == '(':
            depth += 1
        elif stmt[i] == ')':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    
    if depth != 0:
        return stmt  # Unbalanced parens, return as-is
    
    header = stmt[:paren_match.start()].strip()
    columns_str = stmt[start_idx:end_idx]
    footer = stmt[end_idx+1:].strip()  # +1 to skip the closing paren we already account for
    
    # Split columns by comma, but not commas inside parentheses
    columns = split_columns(columns_str)
    
    # Format
    formatted_columns = []
    for col in columns:
        col = col.strip()
        if col:
            formatted_columns.append(f"    {col}")
    
    result = f"{header}\n(\n"
    result += ",\n".join(formatted_columns)
    result += "\n)"
    if footer:
        result += footer
    
    return result


def split_columns(columns_str: str) -> list:
    """Split column definitions by comma, respecting parentheses."""
    columns = []
    current = []
    depth = 0
    
    for char in columns_str:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            columns.append(''.join(current))
            current = []
        else:
            current.append(char)
    
    if current:
        columns.append(''.join(current))
    
    return columns


def extract_components(user_content: str) -> dict:
    """
    Extract schema, hints, and question from user content.
    Handles multiple T9 format variations.
    """
    components = {
        'schema': None,
        'hints': None,
        'question': None
    }
    
    # Pattern 1: Has Schema:, Hints:, Question: labels (standard T9 format)
    if 'Schema:' in user_content and 'Question:' in user_content:
        schema_match = re.search(r'Schema:\s*(.*?)(?=\n\s*Hints:|\n\s*Question:)', user_content, re.DOTALL)
        hints_match = re.search(r'Hints:\s*(.*?)(?=\n\s*Question:)', user_content, re.DOTALL)
        question_match = re.search(r'Question:\s*(.*)', user_content, re.DOTALL)
        
        if schema_match:
            components['schema'] = schema_match.group(1).strip()
        if hints_match:
            hints_text = hints_match.group(1).strip()
            if hints_text and hints_text.lower() != 'none':
                components['hints'] = hints_text
        if question_match:
            components['question'] = question_match.group(1).strip()
    
    # Pattern 2: Question before Schema (no labels)
    # Format: "{question}\n\nSchema:\n{schema}"
    elif 'Schema:' in user_content and 'Question:' not in user_content:
        schema_idx = user_content.find('Schema:')
        question_part = user_content[:schema_idx].strip()
        schema_part = user_content[schema_idx + 7:].strip()  # 7 = len('Schema:')
        
        components['question'] = question_part
        components['schema'] = schema_part
        components['hints'] = None
    
    # Pattern 3: No Schema: label but has CREATE TABLE
    elif 'CREATE TABLE' in user_content:
        # Try to find where schema starts
        create_idx = user_content.find('CREATE TABLE')
        if create_idx > 0:
            components['question'] = user_content[:create_idx].strip()
            components['schema'] = user_content[create_idx:].strip()
        else:
            # Schema is at the beginning - look for question pattern
            components['schema'] = user_content.strip()
    
    return components


def transform_example(example: dict) -> dict:
    """Transform a single T9 example to T10 format."""
    messages = example.get('messages', [])
    
    # Extract user and assistant messages
    user_msg = None
    assistant_msg = None
    
    for msg in messages:
        if msg['role'] == 'user':
            user_msg = msg['content']
        elif msg['role'] == 'assistant':
            assistant_msg = msg['content']
    
    if not user_msg or not assistant_msg:
        return None
    
    # Extract components
    components = extract_components(user_msg)
    
    if not components['schema'] or not components['question']:
        # Try harder to extract
        # Sometimes the entire content is: question\n\nSchema:\nCREATE...
        if 'Schema:' in user_msg:
            parts = user_msg.split('Schema:', 1)
            if len(parts) == 2:
                components['question'] = parts[0].strip()
                components['schema'] = parts[1].strip()
    
    if not components['schema'] or not components['question']:
        return None
    
    # Format schema as proper multi-line DDL
    formatted_schema = format_schema_multiline(components['schema'])
    
    # Build canonical user prompt
    hints_text = components['hints'] if components['hints'] else 'None'
    
    canonical_user = f"""Schema:
{formatted_schema}

Hints:
{hints_text}

Question:
{components['question']}"""
    
    # Build T10 example
    t10_example = {
        'messages': [
            {'role': 'system', 'content': STANDARDIZED_SYSTEM_PROMPT},
            {'role': 'user', 'content': canonical_user},
            {'role': 'assistant', 'content': assistant_msg}
        ]
    }
    
    # Preserve db_id if present
    if 'db_id' in example:
        t10_example['db_id'] = example['db_id']
    
    return t10_example


def transform_dataset(input_path: Path, output_path: Path) -> dict:
    """Transform entire dataset file."""
    stats = {
        'total': 0,
        'transformed': 0,
        'failed': 0,
        'had_hints': 0,
        'hints_none': 0
    }
    
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    stats['total'] = len(examples)
    
    transformed = []
    for ex in examples:
        t10_ex = transform_example(ex)
        if t10_ex:
            transformed.append(t10_ex)
            stats['transformed'] += 1
            
            # Check hints
            user_content = t10_ex['messages'][1]['content']
            if 'Hints:\nNone' in user_content:
                stats['hints_none'] += 1
            else:
                stats['had_hints'] += 1
        else:
            stats['failed'] += 1
    
    # Write output
    with open(output_path, 'w') as f:
        for ex in transformed:
            f.write(json.dumps(ex) + '\n')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Transform T9 to T10 dataset format')
    parser.add_argument('--t9-dir', type=str, default='data/training/t9',
                        help='Directory containing T9 dataset')
    parser.add_argument('--t10-dir', type=str, default='data/training/t10',
                        help='Directory for T10 output')
    parser.add_argument('--intermediate', action='store_true',
                        help='Generate intermediate files (before hint generation)')
    args = parser.parse_args()
    
    t9_dir = Path(args.t9_dir)
    t10_dir = Path(args.t10_dir)
    t10_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_nohints' if args.intermediate else ''
    
    # Transform train
    train_input = t9_dir / 'train_v4.jsonl'
    train_output = t10_dir / f'train_t10{suffix}.jsonl'
    print(f"Transforming {train_input} -> {train_output}")
    train_stats = transform_dataset(train_input, train_output)
    
    # Transform dev
    dev_input = t9_dir / 'dev_v4.jsonl'
    dev_output = t10_dir / f'dev_t10{suffix}.jsonl'
    print(f"Transforming {dev_input} -> {dev_output}")
    dev_stats = transform_dataset(dev_input, dev_output)
    
    # Print summary
    print("\n=== Transformation Summary ===")
    print(f"\nTrain set:")
    print(f"  Total examples: {train_stats['total']}")
    print(f"  Transformed: {train_stats['transformed']}")
    print(f"  Failed: {train_stats['failed']}")
    print(f"  With hints: {train_stats['had_hints']}")
    print(f"  Hints: None: {train_stats['hints_none']}")
    
    print(f"\nDev set:")
    print(f"  Total examples: {dev_stats['total']}")
    print(f"  Transformed: {dev_stats['transformed']}")
    print(f"  Failed: {dev_stats['failed']}")
    print(f"  With hints: {dev_stats['had_hints']}")
    print(f"  Hints: None: {dev_stats['hints_none']}")


if __name__ == '__main__':
    main()

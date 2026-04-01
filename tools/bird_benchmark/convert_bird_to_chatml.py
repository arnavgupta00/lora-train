#!/usr/bin/env python3
"""
Convert BIRD dataset to ChatML format for training.
"""

import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Only output the SQL query, nothing else."""


def convert_bird_to_chatml(input_path: str, output_path: str, include_evidence: bool = True):
    """
    Convert BIRD train format to ChatML format.
    
    Input format (BIRD):
    {
        "db_id": "...",
        "question": "...",
        "evidence": "...",  # hints/clarifications
        "SQL": "...",
        "schema": "CREATE TABLE ..."
    }
    
    Output format (ChatML):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Schema:\n...\n\nQuestion: ..."},
            {"role": "assistant", "content": "SELECT ..."}
        ]
    }
    """
    
    examples = []
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            
            # Build user message
            user_content = f"Schema:\n{ex['schema']}\n\n"
            
            if include_evidence and ex.get('evidence'):
                user_content += f"Hints: {ex['evidence']}\n\n"
            
            user_content += f"Question: {ex['question']}"
            
            chatml = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ex['SQL']}
                ],
                "db_id": ex['db_id']  # Keep for reference
            }
            examples.append(chatml)
    
    # Save
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Converted {len(examples)} examples to {output_path}")
    return examples


def convert_bird_dev_to_chatml(input_path: str, output_path: str):
    """
    Convert BIRD dev format (different structure - no schema field).
    For dev, we'll need to include schema from DB files or use a placeholder.
    """
    
    examples = []
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            
            # Dev format has: question_id, db_id, question, evidence, SQL, difficulty
            # No schema field - would need actual DB files to get it
            user_content = f"Database: {ex['db_id']}\n\n"
            
            if ex.get('evidence'):
                user_content += f"Hints: {ex['evidence']}\n\n"
            
            user_content += f"Question: {ex['question']}"
            
            chatml = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ex['SQL']}
                ],
                "db_id": ex['db_id'],
                "difficulty": ex.get('difficulty'),
                "question_id": ex.get('question_id')
            }
            examples.append(chatml)
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Converted {len(examples)} dev examples to {output_path}")
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input BIRD jsonl file")
    parser.add_argument("--output", required=True, help="Output ChatML jsonl file")
    parser.add_argument("--dev", action="store_true", help="Use dev format (no schema)")
    parser.add_argument("--no-evidence", action="store_true", help="Don't include evidence hints")
    args = parser.parse_args()
    
    if args.dev:
        convert_bird_dev_to_chatml(args.input, args.output)
    else:
        convert_bird_to_chatml(args.input, args.output, include_evidence=not args.no_evidence)

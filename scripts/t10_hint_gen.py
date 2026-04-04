#!/usr/bin/env python3
"""
T10 Hint Generation Script

Generates hints for examples using LLM inference.
Saves checkpoints after each batch to allow resumption.
"""

import json
import sqlite3
import os
from pathlib import Path

# Paths
QUEUE_FILE = Path('/Users/arnav/programming/lm/data/training/t10/hint_queue.jsonl')
CHECKPOINT_FILE = Path('/Users/arnav/programming/lm/data/training/t10/hints_checkpoint.jsonl')
DB_PATH = Path(os.path.expanduser('~/.copilot/session-state/5ac2d0d6-7f02-441b-8056-972558305eb7/session.db'))

HINT_GENERATION_PROMPT = r"""You are generating SQL hints for a text-to-SQL training dataset.

Given a database schema and a natural language question, generate helpful hints that would help a model write the correct SQL query.

Use ONLY these 4 hint types:
1. **Derived metric**: When the question implies a calculation
   Example: `eligible free rate = \`Free Meal Count (K-12)\` / \`Enrollment (K-12)\``
2. **Disambiguating column meaning**: When a column name could be misunderstood
   Example: `'weight' refers to weight in pounds`
3. **Categorical value mapping**: When the question uses natural language for a coded value
   Example: `'active' customer refers to status = 'A'`
4. **Join key mapping**: When join relationships are non-obvious
   Example: `\`frpm\`.\`CDSCode\` joins \`schools\`.\`CDSCode\``

Rules:
- Keep hints SHORT and line-based
- Use exact column/table names from schema with backticks if they contain special chars
- Only provide hints that add real value - don't restate the obvious
- If the question and schema are straightforward, respond with just: None

Format your response as:
- hint 1
- hint 2

Or just: None

Schema:
{schema}

Question:
{question}

Hints:"""


def load_queue():
    """Load the hint generation queue."""
    items = []
    with open(QUEUE_FILE) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def get_pending_items(limit=100):
    """Get pending items from database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        SELECT example_id, dataset, original_idx 
        FROM hint_progress 
        WHERE status = 'pending' 
        ORDER BY example_id 
        LIMIT ?
    ''', (limit,))
    items = cur.fetchall()
    conn.close()
    return items


def load_queue_item(example_id):
    """Load full queue item by ID."""
    with open(QUEUE_FILE) as f:
        for line in f:
            item = json.loads(line)
            if item['example_id'] == example_id:
                return item
    return None


def save_checkpoint(example_id, hint_text):
    """Save a generated hint to checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(json.dumps({
            'example_id': example_id,
            'hint': hint_text
        }) + '\n')


def update_status(example_id, status, hint_text=None):
    """Update item status in database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if hint_text:
        cur.execute('''
            UPDATE hint_progress 
            SET status = ?, hint_text = ?, updated_at = CURRENT_TIMESTAMP
            WHERE example_id = ?
        ''', (status, hint_text, example_id))
    else:
        cur.execute('''
            UPDATE hint_progress 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE example_id = ?
        ''', (status, example_id))
    conn.commit()
    conn.close()


def get_progress():
    """Get current progress stats."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        SELECT status, COUNT(*) FROM hint_progress GROUP BY status
    ''')
    stats = dict(cur.fetchall())
    conn.close()
    return stats


def format_prompt(item):
    """Format the hint generation prompt."""
    return HINT_GENERATION_PROMPT.format(
        schema=item['schema'][:4000],  # Truncate very long schemas
        question=item['question']
    )


if __name__ == '__main__':
    # Show current progress
    stats = get_progress()
    print("Current progress:")
    for status, count in stats.items():
        print(f"  {status}: {count}")
    
    # Get next batch of pending items
    pending = get_pending_items(limit=10)
    print(f"\nNext {len(pending)} items to process:")
    for example_id, dataset, idx in pending:
        print(f"  {example_id}")

#!/usr/bin/env python3
"""
T10 Utilities Module

Shared utilities for T10-aligned training, prediction, and evaluation:
- T10 system prompt constant
- Prompt building and validation
- Schema extraction from SQLite databases
- Schema formatting
"""

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# T10 System Prompt (exact, canonical)
# =============================================================================

T10_SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Copy table names and column names exactly from the schema.
Never invent normalized identifiers.
If an identifier contains spaces, punctuation, %, hyphens, slashes, or parentheses, use it exactly and wrap it in backticks.
Use only the tables and columns that exist in the schema.
Only output the SQL query, nothing else."""


def get_t10_system_prompt_hash() -> str:
    """Get SHA256 hash of the T10 system prompt for verification."""
    return hashlib.sha256(T10_SYSTEM_PROMPT.encode()).hexdigest()[:16]


# =============================================================================
# Schema Extraction
# =============================================================================

def get_ddl_schema_from_db(db_path: str) -> str:
    """
    Extract CREATE TABLE statements from SQLite database.
    
    Returns multiline DDL schema with proper formatting.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND sql IS NOT NULL
            ORDER BY name
        """)
        create_statements = []
        for row in cursor.fetchall():
            if row[0]:
                # Clean up the SQL but preserve structure
                sql = row[0].strip()
                create_statements.append(sql)
        conn.close()
        return "\n".join(create_statements)
    except Exception as e:
        raise RuntimeError(f"Failed to extract schema from {db_path}: {e}")


def find_database(db_dir: str, db_id: str) -> Optional[str]:
    """Find SQLite database file for a given db_id."""
    db_dir = Path(db_dir)
    
    # Primary path: db_dir/db_id/db_id.sqlite
    db_file = db_dir / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    
    # Fallback patterns
    for pattern in [f"*/{db_id}.sqlite", f"*/{db_id}.db"]:
        matches = list(db_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None


# =============================================================================
# Schema Formatting
# =============================================================================

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
    
    formatted = []
    for stmt in statements:
        formatted.append(_format_single_create_table(stmt))
    
    return '\n'.join(formatted)


def _format_single_create_table(stmt: str) -> str:
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
    footer = stmt[end_idx+1:].strip()
    
    # Split columns by comma, but not commas inside parentheses
    columns = _split_columns(columns_str)
    
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


def _split_columns(columns_str: str) -> List[str]:
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


# =============================================================================
# Prompt Building
# =============================================================================

def build_t10_prompt(
    schema: str,
    question: str,
    hints: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a T10-formatted prompt.
    
    Args:
        schema: DDL schema (will be formatted as multiline)
        question: Natural language question
        hints: Optional hint text (None or empty -> "None")
    
    Returns:
        Dict with 'system' and 'user' keys containing prompt text
    """
    # Format schema
    formatted_schema = format_schema_multiline(schema)
    
    # Handle hints
    hints_text = hints.strip() if hints and hints.strip() else "None"
    
    # Build user prompt
    user_content = f"""Schema:
{formatted_schema}

Hints:
{hints_text}

Question:
{question}"""
    
    return {
        "system": T10_SYSTEM_PROMPT,
        "user": user_content,
    }


def build_t10_messages(
    schema: str,
    question: str,
    hints: Optional[str] = None,
    assistant_response: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build T10-formatted messages list (for chat template).
    
    Args:
        schema: DDL schema
        question: Natural language question
        hints: Optional hint text
        assistant_response: Optional SQL response (for training data)
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    prompt = build_t10_prompt(schema, question, hints)
    
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages


# =============================================================================
# Prompt Validation
# =============================================================================

class T10ValidationError(Exception):
    """Raised when T10 prompt validation fails."""
    pass


def validate_t10_prompt(
    system: str,
    user: str,
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate that a prompt follows T10 contract.
    
    Args:
        system: System prompt text
        user: User prompt text
        strict: If True, raise exception on failure
    
    Returns:
        (is_valid, list_of_errors)
    
    Raises:
        T10ValidationError if strict=True and validation fails
    """
    errors = []
    
    # Check system prompt matches T10 exactly
    if system.strip() != T10_SYSTEM_PROMPT.strip():
        errors.append("System prompt does not match T10 contract")
    
    # Check user prompt structure
    if "Schema:" not in user:
        errors.append("User prompt missing 'Schema:' section")
    
    if "Hints:" not in user:
        errors.append("User prompt missing 'Hints:' section")
    
    if "Question:" not in user:
        errors.append("User prompt missing 'Question:' section")
    
    # Check order: Schema -> Hints -> Question
    schema_idx = user.find("Schema:")
    hints_idx = user.find("Hints:")
    question_idx = user.find("Question:")
    
    if schema_idx != -1 and hints_idx != -1 and schema_idx > hints_idx:
        errors.append("'Schema:' must come before 'Hints:'")
    
    if hints_idx != -1 and question_idx != -1 and hints_idx > question_idx:
        errors.append("'Hints:' must come before 'Question:'")
    
    # Check for forbidden patterns
    if "/no_think" in user or "/think" in user:
        errors.append("User prompt contains forbidden '/no_think' or '/think'")
    
    if "/no_think" in system or "/think" in system:
        errors.append("System prompt contains forbidden '/no_think' or '/think'")
    
    # Check schema is multiline
    if schema_idx != -1 and hints_idx != -1:
        schema_section = user[schema_idx + 7:hints_idx].strip()
        if "\n" not in schema_section:
            errors.append("Schema section is not multiline (should have proper DDL formatting)")
    
    is_valid = len(errors) == 0
    
    if strict and not is_valid:
        raise T10ValidationError(f"T10 validation failed: {'; '.join(errors)}")
    
    return is_valid, errors


def validate_t10_messages(
    messages: List[Dict[str, str]],
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate a messages list follows T10 contract.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        strict: If True, raise exception on failure
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check structure
    if len(messages) < 2:
        errors.append("Messages must have at least system and user roles")
        if strict:
            raise T10ValidationError(f"T10 validation failed: {'; '.join(errors)}")
        return False, errors
    
    if messages[0].get("role") != "system":
        errors.append("First message must be system role")
    
    if messages[1].get("role") != "user":
        errors.append("Second message must be user role")
    
    if errors:
        if strict:
            raise T10ValidationError(f"T10 validation failed: {'; '.join(errors)}")
        return False, errors
    
    # Validate content
    system = messages[0].get("content", "")
    user = messages[1].get("content", "")
    
    return validate_t10_prompt(system, user, strict=strict)


# =============================================================================
# Prompt Parity Checking
# =============================================================================

def check_prompt_parity(
    training_file: str,
    eval_system_prompt: str,
) -> Tuple[bool, str]:
    """
    Check that eval system prompt matches training data system prompt.
    
    Args:
        training_file: Path to training JSONL file
        eval_system_prompt: System prompt to verify
    
    Returns:
        (is_match, message)
    """
    # Load first training example to get system prompt
    with open(training_file, 'r') as f:
        first_line = f.readline()
        if not first_line.strip():
            return False, "Training file is empty"
        
        example = json.loads(first_line)
        messages = example.get("messages", [])
        
        if not messages or messages[0].get("role") != "system":
            return False, "Training file first example has no system message"
        
        training_system = messages[0].get("content", "").strip()
    
    eval_system = eval_system_prompt.strip()
    
    if training_system == eval_system:
        return True, "System prompts match exactly"
    else:
        # Provide diff info
        if len(training_system) != len(eval_system):
            return False, f"System prompt length mismatch: training={len(training_system)}, eval={len(eval_system)}"
        else:
            # Find first difference
            for i, (c1, c2) in enumerate(zip(training_system, eval_system)):
                if c1 != c2:
                    return False, f"System prompt differs at position {i}: training='{c1}', eval='{c2}'"
            return False, "System prompts differ (unknown reason)"


# =============================================================================
# SQL Normalization (for evaluation)
# =============================================================================

def normalize_sql(sql: str) -> str:
    """
    Clean up raw model output into a single executable SQL statement.
    
    Handles:
    - Qwen3 thinking tags (<think>...</think>)
    - Markdown code fences (```sql ... ```)
    - Multiple statements (keeps only the first)
    - Whitespace normalization
    """
    if not sql:
        return ""
    
    s = sql.strip()
    
    # Strip Qwen3 thinking tags
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    
    # Strip markdown code fences
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    
    # Keep only the first statement
    s = s.split(";")[0].strip()
    
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    return s


# =============================================================================
# Git Utilities (for run manifest)
# =============================================================================

def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    # Test the module
    print("T10 Utilities Module")
    print("=" * 60)
    print(f"System prompt hash: {get_t10_system_prompt_hash()}")
    print()
    
    # Test prompt building
    test_schema = """CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT
);"""
    test_question = "How many users are there?"
    test_hints = "Count all rows in users table"
    
    prompt = build_t10_prompt(test_schema, test_question, test_hints)
    print("Test prompt (with hints):")
    print("-" * 40)
    print(f"System: {prompt['system'][:50]}...")
    print(f"User:\n{prompt['user']}")
    print()
    
    # Test validation
    is_valid, errors = validate_t10_prompt(prompt["system"], prompt["user"], strict=False)
    print(f"Validation: valid={is_valid}, errors={errors}")
    
    # Test with None hints
    prompt_no_hints = build_t10_prompt(test_schema, test_question, None)
    print("\nTest prompt (no hints):")
    print("-" * 40)
    print(f"User:\n{prompt_no_hints['user']}")

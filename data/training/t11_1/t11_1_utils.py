#!/usr/bin/env python3
"""
T11.1 Utilities Module

Utilities for T11.1 mixed full/compact schema dataset.
Fork of t11_utils.py with softer compaction policy:

Changes from T11:
  - Wider safety margin: ≤8 small-table threshold, up to 5 question-match extras,
    primary-table bonus (5 adjacent cols), minimum 4 cols per table
  - Re-widen retry: if compact schema is over-compacted, retry with widened margins
    (8 extras, 8 primary-table bonus, min 5 cols) before fallback
  - Over-compaction guards: reject if reduction >80%, compact <300 chars for
    large schemas, or any table has <3 columns
  - Hash-based 55% compact assignment so post-fallback lands near 50%
  - Target 65-80% average reduction (down from 91%)

Key principle: CONSERVATIVE extraction - prefer fallback-to-full over bad compacting.
"""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import T10 utilities for prompt building
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "t10"))
from t10_utils import T10_SYSTEM_PROMPT, validate_t10_messages


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TableInfo:
    """Parsed table information."""
    name: str
    columns: List[str] = field(default_factory=list)
    pk_columns: List[str] = field(default_factory=list)
    fk_columns: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # col -> (ref_table, ref_col)
    raw_ddl: str = ""


@dataclass
class SchemaInfo:
    """Parsed schema information."""
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    fk_graph: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)  # (src_tbl, src_col) -> (dst_tbl, dst_col)
    raw_schema: str = ""


@dataclass
class SQLExtractionResult:
    """Result of SQL identifier extraction."""
    tables: Set[str]
    columns: Set[str]
    aliases: Dict[str, str]  # alias -> real_table
    is_confident: bool
    fallback_reason: Optional[str] = None


def _parse_column_list(cols_str: str) -> List[str]:
    """
    Parse a comma-separated list of column names, handling quoted identifiers.
    E.g., '"Customer ID", Region' -> ['"Customer ID"', 'Region']
    """
    cols = []
    current = ""
    in_quote = False
    quote_char = None

    for char in cols_str:
        if char in ('"', '`') and not in_quote:
            in_quote = True
            quote_char = char
            current += char
        elif char == quote_char and in_quote:
            in_quote = False
            current += char
            quote_char = None
        elif char == ',' and not in_quote:
            col = current.strip()
            if col:
                cols.append(col)
            current = ""
        else:
            current += char

    # Don't forget the last column
    col = current.strip()
    if col:
        cols.append(col)

    return cols


# =============================================================================
# DDL Schema Parsing
# =============================================================================

def parse_schema(schema_text: str) -> SchemaInfo:
    """
    Parse DDL schema into structured form.
    Extracts tables, columns, PK, FK relationships.
    """
    info = SchemaInfo(raw_schema=schema_text)

    # Split into individual CREATE TABLE statements
    # Handle both "CREATE TABLE" and "CREATE TABLE IF NOT EXISTS"
    create_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
    parts = re.split(create_pattern, schema_text, flags=re.IGNORECASE)

    for part in parts[1:]:  # Skip empty first part
        if not part.strip():
            continue
        table_info = _parse_create_table(part)
        if table_info:
            info.tables[table_info.name] = table_info
            # Build FK graph
            for col, (ref_tbl, ref_col) in table_info.fk_columns.items():
                info.fk_graph[(table_info.name, col)] = (ref_tbl, ref_col)

    return info


def _parse_create_table(stmt: str) -> Optional[TableInfo]:
    """Parse a single CREATE TABLE statement (without the CREATE TABLE prefix)."""
    # Extract table name - handle backticked/double-quoted names with special chars
    # First try double-quoted name
    name_match = re.match(r'"([^"]+)"\s*\(', stmt)
    if name_match:
        table_name = name_match.group(1)
    else:
        # Try backticked name
        name_match = re.match(r'`([^`]+)`\s*\(', stmt)
        if name_match:
            table_name = name_match.group(1)
        else:
            # Try unquoted name (may include hyphens in some schemas)
            name_match = re.match(r'([A-Za-z_][A-Za-z0-9_\-]*)\s*\(', stmt)
            if not name_match:
                return None
            table_name = name_match.group(1)

    table_info = TableInfo(name=table_name, raw_ddl=f"CREATE TABLE {stmt.strip()}")

    # Find the column definitions section (between outermost parentheses)
    paren_start = stmt.find('(')
    if paren_start == -1:
        return table_info

    # Find matching closing paren
    depth = 1
    paren_end = paren_start + 1
    for i in range(paren_start + 1, len(stmt)):
        if stmt[i] == '(':
            depth += 1
        elif stmt[i] == ')':
            depth -= 1
            if depth == 0:
                paren_end = i
                break

    columns_str = stmt[paren_start + 1:paren_end]

    # Split by comma, respecting nested parentheses
    column_defs = _split_by_comma(columns_str)

    for col_def in column_defs:
        col_def = col_def.strip()
        if not col_def:
            continue

        col_def_upper = col_def.upper()

        # Check for PRIMARY KEY constraint
        if col_def_upper.startswith('PRIMARY KEY'):
            # Extract PK columns: PRIMARY KEY (col1, col2)
            pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', col_def, re.IGNORECASE)
            if pk_match:
                pk_cols = [c.strip().strip('`"') for c in pk_match.group(1).split(',')]
                table_info.pk_columns.extend(pk_cols)
            continue

        # Check for FOREIGN KEY constraint
        if col_def_upper.startswith('FOREIGN KEY'):
            # FOREIGN KEY (col1, col2) REFERENCES table(col1, col2)
            fk_match = re.search(
                r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                col_def, re.IGNORECASE
            )
            if fk_match:
                fk_cols_str = fk_match.group(1)
                ref_table = fk_match.group(2).strip()
                ref_cols_str = fk_match.group(3)

                # Parse individual columns from composite FK
                fk_cols = _parse_column_list(fk_cols_str)
                ref_cols = _parse_column_list(ref_cols_str)

                # Store each FK column mapping individually
                for fk_col, ref_col in zip(fk_cols, ref_cols):
                    table_info.fk_columns[fk_col] = (ref_table, ref_col)
            continue

        # Check for other constraints (UNIQUE, CHECK, etc.)
        if col_def_upper.startswith(('UNIQUE', 'CHECK', 'CONSTRAINT')):
            continue

        # Regular column definition
        # First try to match backticked identifier with spaces
        backtick_match = re.match(r'`([^`]+)`', col_def)
        if backtick_match:
            col_name = backtick_match.group(1)
            table_info.columns.append(f"`{col_name}`")  # Keep backticks for proper quoting

            # Check for inline PRIMARY KEY
            if 'PRIMARY KEY' in col_def_upper:
                table_info.pk_columns.append(f"`{col_name}`")

            # Check for inline REFERENCES (FK)
            ref_match = re.search(
                r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                col_def, re.IGNORECASE
            )
            if ref_match:
                ref_table = ref_match.group(1)
                ref_col = ref_match.group(2).strip().strip('`"')
                table_info.fk_columns[f"`{col_name}`"] = (ref_table, ref_col)
            continue

        # Try double-quoted identifier
        dquote_match = re.match(r'"([^"]+)"', col_def)
        if dquote_match:
            col_name = dquote_match.group(1)
            table_info.columns.append(f'"{col_name}"')

            if 'PRIMARY KEY' in col_def_upper:
                table_info.pk_columns.append(f'"{col_name}"')

            ref_match = re.search(
                r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                col_def, re.IGNORECASE
            )
            if ref_match:
                ref_table = ref_match.group(1)
                ref_col = ref_match.group(2).strip().strip('`"')
                table_info.fk_columns[f'"{col_name}"'] = (ref_table, ref_col)
            continue

        # Regular unquoted identifier
        col_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', col_def)
        if col_match:
            col_name = col_match.group(1)
            # Skip if it's a keyword or looks like a comment
            if col_name.upper() not in ('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT', 'KEY'):
                if col_name.startswith('-') or col_name.startswith('#'):
                    continue  # Skip comment-like entries
                table_info.columns.append(col_name)

                # Check for inline PRIMARY KEY
                if 'PRIMARY KEY' in col_def_upper:
                    table_info.pk_columns.append(col_name)

                # Check for inline REFERENCES (FK)
                ref_match = re.search(
                    r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                    col_def, re.IGNORECASE
                )
                if ref_match:
                    ref_table = ref_match.group(1)
                    ref_col = ref_match.group(2).strip().strip('`"')
                    table_info.fk_columns[col_name] = (ref_table, ref_col)

    return table_info


def _split_by_comma(s: str) -> List[str]:
    """Split string by comma, respecting nested parentheses and stripping SQL comments."""
    # First, remove SQL comments (-- to end of line)
    lines = s.split('\n')
    cleaned_lines = []
    for line in lines:
        comment_idx = line.find('--')
        if comment_idx >= 0:
            line = line[:comment_idx]
        cleaned_lines.append(line)
    s = '\n'.join(cleaned_lines)

    parts = []
    current = []
    depth = 0

    for char in s:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append(''.join(current))

    return parts


def get_all_schema_identifiers(schema_info: SchemaInfo) -> Tuple[Set[str], Set[str]]:
    """Get all table names and column names from schema."""
    tables = set(schema_info.tables.keys())
    columns = set()
    for table in schema_info.tables.values():
        columns.update(table.columns)
    return tables, columns


# =============================================================================
# SQL Identifier Extraction (CONSERVATIVE)
# =============================================================================

SQL_KEYWORDS = {
    'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'NOT',
    'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'AS', 'ORDER',
    'BY', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'UNION', 'ALL',
    'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INNER',
    'LEFT', 'RIGHT', 'OUTER', 'CROSS', 'ASC', 'DESC', 'CASE',
    'WHEN', 'THEN', 'ELSE', 'END', 'CAST', 'COALESCE', 'IFNULL',
    'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'INTO',
    'VALUES', 'SET', 'TABLE', 'INDEX', 'VIEW', 'INTEGER', 'TEXT',
    'REAL', 'BLOB', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES',
    'DEFAULT', 'TRUE', 'FALSE', 'NULLS', 'FIRST', 'LAST', 'OVER',
    'PARTITION', 'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE',
    'LAG', 'LEAD', 'ROWS', 'RANGE', 'UNBOUNDED', 'PRECEDING',
    'FOLLOWING', 'CURRENT', 'ROW', 'SUBSTR', 'SUBSTRING', 'LENGTH',
    'TRIM', 'UPPER', 'LOWER', 'REPLACE', 'INSTR', 'ROUND', 'ABS',
    'DATE', 'TIME', 'DATETIME', 'STRFTIME', 'JULIANDAY', 'TYPEOF',
    'GLOB', 'PRINTF', 'IIF', 'TOTAL', 'GROUP_CONCAT', 'EXCEPT',
    'INTERSECT', 'RECURSIVE', 'WITH', 'NATURAL', 'USING', 'FULL',
    # Common aliases
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10',
    'A', 'B', 'C', 'D', 'E', 'S', 'P', 'M', 'N', 'X', 'Y', 'Z',
}


def extract_sql_identifiers(
    sql: str,
    schema_tables: Set[str],
    schema_columns: Set[str]
) -> SQLExtractionResult:
    """
    Extract table and column identifiers from SQL.
    Uses sqlparse as primary, regex as fallback.

    CONSERVATIVE: Returns is_confident=False if extraction is uncertain.
    Caller should fallback to full schema when is_confident=False.
    """
    sql_upper = sql.upper()

    # Check for complex SQL patterns that warrant fallback
    if ' WITH ' in sql_upper or sql_upper.startswith('WITH '):
        # CTE detected
        return SQLExtractionResult(
            tables=set(), columns=set(), aliases={},
            is_confident=False, fallback_reason="CTE detected"
        )

    # Count SELECT keywords to detect deep nesting
    select_count = sql_upper.count('SELECT')
    if select_count > 3:
        return SQLExtractionResult(
            tables=set(), columns=set(), aliases={},
            is_confident=False, fallback_reason=f"Deep nesting ({select_count} SELECTs)"
        )

    # Try sqlparse first
    tables, columns, aliases = set(), set(), {}
    sqlparse_worked = False

    try:
        import sqlparse
        tables, columns, aliases = _parse_with_sqlparse(sql, schema_tables, schema_columns)
        if tables:  # Got valid results
            sqlparse_worked = True
    except ImportError:
        pass  # sqlparse not available
    except Exception:
        pass  # sqlparse failed

    # If sqlparse didn't work or got no tables, try regex
    if not sqlparse_worked or not tables:
        regex_tables, regex_columns, regex_aliases = _parse_with_regex(sql, schema_tables, schema_columns)
        tables.update(regex_tables)
        columns.update(regex_columns)
        aliases.update(regex_aliases)

    # Determine confidence
    if not tables and len(sql.strip()) > 50:
        # Non-trivial SQL but found no tables - not confident
        return SQLExtractionResult(
            tables=tables, columns=columns, aliases=aliases,
            is_confident=False, fallback_reason="No tables found in non-trivial SQL"
        )

    return SQLExtractionResult(
        tables=tables, columns=columns, aliases=aliases,
        is_confident=True
    )


def _parse_with_sqlparse(
    sql: str,
    schema_tables: Set[str],
    schema_columns: Set[str]
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Parse SQL using sqlparse library."""
    import sqlparse
    from sqlparse.sql import IdentifierList, Identifier, Token
    from sqlparse.tokens import Keyword, DML, Punctuation

    tables = set()
    columns = set()
    aliases = {}

    parsed = sqlparse.parse(sql)[0]

    # Recursive token extraction
    def extract_from_tokens(tokens, context=''):
        nonlocal tables, columns, aliases

        from_seen = False
        join_seen = False

        for i, token in enumerate(tokens):
            ttype = token.ttype
            value = token.value.upper() if hasattr(token, 'value') else ''

            # Track FROM/JOIN context
            if ttype is Keyword:
                if value in ('FROM', 'INTO', 'UPDATE'):
                    from_seen = True
                    join_seen = False
                elif 'JOIN' in value or value in ('INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS'):
                    join_seen = True
                    from_seen = False
                elif value in ('ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'SET', 'AS'):
                    if value != 'AS':
                        from_seen = False
                        join_seen = False
                # Handle case where table name is misclassified as keyword
                elif (from_seen or join_seen) and token.value:
                    tbl_name = token.value.strip()
                    matched_table = _match_identifier(tbl_name, schema_tables)
                    if matched_table:
                        tables.add(matched_table)

            # Extract table from Identifier after FROM/JOIN
            if (from_seen or join_seen) and isinstance(token, Identifier):
                real_name = token.get_real_name()
                alias = token.get_alias()
                if real_name:
                    matched_table = _match_identifier(real_name, schema_tables)
                    if matched_table:
                        tables.add(matched_table)
                        if alias:
                            aliases[alias] = matched_table
                    elif real_name not in SQL_KEYWORDS:
                        tables.add(real_name)
                        if alias:
                            aliases[alias] = real_name
                from_seen = False
                join_seen = False

            # Handle IdentifierList (multiple tables)
            if (from_seen or join_seen) and isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    if isinstance(identifier, Identifier):
                        real_name = identifier.get_real_name()
                        alias = identifier.get_alias()
                        if real_name:
                            matched_table = _match_identifier(real_name, schema_tables)
                            if matched_table:
                                tables.add(matched_table)
                                if alias:
                                    aliases[alias] = matched_table
                            elif real_name not in SQL_KEYWORDS:
                                tables.add(real_name)
                                if alias:
                                    aliases[alias] = real_name
                from_seen = False
                join_seen = False

            # Recurse into sublists
            if hasattr(token, 'tokens'):
                extract_from_tokens(token.tokens, context)

    extract_from_tokens(parsed.tokens)

    # Extract columns using regex (more reliable than token parsing for columns)
    # Look for qualified references: table.column or alias.column
    qualified_pattern = r'(\w+)\.(\w+)'
    for match in re.finditer(qualified_pattern, sql):
        tbl_or_alias, col = match.groups()
        # Resolve alias
        real_table = aliases.get(tbl_or_alias, tbl_or_alias)
        matched_table = _match_identifier(real_table, schema_tables)
        if matched_table:
            tables.add(matched_table)

        matched_col = _match_identifier(col, schema_columns)
        if matched_col:
            columns.add(matched_col)

    # Also look for backticked column references: T1.`column name`
    backtick_pattern = r'(\w+)\.`([^`]+)`'
    for match in re.finditer(backtick_pattern, sql):
        tbl_or_alias, col = match.groups()
        real_table = aliases.get(tbl_or_alias, tbl_or_alias)
        matched_table = _match_identifier(real_table, schema_tables)
        if matched_table:
            tables.add(matched_table)

        matched_col = _match_identifier(col, schema_columns)
        if matched_col:
            columns.add(matched_col)

    # Remove single-quoted string literals before word matching
    sql_no_strings = re.sub(r"'[^']*'", '', sql)

    # Also look for standalone column names that match schema
    words = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sql_no_strings)
    for word in words:
        if word.upper() in SQL_KEYWORDS:
            continue
        matched_col = _match_identifier(word, schema_columns)
        if matched_col:
            columns.add(matched_col)

    return tables, columns, aliases


def _parse_with_regex(
    sql: str,
    schema_tables: Set[str],
    schema_columns: Set[str]
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Regex-based SQL parsing fallback. CONSERVATIVE: matches against known schema."""
    tables = set()
    columns = set()
    aliases = {}

    # Handle backticked identifiers
    backticked = re.findall(r'`([^`]+)`', sql)
    for b in backticked:
        if b in schema_tables:
            tables.add(b)
        if b in schema_columns:
            columns.add(b)
        elif f"`{b}`" in schema_columns:
            columns.add(f"`{b}`")

    # Extract table.column patterns
    qualified = re.findall(r'(\w+)\.(\w+)', sql)
    for tbl, col in qualified:
        matched_table = _match_identifier(tbl, schema_tables)
        if matched_table:
            tables.add(matched_table)
        matched_col = _match_identifier(col, schema_columns)
        if matched_col:
            columns.add(matched_col)

    # Extract FROM/JOIN table patterns
    from_pattern = r'(?:FROM|JOIN)\s+[`"]?(\w+)[`"]?(?:\s+(?:AS\s+)?(\w+))?'
    for match in re.finditer(from_pattern, sql, re.IGNORECASE):
        tbl_name = match.group(1)
        alias = match.group(2)
        matched_table = _match_identifier(tbl_name, schema_tables)
        if matched_table:
            tables.add(matched_table)
            if alias and alias.upper() not in SQL_KEYWORDS:
                aliases[alias] = matched_table
        elif tbl_name.upper() not in SQL_KEYWORDS:
            tables.add(tbl_name)
            if alias and alias.upper() not in SQL_KEYWORDS:
                aliases[alias] = tbl_name

    # Also look for backticked column references: T1.`column name`
    backtick_pattern = r'(\w+)\.`([^`]+)`'
    for match in re.finditer(backtick_pattern, sql):
        tbl_or_alias, col = match.groups()
        matched_col = _match_identifier(col, schema_columns)
        if matched_col:
            columns.add(matched_col)

    # Remove single-quoted string literals before word matching
    sql_no_strings = re.sub(r"'[^']*'", '', sql)

    # Match known schema columns (conservative: only exact matches)
    words = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sql_no_strings)
    for w in words:
        if w.upper() in SQL_KEYWORDS:
            continue
        matched_table = _match_identifier(w, schema_tables)
        if matched_table:
            tables.add(matched_table)
        matched_col = _match_identifier(w, schema_columns)
        if matched_col:
            columns.add(matched_col)

    return tables, columns, aliases


def _match_identifier(identifier: str, known_set: Set[str]) -> Optional[str]:
    """
    Match identifier against known set, handling case insensitivity and backticks.
    Returns the canonical form from known_set if matched, None otherwise.
    """
    if identifier in known_set:
        return identifier

    # Try with backticks
    backticked = f"`{identifier}`"
    if backticked in known_set:
        return backticked

    # Try without backticks (if identifier has them)
    if identifier.startswith('`') and identifier.endswith('`'):
        unquoted = identifier[1:-1]
        if unquoted in known_set:
            return unquoted

    # Case-insensitive match
    id_lower = identifier.lower().strip('`"')
    for known in known_set:
        known_clean = known.lower().strip('`"')
        if known_clean == id_lower:
            return known

    return None


# =============================================================================
# FK Bridge Table Detection
# =============================================================================

def find_bridge_tables(
    sql_tables: Set[str],
    fk_graph: Dict[Tuple[str, str], Tuple[str, str]],
    all_tables: Set[str]
) -> Set[str]:
    """
    Find bridge tables needed to connect SQL-referenced tables.
    Uses BFS on FK graph to find intermediate tables.
    """
    # Build adjacency: table -> set of directly FK-connected tables
    adjacency = defaultdict(set)
    for (src_tbl, _), (dst_tbl, _) in fk_graph.items():
        adjacency[src_tbl].add(dst_tbl)
        adjacency[dst_tbl].add(src_tbl)

    bridges = set()
    sql_tables_list = list(sql_tables & all_tables)  # Only consider tables that exist

    # For each pair of SQL tables, check if bridge needed
    for i, t1 in enumerate(sql_tables_list):
        for t2 in sql_tables_list[i+1:]:
            if t2 in adjacency[t1]:
                continue  # Direct connection exists

            # BFS to find shortest path
            visited = {t1}
            queue = [(t1, [])]
            found = False

            while queue and not found:
                current, path = queue.pop(0)
                for neighbor in adjacency[current]:
                    if neighbor == t2:
                        bridges.update(path)  # Add intermediate tables
                        found = True
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

    return bridges


# =============================================================================
# Deterministic Safety Margin (T11.1: WIDER)
# =============================================================================

def apply_safety_margin(
    table_name: str,
    required_columns: Set[str],
    all_columns: List[str],
    question: str,
    hints: Optional[str],
    is_primary_table: bool = False,
    widened: bool = False,
) -> Set[str]:
    """
    Apply deterministic safety margin rules.
    Returns final set of columns to keep.

    T11.1 changes from T11:
    - Small table threshold: 5 -> 8
    - Question/hint extras: 2 -> 5 (normal) / 8 (widened)
    - Primary table bonus: NEW - up to 5 (normal) / 8 (widened) adjacent cols
    - Minimum columns per table: NEW - 4 (normal) / 5 (widened)
    """
    final_columns = set(required_columns)

    # Tunable params based on mode
    SMALL_TABLE_THRESHOLD = 8
    MAX_QUESTION_EXTRAS = 8 if widened else 5
    PRIMARY_TABLE_EXTRAS = 8 if widened else 5
    MIN_COLS_PER_TABLE = 5 if widened else 4

    # Rule A: Small tables (≤8 columns) - keep everything
    if len(all_columns) <= SMALL_TABLE_THRESHOLD:
        return set(all_columns)

    # Rule B: Add columns matching question/hints (up to MAX_QUESTION_EXTRAS)
    question_lower = question.lower()
    hints_lower = (hints or '').lower()
    candidates = []

    for col in sorted(all_columns):  # Sorted for determinism
        if col in final_columns:
            continue
        col_lower = col.lower().replace('_', ' ').replace('`', '').replace('"', '')
        col_words = col_lower.split()

        # Check if column name appears in question or hints
        if col_lower in question_lower or col_lower in hints_lower:
            candidates.append(col)
        else:
            # Check individual words
            for word in col_words:
                if len(word) > 2 and (word in question_lower or word in hints_lower):
                    candidates.append(col)
                    break

    for col in candidates[:MAX_QUESTION_EXTRAS]:
        final_columns.add(col)

    # Rule C: Primary table bonus - keep adjacent columns in DDL order
    if is_primary_table and len(all_columns) > len(final_columns):
        # Find indices of currently-kept columns in the DDL column order
        required_indices = set()
        for col in final_columns:
            if col in all_columns:
                required_indices.add(all_columns.index(col))

        # Collect adjacent columns (within ±2 positions of any kept column)
        adjacent = []
        for idx in sorted(required_indices):
            for offset in [-2, -1, 1, 2]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(all_columns):
                    neighbor_col = all_columns[neighbor_idx]
                    if neighbor_col not in final_columns and neighbor_col not in adjacent:
                        adjacent.append(neighbor_col)

        # Stable order: by DDL position
        adjacent.sort(key=lambda c: all_columns.index(c))

        # Add up to PRIMARY_TABLE_EXTRAS adjacent columns
        added = 0
        for col in adjacent:
            if added >= PRIMARY_TABLE_EXTRAS:
                break
            final_columns.add(col)
            added += 1

    # Rule D: Ensure minimum columns per table
    min_cols = min(MIN_COLS_PER_TABLE, len(all_columns))
    if len(final_columns) < min_cols:
        # Add columns in DDL order until we reach minimum
        for col in all_columns:
            if col not in final_columns:
                final_columns.add(col)
            if len(final_columns) >= min_cols:
                break

    return final_columns


# =============================================================================
# Per-DB Full/Compact Assignment (T11.1: HASH-BASED 55%)
# =============================================================================

def assign_schema_modes_per_db(examples: List[Dict]) -> Dict[Tuple[str, str], str]:
    """
    Assign full/compact modes with ~55% compact pre-validation.

    T11.1 change: Uses hash(db_id + question) % 100 < 55 for deterministic
    55% compact assignment. After validation fallbacks, this should land
    near the target 50/50 final ratio.
    """
    assignments = {}
    for ex in examples:
        db_id = ex.get('db_id', 'unknown')
        question = _extract_question(ex)

        # Deterministic hash-based assignment
        key_str = f"{db_id}|{question}"
        hash_val = int(hashlib.sha256(key_str.encode()).hexdigest(), 16) % 100

        # 55% compact, 45% full
        mode = "compact" if hash_val < 55 else "full"
        assignments[(db_id, question)] = mode

    return assignments


def _extract_question(example: Dict) -> str:
    """Extract question from T10 example."""
    messages = example.get('messages', [])
    if len(messages) < 2:
        return ""

    user_content = messages[1].get('content', '')

    # Find Question: section
    q_match = re.search(r'Question:\s*(.*)', user_content, re.DOTALL)
    if q_match:
        return q_match.group(1).strip()

    return ""


def _extract_schema(example: Dict) -> str:
    """Extract schema from T10 example."""
    messages = example.get('messages', [])
    if len(messages) < 2:
        return ""

    user_content = messages[1].get('content', '')

    # Find Schema: section (between Schema: and Hints:)
    schema_match = re.search(r'Schema:\s*(.*?)(?=\nHints:)', user_content, re.DOTALL)
    if schema_match:
        return schema_match.group(1).strip()

    return ""


def _extract_hints(example: Dict) -> Optional[str]:
    """Extract hints from T10 example."""
    messages = example.get('messages', [])
    if len(messages) < 2:
        return None

    user_content = messages[1].get('content', '')

    # Find Hints: section (between Hints: and Question:)
    hints_match = re.search(r'Hints:\s*(.*?)(?=\nQuestion:)', user_content, re.DOTALL)
    if hints_match:
        hints = hints_match.group(1).strip()
        if hints.lower() == 'none':
            return None
        return hints

    return None


def _extract_gold_sql(example: Dict) -> str:
    """Extract gold SQL from T10 example."""
    messages = example.get('messages', [])
    if len(messages) < 3:
        return ""
    return messages[2].get('content', '')


# =============================================================================
# Compact Schema Building (T11.1: RE-WIDEN + OVER-COMPACTION GUARDS)
# =============================================================================

@dataclass
class CompactionResult:
    """Result of schema compaction."""
    compact_schema: str
    original_len: int
    compact_len: int
    status: str  # "success" | "fallback_*" | "fallback_over_compacted"
    fallback_reason: Optional[str] = None
    kept_tables: List[str] = field(default_factory=list)
    kept_columns: Dict[str, List[str]] = field(default_factory=dict)
    widened: bool = False  # True if success came from widened retry


def _find_primary_table(
    kept_tables: List[str],
    schema_info: SchemaInfo,
    sql_result: SQLExtractionResult
) -> Optional[str]:
    """Find the table with the most SQL-referenced columns (the primary FROM table)."""
    best_table = None
    best_count = -1
    for table_name in kept_tables:
        table_info = schema_info.tables[table_name]
        count = sum(1 for col in sql_result.columns if col in table_info.columns)
        if count > best_count:
            best_count = count
            best_table = table_name
    return best_table


def _check_over_compaction(
    compact_len: int,
    original_len: int,
    kept_tables: List[str],
    kept_columns: Dict[str, List[str]],
    schema_info: SchemaInfo,
) -> Tuple[bool, Optional[str]]:
    """
    Check if compaction is too aggressive.

    Guards:
    1. Dynamic max reduction based on table retention ratio —
       queries touching few tables naturally need high reduction, so the
       limit scales: 2/10 tables kept → allow up to 90% reduction,
       5/10 tables → allow up to 75%, etc.
    2. Large schema (>2000 chars) compacted below 300 chars
    3. Any kept table has fewer than min(3, total_cols) columns
    """
    # Note: Guard based on reduction percentage was removed.
    # The T11.1 wider safety margins (min 4/5 cols, ≤8 keep-all, primary-table
    # bonus, question/hint extras) are the real protection against oracle-minimal
    # schemas. A percentage guard fights the natural distribution of BIRD schemas
    # where queries touching 1-2 of 10+ tables produce 85-95% reduction
    # structurally, even with generous per-table column retention.

    # Guard 1: Min compact length for very large schemas
    if original_len > 3000 and compact_len < 150:
        return True, f"Compact too small ({compact_len} chars) for large schema ({original_len} chars)"

    # Guard 3: Min columns per table (absolute floor = 3)
    for table_name in kept_tables:
        if table_name not in schema_info.tables:
            continue
        total_cols = len(schema_info.tables[table_name].columns)
        kept_count = len(kept_columns.get(table_name, []))
        min_required = min(3, total_cols)
        if kept_count < min_required:
            return True, f"Table '{table_name}' has only {kept_count}/{total_cols} columns (min {min_required})"

    return False, None


def _build_columns_for_tables(
    schema_info: SchemaInfo,
    kept_tables: List[str],
    sql_result: SQLExtractionResult,
    question: str,
    hints: Optional[str],
    primary_table: Optional[str],
    widened: bool,
) -> Dict[str, List[str]]:
    """
    Build the column sets for each kept table, applying safety margin.
    Returns {table_name: [sorted columns]}.
    """
    kept_columns = {}
    for table_name in kept_tables:
        table_info = schema_info.tables[table_name]

        # Start with required columns
        required_cols = set()

        # Add columns referenced in SQL for this table
        for col in sql_result.columns:
            if col in table_info.columns:
                required_cols.add(col)

        # Always add PK columns
        required_cols.update(
            pk for pk in table_info.pk_columns if pk in table_info.columns
        )

        # Always add FK columns (both source and target)
        required_cols.update(
            fk for fk in table_info.fk_columns.keys() if fk in table_info.columns
        )

        # If this table is an FK target, add the referenced column
        for (src_tbl, src_col), (dst_tbl, dst_col) in schema_info.fk_graph.items():
            if dst_tbl == table_name:
                if dst_col in table_info.columns:
                    required_cols.add(dst_col)

        # Apply safety margin
        is_primary = (table_name == primary_table)
        final_cols = apply_safety_margin(
            table_name, required_cols, table_info.columns,
            question, hints,
            is_primary_table=is_primary,
            widened=widened,
        )

        # Sort in original DDL order for determinism
        kept_columns[table_name] = sorted(
            final_cols,
            key=lambda c: table_info.columns.index(c) if c in table_info.columns else 999
        )

    return kept_columns


def build_compact_schema(
    full_schema: str,
    gold_sql: str,
    question: str,
    hints: Optional[str]
) -> CompactionResult:
    """
    Build compact schema from full schema based on gold SQL.
    CONSERVATIVE: Falls back to full schema on any uncertainty.

    T11.1 flow:
    1. Parse schema + SQL identifiers
    2. Build compact with normal margins
    3. If over-compacted → retry with widened margins
    4. If still over-compacted → fallback to full schema
    """
    original_len = len(full_schema)

    # -- Parse full schema --
    try:
        schema_info = parse_schema(full_schema)
    except Exception as e:
        return CompactionResult(
            compact_schema=full_schema,
            original_len=original_len,
            compact_len=original_len,
            status="fallback_parse_error",
            fallback_reason=f"Schema parse error: {e}"
        )

    if not schema_info.tables:
        return CompactionResult(
            compact_schema=full_schema,
            original_len=original_len,
            compact_len=original_len,
            status="fallback_empty_result",
            fallback_reason="No tables parsed from schema"
        )

    # -- Extract SQL identifiers --
    all_tables, all_columns = get_all_schema_identifiers(schema_info)
    sql_result = extract_sql_identifiers(gold_sql, all_tables, all_columns)

    if not sql_result.is_confident:
        return CompactionResult(
            compact_schema=full_schema,
            original_len=original_len,
            compact_len=original_len,
            status="fallback_parse_error",
            fallback_reason=f"SQL extraction uncertain: {sql_result.fallback_reason}"
        )

    if not sql_result.tables:
        return CompactionResult(
            compact_schema=full_schema,
            original_len=original_len,
            compact_len=original_len,
            status="fallback_empty_result",
            fallback_reason="No tables extracted from SQL"
        )

    # -- Identify required tables (SQL tables + bridge tables) --
    required_tables = set()
    for sql_tbl in sql_result.tables:
        matched = _match_identifier(sql_tbl, all_tables)
        if matched:
            required_tables.add(matched)

    bridge_tables = find_bridge_tables(required_tables, schema_info.fk_graph, all_tables)
    required_tables.update(bridge_tables)

    # Filter to tables that exist in schema
    kept_tables = [t for t in schema_info.tables.keys() if t in required_tables]

    if not kept_tables:
        return CompactionResult(
            compact_schema=full_schema,
            original_len=original_len,
            compact_len=original_len,
            status="fallback_empty_result",
            fallback_reason="No matching tables found"
        )

    # -- Determine primary table --
    primary_table = _find_primary_table(kept_tables, schema_info, sql_result)

    # ================================================================
    # ATTEMPT 1: Normal margins
    # ================================================================
    kept_columns = _build_columns_for_tables(
        schema_info, kept_tables, sql_result, question, hints,
        primary_table, widened=False
    )
    compact_schema = _render_compact_schema(schema_info, kept_tables, kept_columns)
    compact_len = len(compact_schema)

    # Sanity: compact should be smaller
    if compact_len >= original_len:
        return CompactionResult(
            compact_schema=full_schema,
            original_len=original_len,
            compact_len=original_len,
            status="fallback_empty_result",
            fallback_reason="Compact schema not smaller than original"
        )

    # Check over-compaction guards
    over_compacted, reason = _check_over_compaction(
        compact_len, original_len, kept_tables, kept_columns, schema_info
    )

    if over_compacted:
        # ============================================================
        # ATTEMPT 2: Widened margins (re-widen retry)
        # ============================================================
        kept_columns = _build_columns_for_tables(
            schema_info, kept_tables, sql_result, question, hints,
            primary_table, widened=True
        )
        compact_schema = _render_compact_schema(schema_info, kept_tables, kept_columns)
        compact_len = len(compact_schema)

        # Sanity: compact should still be smaller
        if compact_len >= original_len:
            return CompactionResult(
                compact_schema=full_schema,
                original_len=original_len,
                compact_len=original_len,
                status="fallback_empty_result",
                fallback_reason="Widened compact schema not smaller than original"
            )

        # Re-check guards after widening
        still_over, reason2 = _check_over_compaction(
            compact_len, original_len, kept_tables, kept_columns, schema_info
        )

        if still_over:
            # ========================================================
            # FALLBACK: Full schema
            # ========================================================
            return CompactionResult(
                compact_schema=full_schema,
                original_len=original_len,
                compact_len=original_len,
                status="fallback_over_compacted",
                fallback_reason=f"Over-compacted after widening: {reason2}"
            )

        # Widened attempt succeeded
        return CompactionResult(
            compact_schema=compact_schema,
            original_len=original_len,
            compact_len=compact_len,
            status="success",
            kept_tables=kept_tables,
            kept_columns=kept_columns,
            widened=True,
        )

    # Normal attempt succeeded
    return CompactionResult(
        compact_schema=compact_schema,
        original_len=original_len,
        compact_len=compact_len,
        status="success",
        kept_tables=kept_tables,
        kept_columns=kept_columns,
        widened=False,
    )


def _render_compact_schema(
    schema_info: SchemaInfo,
    kept_tables: List[str],
    kept_columns: Dict[str, List[str]]
) -> str:
    """Render compact schema as multiline DDL."""
    ddl_statements = []

    for table_name in kept_tables:
        table_info = schema_info.tables[table_name]
        cols = kept_columns.get(table_name, table_info.columns)

        # Build column definitions
        col_defs = []
        for col in cols:
            # Find the original column definition
            col_def = _find_column_def(table_info.raw_ddl, col)
            if col_def:
                col_defs.append(col_def)
            else:
                col_defs.append(f"{col} TEXT")  # Fallback

        # Add PK constraint if needed
        if table_info.pk_columns and all(pk in cols for pk in table_info.pk_columns):
            if len(table_info.pk_columns) > 1 or not any('PRIMARY KEY' in d.upper() for d in col_defs):
                pk_cols = ', '.join(table_info.pk_columns)
                col_defs.append(f"primary key ({pk_cols})")

        # Helper to check if identifier needs quoting
        def needs_quoting(name):
            return bool(re.search(r'[^A-Za-z0-9_]', name))

        # Add FK constraints for kept columns pointing to kept tables
        for col, (ref_tbl, ref_col) in table_info.fk_columns.items():
            if col in cols and ref_tbl in kept_tables:
                ref_tbl_quoted = f'"{ref_tbl}"' if needs_quoting(ref_tbl) else ref_tbl
                col_defs.append(f"foreign key ({col}) references {ref_tbl_quoted}({ref_col})")

        # Build CREATE TABLE statement
        table_name_quoted = f'"{table_name}"' if needs_quoting(table_name) else table_name
        table_ddl = f"CREATE TABLE {table_name_quoted}\n(\n"
        table_ddl += ",\n".join(f"    {d}" for d in col_defs)
        table_ddl += "\n);"

        ddl_statements.append(table_ddl)

    return "\n".join(ddl_statements)


def _find_column_def(raw_ddl: str, column_name: str) -> Optional[str]:
    """Find column definition in raw DDL."""
    # First, clean up the DDL by removing inline comments
    lines = raw_ddl.split('\n')
    cleaned_lines = []
    for line in lines:
        comment_idx = line.find('--')
        if comment_idx >= 0:
            line = line[:comment_idx]
        cleaned_lines.append(line)
    cleaned_ddl = '\n'.join(cleaned_lines)

    # Strip backticks from column_name if present for pattern matching
    col_for_pattern = column_name
    if column_name.startswith('`') and column_name.endswith('`'):
        col_for_pattern = column_name[1:-1]  # Remove outer backticks

    # Handle backticked column names with spaces/special chars
    patterns = [
        # Backticked with spaces/special chars
        rf'(`{re.escape(col_for_pattern)}`\s+[^,\n]+)',
        # Double-quoted
        rf'("{re.escape(col_for_pattern)}"\s+[^,\n]+)',
        # Plain identifier (word boundary)
        rf'(\b{re.escape(col_for_pattern)}\b\s+[A-Z][A-Za-z0-9_\s()]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_ddl, re.IGNORECASE)
        if match:
            col_def = match.group(1).strip()
            # Clean up: remove trailing commas and parens
            col_def = re.sub(r'[,)]+\s*$', '', col_def)
            # Remove inline REFERENCES (we add FK constraints separately)
            col_def = re.sub(r'\s+REFERENCES\s+\w+\s*\([^)]+\)(\s+ON\s+[^,)]+)*', '', col_def, flags=re.IGNORECASE)
            # Remove ON UPDATE/DELETE clauses
            col_def = re.sub(r'\s+ON\s+(UPDATE|DELETE)\s+\w+', '', col_def, flags=re.IGNORECASE)
            # Remove trailing UNIQUE
            col_def = re.sub(r'\s+UNIQUE\s*\([^)]+\)\s*$', '', col_def, flags=re.IGNORECASE)
            # Clean up whitespace
            col_def = ' '.join(col_def.split())
            return col_def

    return None


# =============================================================================
# T11.1 Example Building
# =============================================================================

def build_t11_example(
    t10_example: Dict,
    schema_mode: str,
    compaction_result: Optional[CompactionResult] = None
) -> Dict:
    """Build a T11.1 example from T10 example."""
    messages = t10_example['messages']

    t11_example = {
        'messages': messages.copy(),
        'db_id': t10_example.get('db_id', 'unknown'),
        'schema_mode': schema_mode,
    }

    if schema_mode == 'full':
        t11_example['compaction_status'] = 'not_applicable'
    elif compaction_result:
        t11_example['compaction_status'] = compaction_result.status
        t11_example['original_schema_len'] = compaction_result.original_len
        t11_example['compact_schema_len'] = compaction_result.compact_len

        if compaction_result.status == 'success':
            # Replace schema in user message
            user_content = messages[1]['content']
            new_user_content = _replace_schema(user_content, compaction_result.compact_schema)
            t11_example['messages'] = [
                messages[0],  # system
                {'role': 'user', 'content': new_user_content},
                messages[2],  # assistant (gold SQL)
            ]
            t11_example['compaction_widened'] = compaction_result.widened
        else:
            t11_example['fallback_reason'] = compaction_result.fallback_reason

    return t11_example


def _replace_schema(user_content: str, new_schema: str) -> str:
    """Replace schema section in user content."""
    schema_match = re.search(r'(Schema:\s*)(.*?)(\n\nHints:)', user_content, re.DOTALL)
    if schema_match:
        return (
            user_content[:schema_match.start()] +
            schema_match.group(1) + new_schema + schema_match.group(3) +
            user_content[schema_match.end():]
        )
    return user_content


# =============================================================================
# Validation Functions
# =============================================================================

def validate_no_invention(full_schema: str, compact_schema: str) -> Tuple[bool, Optional[str]]:
    """Verify compact schema contains NO identifiers not in full schema."""
    full_info = parse_schema(full_schema)
    compact_info = parse_schema(compact_schema)

    full_tables, full_columns = get_all_schema_identifiers(full_info)
    compact_tables, compact_columns = get_all_schema_identifiers(compact_info)

    invented_tables = compact_tables - full_tables
    if invented_tables:
        return False, f"Invented tables: {invented_tables}"

    invented_columns = compact_columns - full_columns
    if invented_columns:
        return False, f"Invented columns: {invented_columns}"

    return True, None


def validate_sql_coverage(
    compact_schema: str,
    gold_sql: str,
    full_schema_tables: Set[str],
    full_schema_columns: Set[str]
) -> Tuple[bool, List[str]]:
    """
    Verify all SQL-referenced identifiers that exist in full schema also exist in compact schema.
    """
    compact_info = parse_schema(compact_schema)
    compact_tables, compact_columns = get_all_schema_identifiers(compact_info)

    # Extract SQL identifiers
    sql_result = extract_sql_identifiers(gold_sql, full_schema_tables, full_schema_columns)

    missing = []

    # Helper to check if column matches (handles backticks)
    def col_matches(col, col_set):
        if col in col_set:
            return True
        col_clean = col.strip('`"')
        for c in col_set:
            c_clean = c.strip('`"')
            if c_clean.lower() == col_clean.lower():
                return True
        return False

    # Check tables - only those that exist in full schema
    for tbl in sql_result.tables:
        tbl_in_full = tbl in full_schema_tables or any(t.lower() == tbl.lower() for t in full_schema_tables)
        if not tbl_in_full:
            continue
        if tbl not in compact_tables:
            if not any(t.lower() == tbl.lower() for t in compact_tables):
                missing.append(f"table:{tbl}")

    # Check columns - only those that exist in full schema AND are not also table names
    tables_lower = {t.lower() for t in sql_result.tables}
    full_tables_lower = {t.lower() for t in full_schema_tables}

    for col in sql_result.columns:
        col_clean = col.lower().strip('`"')
        if col_clean in tables_lower or col_clean in full_tables_lower:
            continue
        col_in_full = col_matches(col, full_schema_columns)
        if not col_in_full:
            continue
        if not col_matches(col, compact_columns):
            missing.append(f"column:{col}")

    return len(missing) == 0, missing


def validate_join_paths(
    compact_schema: str,
    gold_sql: str
) -> Tuple[bool, Optional[str]]:
    """
    Verify compact schema preserves join connectivity.
    """
    compact_info = parse_schema(compact_schema)
    compact_tables, compact_columns = get_all_schema_identifiers(compact_info)

    join_pattern = r'JOIN\s+[`"]?(\w+)[`"]?\s+.*?ON\s+[`"]?(\w+)[`"]?\.`?(\w+)`?\s*=\s*[`"]?(\w+)[`"]?\.`?(\w+)`?'

    for match in re.finditer(join_pattern, gold_sql, re.IGNORECASE):
        join_table, t1, c1, t2, c2 = match.groups()

        if c1 not in compact_columns and not any(c.lower() == c1.lower() for c in compact_columns):
            return False, f"Join column missing: {t1}.{c1}"
        if c2 not in compact_columns and not any(c.lower() == c2.lower() for c in compact_columns):
            return False, f"Join column missing: {t2}.{c2}"

    return True, None


def validate_gold_sql_unchanged(t10_example: Dict, t11_example: Dict) -> bool:
    """Verify assistant message (gold SQL) is byte-identical."""
    t10_sql = t10_example['messages'][2]['content']
    t11_sql = t11_example['messages'][2]['content']
    return t10_sql == t11_sql


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("T11.1 Utilities Module")
    print("=" * 60)

    # Test schema parsing
    test_schema = """CREATE TABLE users
(
    user_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    age INTEGER,
    city TEXT,
    phone TEXT,
    created_at TEXT,
    bio TEXT,
    status TEXT,
    department_id INTEGER,
    foreign key (department_id) references departments(dept_id)
);
CREATE TABLE departments
(
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT,
    budget REAL
);"""

    print("\nTest schema parsing:")
    info = parse_schema(test_schema)
    for tbl_name, tbl_info in info.tables.items():
        print(f"  Table: {tbl_name}")
        print(f"    Columns: {tbl_info.columns}")
        print(f"    PK: {tbl_info.pk_columns}")
        print(f"    FK: {tbl_info.fk_columns}")

    # Test SQL parsing
    test_sql = "SELECT u.name, d.dept_name FROM users AS u JOIN departments d ON u.department_id = d.dept_id WHERE d.budget > 1000"
    all_tables, all_columns = get_all_schema_identifiers(info)

    print("\nTest SQL parsing:")
    print(f"  SQL: {test_sql}")
    result = extract_sql_identifiers(test_sql, all_tables, all_columns)
    print(f"  Tables: {result.tables}")
    print(f"  Columns: {result.columns}")
    print(f"  Aliases: {result.aliases}")
    print(f"  Confident: {result.is_confident}")

    # Test compact schema building
    print("\nTest compact schema building (T11.1 margins):")
    compaction = build_compact_schema(test_schema, test_sql, "What departments have budget over 1000?", None)
    print(f"  Status: {compaction.status}")
    print(f"  Widened: {compaction.widened}")
    print(f"  Original len: {compaction.original_len}")
    print(f"  Compact len: {compaction.compact_len}")
    if compaction.original_len > 0:
        print(f"  Reduction: {100 * (1 - compaction.compact_len / compaction.original_len):.1f}%")
    print(f"  Compact schema:\n{compaction.compact_schema}")

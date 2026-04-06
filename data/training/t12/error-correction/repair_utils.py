#!/usr/bin/env python3
"""
Repair Utilities Module

Shared utilities for the T10 error-correction pipeline:
- SQL parsing (extract tables, columns, aliases)
- Schema loading and caching
- SQL normalization (strip thinking tags, code fences)
- Fuzzy string matching
- Column/table name extraction from error messages
"""

import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Schema Loading and Caching
# =============================================================================

class SchemaInfo:
    """Container for database schema information."""
    
    def __init__(self, db_id: str, db_path: str):
        self.db_id = db_id
        self.db_path = db_path
        self.tables: Dict[str, TableInfo] = {}
        self.foreign_keys: List[ForeignKey] = []
        self._load_schema()
    
    def _load_schema(self):
        """Load schema from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            table_names = [row[0] for row in cursor.fetchall()]
            
            # Get columns for each table
            for table_name in table_names:
                cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                columns = []
                pk_columns = []
                for row in cursor.fetchall():
                    col_name = row[1]
                    col_type = row[2]
                    is_pk = row[5] > 0
                    columns.append(ColumnInfo(col_name, col_type, is_pk, False))
                    if is_pk:
                        pk_columns.append(col_name)
                
                self.tables[table_name] = TableInfo(table_name, columns, pk_columns)
            
            # Get foreign keys
            for table_name in table_names:
                cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
                for row in cursor.fetchall():
                    ref_table = row[2]
                    from_col = row[3]
                    to_col = row[4]
                    self.foreign_keys.append(ForeignKey(table_name, from_col, ref_table, to_col))
                    
                    # Mark column as FK
                    if table_name in self.tables:
                        for col in self.tables[table_name].columns:
                            if col.name == from_col:
                                col.is_fk = True
            
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Failed to load schema from {self.db_path}: {e}")
    
    def get_all_columns(self) -> List[Tuple[str, str]]:
        """Get all (table, column) pairs in schema."""
        result = []
        for table_name, table_info in self.tables.items():
            for col in table_info.columns:
                result.append((table_name, col.name))
        return result
    
    def get_all_column_names(self) -> Set[str]:
        """Get set of all column names (without table prefix)."""
        result = set()
        for table_info in self.tables.values():
            for col in table_info.columns:
                result.add(col.name)
        return result
    
    def column_exists_in_table(self, col_name: str, table_name: str) -> bool:
        """Check if column exists in specific table."""
        if table_name not in self.tables:
            return False
        return any(c.name.lower() == col_name.lower() for c in self.tables[table_name].columns)
    
    def find_column_tables(self, col_name: str) -> List[str]:
        """Find all tables containing a column with this name."""
        tables = []
        for table_name, table_info in self.tables.items():
            for col in table_info.columns:
                if col.name.lower() == col_name.lower():
                    tables.append(table_name)
                    break
        return tables

    def get_neighbor_tables(self, table_names: List[str], hops: int = 1) -> Set[str]:
        """Return FK-neighbor tables within N hops of the provided tables."""
        if hops <= 0:
            return set()

        graph: Dict[str, Set[str]] = {}
        for table_name in self.tables:
            graph[table_name.lower()] = set()
        for fk in self.foreign_keys:
            graph[fk.from_table.lower()].add(fk.to_table.lower())
            graph[fk.to_table.lower()].add(fk.from_table.lower())

        canonical = {table.lower(): table for table in self.tables}
        frontier = {t.lower() for t in table_names if t and t.lower() in canonical}
        visited = set(frontier)

        for _ in range(hops):
            next_frontier = set()
            for table in frontier:
                next_frontier.update(graph.get(table, set()))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return {canonical[t] for t in visited if t in canonical}


class TableInfo:
    """Information about a database table."""
    
    def __init__(self, name: str, columns: List['ColumnInfo'], pk_columns: List[str]):
        self.name = name
        self.columns = columns
        self.pk_columns = pk_columns
    
    def get_column_names(self) -> List[str]:
        return [c.name for c in self.columns]


class ColumnInfo:
    """Information about a table column."""
    
    def __init__(self, name: str, col_type: str, is_pk: bool, is_fk: bool):
        self.name = name
        self.col_type = col_type
        self.is_pk = is_pk
        self.is_fk = is_fk


class ForeignKey:
    """Foreign key relationship."""
    
    def __init__(self, from_table: str, from_col: str, to_table: str, to_col: str):
        self.from_table = from_table
        self.from_col = from_col
        self.to_table = to_table
        self.to_col = to_col
    
    def __repr__(self):
        return f"{self.from_table}.{self.from_col} -> {self.to_table}.{self.to_col}"


class SchemaCache:
    """Cache for loaded schemas."""
    
    def __init__(self, db_dir: str):
        self.db_dir = Path(db_dir)
        self._cache: Dict[str, SchemaInfo] = {}
    
    def get_schema(self, db_id: str) -> Optional[SchemaInfo]:
        """Get schema for database, loading if necessary."""
        if db_id not in self._cache:
            db_path = self._find_database(db_id)
            if db_path:
                self._cache[db_id] = SchemaInfo(db_id, db_path)
            else:
                return None
        return self._cache[db_id]
    
    def _find_database(self, db_id: str) -> Optional[str]:
        """Find SQLite database file for a given db_id."""
        # Primary path: db_dir/db_id/db_id.sqlite
        db_file = self.db_dir / db_id / f"{db_id}.sqlite"
        if db_file.exists():
            return str(db_file)
        
        # Fallback patterns
        for pattern in [f"*/{db_id}.sqlite", f"*/{db_id}.db"]:
            matches = list(self.db_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def get_db_path(self, db_id: str) -> Optional[str]:
        """Get database file path for db_id."""
        return self._find_database(db_id)


# =============================================================================
# SQL Normalization
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
    
    # Handle incomplete thinking (started but not ended)
    if "<think>" in s and "</think>" not in s:
        # Try to extract SQL after the thinking
        match = re.search(r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s', s, re.IGNORECASE)
        if match:
            s = s[match.start():]
        else:
            # No SQL found, return empty
            return ""
    
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


def strip_thinking_tags(text: str) -> str:
    """Remove thinking tags from output, keeping only final content.
    
    Handles:
    - Complete thinking blocks: <think>...</think>
    - Incomplete thinking with SQL after: <think>... SELECT ...
    - Text before incomplete thinking: SELECT ... <think>...
    """
    if not text:
        return ""
    
    # Remove complete thinking blocks
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Handle incomplete thinking (started but not ended)
    if "<think>" in result:
        # Get content before the thinking tag
        before_think = result.split("<think>")[0].strip()
        # Get content after the thinking tag
        after_think = result.split("<think>", 1)[1] if "<think>" in result else ""
        
        # Try to extract SQL patterns from within the incomplete thinking content
        sql_keywords = r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\s'
        match = re.search(sql_keywords, after_think, re.IGNORECASE)
        
        if match:
            # Found SQL after thinking started - recover it
            recovered_sql = after_think[match.start():]
            result = (before_think + " " + recovered_sql).strip()
        else:
            # No SQL found after <think>, keep only what was before
            result = before_think
    
    return result


# =============================================================================
# SQL Parsing
# =============================================================================

def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL query.
    Returns list of table names (without aliases).
    """
    if not sql:
        return []
    
    tables = set()

    # Support both bare identifiers and backticked identifiers with spaces or
    # punctuation. Ignore subqueries in FROM/JOIN positions.
    table_pattern = r'(?:`([^`]+)`|([A-Za-z_][\w$]*))'
    patterns = [
        rf'\bFROM\s+{table_pattern}(?!\s*\.)',
        rf'\b(?:INNER|LEFT|RIGHT|CROSS|FULL|OUTER)?\s*JOIN\s+{table_pattern}(?!\s*\.)',
    ]

    sql_keywords = {'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'ON', 'AS'}

    for pattern in patterns:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            table_name = match.group(1) or match.group(2)
            if not table_name:
                continue
            if table_name.upper() not in sql_keywords:
                tables.add(table_name)
    
    return list(tables)


def extract_aliases_from_sql(sql: str) -> Dict[str, str]:
    """
    Extract alias -> table mappings from SQL.
    Returns dict mapping alias to table name.
    """
    if not sql:
        return {}
    
    aliases = {}
    
    # Match: table AS alias, or table alias (where alias is T1, T2, etc.)
    # Handles: FROM, JOIN, LEFT JOIN, RIGHT JOIN, INNER JOIN, CROSS JOIN, OUTER JOIN
    # Supports: backticks for table names with spaces, subqueries with aliases
    
    # Pattern for table names: either `...` (with spaces allowed) or \w+
    table_pattern = r'(?:`([^`]+)`|(\w+))'
    
    # SQL keywords that should not be treated as aliases
    sql_keywords = {'AS', 'ON', 'WHERE', 'AND', 'OR', 'SELECT', 'FROM', 'JOIN', 
                    'LEFT', 'RIGHT', 'INNER', 'CROSS', 'FULL', 'OUTER', 'GROUP',
                    'ORDER', 'LIMIT', 'HAVING', 'UNION', 'SET', 'VALUES', 'INTO'}
    
    patterns = [
        # FROM table AS alias
        (rf'\bFROM\s+{table_pattern}\s+AS\s+(\w+)', 'table'),
        # [LEFT|RIGHT|INNER|CROSS|FULL|OUTER] JOIN table AS alias
        (rf'\bJOIN\s+{table_pattern}\s+AS\s+(\w+)', 'table'),
        # [LEFT|RIGHT|INNER|CROSS|FULL|OUTER] JOIN table alias ON...
        (rf'\bJOIN\s+{table_pattern}\s+(\w+)\s+ON\b', 'table'),
        # FROM table alias (followed by JOIN, WHERE, GROUP, ORDER, LIMIT, or end-ish)
        (rf'\bFROM\s+{table_pattern}\s+(\w+)(?=\s+(?:INNER|LEFT|RIGHT|CROSS|FULL|OUTER|JOIN|WHERE|GROUP|ORDER|LIMIT|HAVING|UNION|$))', 'table'),
        # FROM table alias, (comma-separated tables)
        (rf'\bFROM\s+{table_pattern}\s+(\w+)\s*,', 'table'),
        # Subquery alias: (...) AS alias
        (r'\)\s+AS\s+(\w+)', 'subquery'),
        # Subquery alias: (...) T1 (common pattern)
        (r'\)\s+(T\d+)\b', 'subquery'),
    ]
    
    for pattern, ptype in patterns:
        matches = re.finditer(pattern, sql, re.IGNORECASE)
        for m in matches:
            groups = m.groups()
            if ptype == 'subquery':
                alias = groups[0]
                if alias.upper() not in sql_keywords:
                    aliases[alias] = '__subquery__'
            else:
                # Table is in group 1 (backticked) or group 2 (unquoted)
                table_name = groups[0] if groups[0] else groups[1]
                alias = groups[2]
                if alias.upper() not in sql_keywords:
                    aliases[alias] = table_name
    
    return aliases


def extract_columns_from_sql(sql: str) -> List[Tuple[Optional[str], str]]:
    """
    Extract (table_or_alias, column) pairs from SQL.
    
    Note: Only extracts qualified columns (table.column or alias.column).
    Unqualified bare column names are not extracted.
    """
    if not sql:
        return []
    
    columns = []
    
    # Match both quoted and unquoted qualified identifiers.
    patterns = [
        r'`([^`]+)`\s*\.\s*`([^`]+)`',
        r'`([^`]+)`\s*\.\s*([A-Za-z_][\w$]*)',
        r'([A-Za-z_][\w$]*)\s*\.\s*`([^`]+)`',
        r'([A-Za-z_][\w$]*)\s*\.\s*([A-Za-z_][\w$]*)',
    ]

    sql_keywords = {'AND', 'OR', 'AS', 'ON'}

    for pattern in patterns:
        for match in re.finditer(pattern, sql):
            table_or_alias = match.group(1).strip('`')
            col_name = match.group(2).strip('`')
            if table_or_alias.upper() in sql_keywords:
                continue
            if col_name.upper() in sql_keywords or col_name.isdigit():
                continue
            pair = (table_or_alias, col_name)
            if pair not in columns:
                columns.append(pair)
    
    return columns


def extract_column_from_error(error_msg: str) -> Optional[str]:
    """
    Extract the column name from an error message like:
    "no such column: T1.MailingStreet"
    "no such column: FRPM_COUNT"
    """
    if not error_msg:
        return None
    
    # Pattern: no such column: [table.]`column with spaces`
    match = re.search(r'no such column:\s*(?:\w+\s*\.\s*)?`([^`]+)`', error_msg, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern: no such column: [table.]column
    match = re.search(r'no such column:\s*(?:\w+\s*\.\s*)?`?([^`\s]+)`?', error_msg, re.IGNORECASE)
    if match:
        return match.group(1).strip('`')
    
    # Pattern: near "column": syntax error (for unquoted special columns)
    match = re.search(r'near\s+"([^"]+)":\s*syntax error', error_msg, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def extract_table_from_error(error_msg: str) -> Optional[str]:
    """
    Extract table name from error message like:
    "no such table: users"
    """
    if not error_msg:
        return None
    
    match = re.search(r'no such table:\s*`([^`]+)`', error_msg, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r'no such table:\s*`?(\w+)`?', error_msg, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def extract_alias_from_error(error_msg: str) -> Optional[Tuple[str, str]]:
    """
    Extract (alias, column) from error like:
    "no such column: T2.MailingStreet"
    Returns (alias, column) or None.
    """
    if not error_msg:
        return None
    
    match = re.search(r'no such column:\s*(\w+)\s*\.\s*`?([^`\s]+)`?', error_msg, re.IGNORECASE)
    if match:
        return (match.group(1), match.group(2).strip('`'))
    
    return None


# =============================================================================
# Fuzzy Matching
# =============================================================================

def fuzzy_match(s1: str, s2: str) -> float:
    """
    Compute fuzzy match score between two strings.
    Returns score between 0.0 and 1.0.
    """
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _dedupe_ranked_candidates(candidates: List[Dict[str, Any]], key_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in sorted(candidates, key=lambda c: (-c["score"], c.get("name", ""), c.get("table", ""))):
        key = tuple(candidate.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def rank_column_candidates(
    col_name: str,
    schema: SchemaInfo,
    preferred_tables: Optional[List[str]] = None,
    secondary_tables: Optional[List[str]] = None,
    threshold: float = 0.6,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rank candidate columns with local-first search scopes."""
    if not col_name or not schema:
        return []

    canonical = {table.lower(): table for table in schema.tables}
    preferred = [canonical[t.lower()] for t in (preferred_tables or []) if t and t.lower() in canonical]
    secondary = [canonical[t.lower()] for t in (secondary_tables or []) if t and t.lower() in canonical]
    global_tables = list(schema.tables.keys())

    scoped_tables = [
        ("same_table", preferred),
        ("local_subgraph", [t for t in secondary if t not in preferred]),
        ("global_schema", [t for t in global_tables if t not in set(preferred) | set(secondary)]),
    ]

    candidates: List[Dict[str, Any]] = []
    for scope, tables in scoped_tables:
        for table_name in tables:
            table_info = schema.tables.get(table_name)
            if not table_info:
                continue
            for col in table_info.columns:
                score = fuzzy_match(col_name, col.name)
                if score >= threshold:
                    candidates.append({
                        "name": col.name,
                        "table": table_name,
                        "score": score,
                        "scope": scope,
                    })

    return _dedupe_ranked_candidates(candidates, ("name", "table"))[:top_k]


def rank_table_candidates(
    table_name: str,
    schema: SchemaInfo,
    preferred_tables: Optional[List[str]] = None,
    secondary_tables: Optional[List[str]] = None,
    threshold: float = 0.6,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rank candidate tables with local-first search scopes."""
    if not table_name or not schema:
        return []

    canonical = {table.lower(): table for table in schema.tables}
    preferred = [canonical[t.lower()] for t in (preferred_tables or []) if t and t.lower() in canonical]
    secondary = [canonical[t.lower()] for t in (secondary_tables or []) if t and t.lower() in canonical]
    global_tables = list(schema.tables.keys())

    scoped_tables = [
        ("local_sql", preferred),
        ("local_subgraph", [t for t in secondary if t not in preferred]),
        ("global_schema", [t for t in global_tables if t not in set(preferred) | set(secondary)]),
    ]

    candidates: List[Dict[str, Any]] = []
    for scope, tables in scoped_tables:
        for candidate_name in tables:
            score = fuzzy_match(table_name, candidate_name)
            if score >= threshold:
                candidates.append({
                    "name": candidate_name,
                    "table": candidate_name,
                    "score": score,
                    "scope": scope,
                })

    return _dedupe_ranked_candidates(candidates, ("name",))[:top_k]


def fuzzy_find_column(
    col_name: str,
    schema: SchemaInfo,
    threshold: float = 0.6,
    preferred_tables: Optional[List[str]] = None,
    secondary_tables: Optional[List[str]] = None,
) -> Tuple[Optional[str], float]:
    """Return the best fuzzy column candidate, preferring local tables first."""
    candidates = rank_column_candidates(
        col_name,
        schema,
        preferred_tables=preferred_tables,
        secondary_tables=secondary_tables,
        threshold=threshold,
        top_k=1,
    )
    if candidates:
        return candidates[0]["name"], candidates[0]["score"]
    return None, 0.0


def fuzzy_find_table(
    table_name: str,
    schema: SchemaInfo,
    threshold: float = 0.6,
    preferred_tables: Optional[List[str]] = None,
    secondary_tables: Optional[List[str]] = None,
) -> Tuple[Optional[str], float]:
    """Return the best fuzzy table candidate, preferring local tables first."""
    candidates = rank_table_candidates(
        table_name,
        schema,
        preferred_tables=preferred_tables,
        secondary_tables=secondary_tables,
        threshold=threshold,
        top_k=1,
    )
    if candidates:
        return candidates[0]["name"], candidates[0]["score"]
    return None, 0.0


# =============================================================================
# SQL Validation Helpers
# =============================================================================

def sql_appears_truncated(sql: str) -> bool:
    """
    Check if SQL appears to be truncated/incomplete.
    """
    if not sql:
        return True
    
    s = sql.strip()
    
    # Check for incomplete patterns
    truncation_indicators = [
        # Ends with keyword that expects more
        r'\b(SELECT|FROM|WHERE|AND|OR|JOIN|ON|ORDER BY|GROUP BY|HAVING|LIMIT|SET|VALUES)\s*$',
        # Unbalanced parentheses
        lambda x: x.count('(') != x.count(')'),
        # Unbalanced quotes
        lambda x: x.count("'") % 2 != 0,
        # Very short for a SELECT
        lambda x: x.upper().startswith('SELECT') and len(x) < 20,
    ]
    
    for indicator in truncation_indicators:
        if callable(indicator):
            if indicator(s):
                return True
        else:
            if re.search(indicator, s, re.IGNORECASE):
                return True
    
    return False


def has_unquoted_special_column(sql: str, schema: SchemaInfo) -> bool:
    """
    Check if SQL has columns with special characters that aren't backtick-quoted.
    """
    if not sql or not schema:
        return False
    
    # Get all columns with special characters
    special_cols = set()
    for table_info in schema.tables.values():
        for col in table_info.columns:
            if re.search(r'[\s\(\)\-%/]', col.name):
                special_cols.add(col.name)
    
    if not special_cols:
        return False
    
    # Check if any appear unquoted in SQL
    for col in special_cols:
        # If column appears without backticks, it's a problem
        # Simple heuristic: check if col name words appear individually
        words = col.split()
        if len(words) > 1:
            # Multi-word column - check if first word appears alone
            first_word = words[0]
            if re.search(rf'\b{re.escape(first_word)}\b(?!\s*`)', sql):
                # First word appears but not followed by backtick completion
                if f'`{col}`' not in sql:
                    return True
    
    return False


def compute_sql_diff_ratio(sql1: str, sql2: str) -> float:
    """
    Compute how different two SQL strings are.
    Returns 0.0 for identical, 1.0 for completely different.
    """
    if not sql1 and not sql2:
        return 0.0
    if not sql1 or not sql2:
        return 1.0
    
    # Normalize for comparison
    s1 = re.sub(r'\s+', ' ', sql1.lower().strip())
    s2 = re.sub(r'\s+', ' ', sql2.lower().strip())
    
    similarity = SequenceMatcher(None, s1, s2).ratio()
    return 1.0 - similarity


# =============================================================================
# Execution Helpers
# =============================================================================

def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[bool, Any]:
    """
    Execute SQL and return (success, results_or_error).

    Notes:
    - sqlite3 `timeout` only applies to lock contention, not query runtime.
    - We enforce a wall-clock runtime limit via a progress handler so a single
      pathological query cannot stall the whole repair pipeline.
    """
    if timeout <= 0:
        timeout = 30

    try:
        deadline = time.monotonic() + float(timeout)
        conn = sqlite3.connect(db_path, timeout=float(timeout))
        conn.text_factory = lambda b: b.decode(errors='ignore')
        conn.execute(f"PRAGMA busy_timeout = {int(float(timeout) * 1000)}")

        def _progress_abort() -> int:
            # Non-zero return interrupts the current SQLite operation.
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_progress_abort, 10_000)

        try:
            cursor = conn.execute(sql)
            # Validation only needs successful execution, not full result materialization.
            results = cursor.fetchmany(10)
            return True, results
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg.lower():
            return False, f"query timeout after {timeout}s"
        return False, msg
    except Exception as e:
        return False, str(e)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Repair Utilities Module")
    print("=" * 60)
    
    # Test SQL normalization
    test_sql = """<think>Let me think about this...</think>
    SELECT * FROM users WHERE id = 1"""
    print(f"Normalized SQL: {normalize_sql(test_sql)}")
    
    # Test table extraction
    test_query = "SELECT u.name FROM users u INNER JOIN orders o ON u.id = o.user_id"
    tables = extract_tables_from_sql(test_query)
    print(f"Extracted tables: {tables}")
    
    # Test alias extraction
    aliases = extract_aliases_from_sql(test_query)
    print(f"Extracted aliases: {aliases}")
    
    # Test column extraction
    columns = extract_columns_from_sql(test_query)
    print(f"Extracted columns: {columns}")
    
    # Test error parsing
    error = "no such column: T1.MailingStreet"
    col = extract_column_from_error(error)
    print(f"Column from error: {col}")
    
    # Test fuzzy matching
    score = fuzzy_match("MailingStreet", "MailStreet")
    print(f"Fuzzy match score: {score:.2f}")

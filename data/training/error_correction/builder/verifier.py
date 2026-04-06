#!/usr/bin/env python3
"""
Verifier Module

Validates corrected SQL against reference SQL using:
- Schema validity checks
- Execution validity checks
- Result matching
"""

import hashlib
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import PathConfig


# SQL keywords to ignore when extracting identifiers
SQL_KEYWORDS: Set[str] = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "ON", "AS", "GROUP", "BY", "ORDER",
    "LIMIT", "HAVING", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
    "UNION", "ALL", "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC",
    "IN", "IS", "NOT", "NULL", "LIKE", "BETWEEN", "EXISTS", "WITH", "COUNT", "SUM",
    "AVG", "MIN", "MAX", "CAST", "REAL", "INTEGER", "TEXT", "TRUE", "FALSE",
    "OFFSET", "EXCEPT", "INTERSECT", "VALUES", "INSERT", "UPDATE", "DELETE",
    "CREATE", "DROP", "ALTER", "TABLE", "INDEX", "VIEW", "TRIGGER", "PRIMARY",
    "KEY", "FOREIGN", "REFERENCES", "CONSTRAINT", "DEFAULT", "AUTOINCREMENT",
    "COALESCE", "IFNULL", "NULLIF", "IIF", "LENGTH", "SUBSTR", "SUBSTRING",
    "UPPER", "LOWER", "TRIM", "REPLACE", "INSTR", "PRINTF", "ROUND", "ABS",
    "RANDOM", "DATE", "TIME", "DATETIME", "JULIANDAY", "STRFTIME", "TYPEOF",
    "GLOB", "MATCH", "OVER", "PARTITION", "ROW_NUMBER", "RANK", "DENSE_RANK",
    "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "NTILE",
}


def execute_sql(
    db_path: str,
    sql: str,
    timeout: int = 30,
) -> Tuple[bool, Any]:
    """
    Execute SQL on a SQLite database and return results.
    
    Returns:
        (success, results) where results is the query output or error message
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def results_match(gold_results: Any, pred_results: Any) -> bool:
    """
    Compare two SQL result sets using set-based comparison (order-independent).
    """
    if gold_results is None or pred_results is None:
        return False
    
    def to_set(results: List) -> Set[Tuple]:
        return set(tuple(row) for row in results)
    
    try:
        gold_set = to_set(gold_results)
        pred_set = to_set(pred_results)
        return gold_set == pred_set
    except (TypeError, ValueError):
        # Fallback to list comparison if unhashable
        return sorted(str(r) for r in gold_results) == sorted(str(r) for r in pred_results)


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.
    
    - Lowercase keywords
    - Normalize whitespace
    - Remove comments
    """
    if not sql:
        return ""
    
    # Remove single-line comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Normalize whitespace
    sql = ' '.join(sql.split())
    
    return sql.strip()


def sql_hash(sql: str) -> str:
    """Generate a hash for normalized SQL."""
    normalized = normalize_sql(sql)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from SQL query."""
    if not sql:
        return set()
    
    tables = set()
    
    # Match FROM/JOIN table references
    patterns = [
        r'\bFROM\s+`?(\w+)`?',
        r'\bJOIN\s+`?(\w+)`?',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            table = match.group(1)
            if table.upper() not in SQL_KEYWORDS:
                tables.add(table)
    
    return tables


def extract_columns_from_sql(sql: str) -> Set[str]:
    """Extract column references from SQL query."""
    if not sql:
        return set()
    
    columns = set()
    
    # Match qualified columns (table.column or alias.column)
    qualified_pattern = r'`?(\w+)`?\s*\.\s*`?([^`\s,\)]+)`?'
    for match in re.finditer(qualified_pattern, sql):
        col = match.group(2).strip('`')
        if col.upper() not in SQL_KEYWORDS:
            columns.add(col)
    
    # Match backtick-quoted columns
    backtick_pattern = r'`([^`]+)`'
    for match in re.finditer(backtick_pattern, sql):
        col = match.group(1)
        if col.upper() not in SQL_KEYWORDS and '.' not in col:
            columns.add(col)
    
    return columns


def load_schema_info(db_path: str) -> Dict[str, List[str]]:
    """
    Load schema information from a SQLite database.
    
    Returns:
        Dict mapping table names to list of column names
    """
    schema = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get columns for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            columns = [row[1] for row in cursor.fetchall()]
            schema[table.lower()] = columns
        
        conn.close()
    except Exception:
        pass
    
    return schema


def validate_schema_usage(
    sql: str,
    schema: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """
    Check that SQL only uses tables/columns that exist in schema.
    
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Check tables
    sql_tables = extract_tables_from_sql(sql)
    schema_tables = {t.lower() for t in schema.keys()}
    
    for table in sql_tables:
        if table.lower() not in schema_tables:
            errors.append(f"Unknown table: {table}")
    
    # Note: Column validation is tricky due to aliases and qualified names
    # We do a loose check here
    
    return len(errors) == 0, errors


def check_prose_contamination(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Check if SQL output contains prose contamination.
    
    Returns:
        (is_clean, error_message)
    """
    if not sql:
        return False, "Empty SQL"
    
    stripped = sql.strip()
    lower = stripped.lower()
    
    # Check for common prose indicators
    if lower.startswith("the "):
        return False, "Starts with 'the '"
    if lower.startswith("here "):
        return False, "Starts with 'here '"
    if lower.startswith("this "):
        return False, "Starts with 'this '"
    if lower.startswith("to "):
        return False, "Starts with 'to '"
    if lower.startswith("i "):
        return False, "Starts with 'I '"
    
    # Check for code fences
    if "```" in sql:
        return False, "Contains code fences"
    
    # Check for markdown
    if sql.startswith("#"):
        return False, "Starts with markdown header"
    
    # Check for explanation patterns
    if re.search(r'\bexplanation\b', lower):
        return False, "Contains 'explanation'"
    if re.search(r'\bnote:\b', lower):
        return False, "Contains 'Note:'"
    
    # Check for very short or non-SQL content
    if len(stripped) < 8:
        return False, "SQL too short"
    
    # Check it starts like SQL
    if not re.match(r'^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)', stripped, re.IGNORECASE):
        return False, "Does not start with SQL keyword"
    
    return True, None


def verify_corrected_sql(
    corrected_sql: str,
    reference_sql: str,
    broken_sql: str,
    db_path: str,
    require_different: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that corrected SQL is valid and matches reference.
    
    Args:
        corrected_sql: The proposed corrected SQL
        reference_sql: The known-correct reference SQL
        broken_sql: The original broken SQL
        db_path: Path to the SQLite database
        require_different: Require corrected SQL to differ from broken SQL
    
    Returns:
        (passed, details_dict)
    """
    details = {
        "corrected_executes": False,
        "reference_executes": False,
        "results_match": False,
        "is_different_from_broken": False,
        "prose_clean": False,
        "schema_valid": False,
        "errors": [],
    }
    
    # Check prose contamination
    is_clean, prose_error = check_prose_contamination(corrected_sql)
    details["prose_clean"] = is_clean
    if not is_clean:
        details["errors"].append(f"Prose contamination: {prose_error}")
        return False, details
    
    # Load schema
    schema = load_schema_info(db_path)
    
    # Check schema validity
    schema_valid, schema_errors = validate_schema_usage(corrected_sql, schema)
    details["schema_valid"] = schema_valid
    if not schema_valid:
        details["errors"].extend(schema_errors)
    
    # Execute corrected SQL
    corrected_success, corrected_results = execute_sql(db_path, corrected_sql)
    details["corrected_executes"] = corrected_success
    if not corrected_success:
        details["errors"].append(f"Corrected SQL execution failed: {corrected_results}")
        return False, details
    
    # Execute reference SQL
    reference_success, reference_results = execute_sql(db_path, reference_sql)
    details["reference_executes"] = reference_success
    if not reference_success:
        details["errors"].append(f"Reference SQL execution failed: {reference_results}")
        return False, details
    
    # Compare results
    match = results_match(reference_results, corrected_results)
    details["results_match"] = match
    if not match:
        details["errors"].append("Results do not match reference")
        return False, details
    
    # Check if different from broken
    if require_different:
        corrected_norm = normalize_sql(corrected_sql)
        broken_norm = normalize_sql(broken_sql)
        is_different = corrected_norm != broken_norm
        details["is_different_from_broken"] = is_different
        if not is_different:
            details["errors"].append("Corrected SQL is identical to broken SQL")
            return False, details
    else:
        details["is_different_from_broken"] = True
    
    return True, details


def verify_synthetic_pair(
    parent_sql: str,
    broken_sql: str,
    db_path: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a synthetic corruption pair.
    
    The parent SQL is the corrected SQL for synthetic examples.
    
    Returns:
        (passed, details_dict)
    """
    details = {
        "parent_executes": False,
        "broken_executes": False,
        "are_different": False,
        "errors": [],
    }
    
    # Execute parent SQL (should succeed)
    parent_success, parent_results = execute_sql(db_path, parent_sql)
    details["parent_executes"] = parent_success
    if not parent_success:
        details["errors"].append(f"Parent SQL execution failed: {parent_results}")
        return False, details
    
    # Execute broken SQL (may or may not succeed)
    broken_success, broken_results = execute_sql(db_path, broken_sql)
    details["broken_executes"] = broken_success
    
    # If broken SQL succeeds, results should be different from parent
    if broken_success:
        same_results = results_match(parent_results, broken_results)
        if same_results:
            details["errors"].append("Broken SQL produces same results as parent")
            return False, details
    
    # Check they are syntactically different
    parent_norm = normalize_sql(parent_sql)
    broken_norm = normalize_sql(broken_sql)
    details["are_different"] = parent_norm != broken_norm
    
    if not details["are_different"]:
        details["errors"].append("Parent and broken SQL are identical")
        return False, details
    
    return True, details


class Verifier:
    """Verifier for error-correction examples."""
    
    def __init__(self, paths: PathConfig):
        self.paths = paths
        self._schema_cache: Dict[str, Dict[str, List[str]]] = {}
    
    def get_schema(self, db_id: str) -> Dict[str, List[str]]:
        """Get schema for a database (cached)."""
        if db_id not in self._schema_cache:
            db_path = str(self.paths.database_path(db_id))
            self._schema_cache[db_id] = load_schema_info(db_path)
        return self._schema_cache[db_id]
    
    def verify_real_failure_repair(
        self,
        corrected_sql: str,
        reference_sql: str,
        broken_sql: str,
        db_id: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify a real failure repair example."""
        db_path = str(self.paths.database_path(db_id))
        return verify_corrected_sql(
            corrected_sql=corrected_sql,
            reference_sql=reference_sql,
            broken_sql=broken_sql,
            db_path=db_path,
            require_different=True,
        )
    
    def verify_synthetic(
        self,
        parent_sql: str,
        broken_sql: str,
        db_id: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify a synthetic corruption example."""
        db_path = str(self.paths.database_path(db_id))
        return verify_synthetic_pair(
            parent_sql=parent_sql,
            broken_sql=broken_sql,
            db_path=db_path,
        )
    
    def check_reference_valid(
        self,
        reference_sql: str,
        db_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if reference SQL is valid (executes successfully).
        
        Returns:
            (is_valid, error_message)
        """
        if not reference_sql:
            return False, "Reference SQL is empty"
        
        db_path = str(self.paths.database_path(db_id))
        success, result = execute_sql(db_path, reference_sql)
        
        if not success:
            return False, f"Reference SQL failed: {result}"
        
        return True, None

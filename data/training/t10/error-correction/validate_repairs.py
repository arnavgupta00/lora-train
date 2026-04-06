#!/usr/bin/env python3
"""
Repair Validation Module

Validates repaired SQL for:
1. Schema validity (no invented tables/columns)
2. Execution validity (must execute)
3. Output hygiene (no prose, no repeated patterns)
4. Acceptance decision logic
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from repair_utils import (
    SchemaInfo,
    normalize_sql,
    strip_thinking_tags,
    extract_tables_from_sql,
    extract_columns_from_sql,
    extract_aliases_from_sql,
    compute_sql_diff_ratio,
    execute_sql,
)


SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "ON", "AS", "GROUP", "BY", "ORDER",
    "LIMIT", "HAVING", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
    "UNION", "ALL", "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC",
    "IN", "IS", "NOT", "NULL", "LIKE", "BETWEEN", "EXISTS", "WITH", "COUNT", "SUM",
    "AVG", "MIN", "MAX", "CAST", "REAL", "INTEGER", "TEXT", "TRUE", "FALSE",
}


# =============================================================================
# Schema Validation
# =============================================================================

def extract_unqualified_columns(sql: str) -> List[str]:
    """Heuristically extract bare identifiers that may resolve to schema columns."""
    if not sql:
        return []

    stripped = re.sub(r"'[^']*'", " ", sql)
    stripped = re.sub(r'"[^"]*"', " ", stripped)
    stripped = re.sub(r'`[^`]+`', " ", stripped)
    stripped = re.sub(r'\b[A-Za-z_][\w$]*\s*\.\s*[A-Za-z_][\w$]*\b', " ", stripped)

    aliases = set(extract_aliases_from_sql(sql).keys())
    tables = {t.lower() for t in extract_tables_from_sql(sql)}
    select_aliases = {
        match.group(1).lower()
        for match in re.finditer(r'\bAS\s+([A-Za-z_][\w$]*)\b', sql, re.IGNORECASE)
    }
    bare_identifiers: List[str] = []
    for match in re.finditer(r'\b([A-Za-z_][\w$]*)\b', stripped):
        token = match.group(1)
        upper = token.upper()
        if upper in SQL_KEYWORDS:
            continue
        if token in aliases:
            continue
        if token.lower() in tables or token.lower() in select_aliases:
            continue
        if re.search(rf'\b{re.escape(token)}\s*\(', stripped[match.start():], re.IGNORECASE):
            continue
        bare_identifiers.append(token)

    return list(dict.fromkeys(bare_identifiers))

def validate_schema_usage(
    sql: str,
    schema: SchemaInfo,
) -> Tuple[bool, List[str]]:
    """
    Check that SQL only uses tables/columns that exist in schema.
    
    Note: Only validates qualified columns (e.g., T1.column). Unqualified
    columns are not validated as SQLite resolves them at runtime, and the
    main risk is invented qualified references like T1.FakeColumn.
    
    Args:
        sql: The SQL to validate
        schema: SchemaInfo for the database
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if not sql:
        return False, ["Empty SQL"]
    
    # Check tables
    tables_used = extract_tables_from_sql(sql)
    schema_tables_lower = {t.lower() for t in schema.tables.keys()}
    
    for table in tables_used:
        if table.lower() not in schema_tables_lower:
            errors.append(f"Table '{table}' not in schema")
    
    # Build alias -> table mapping from SQL
    aliases = extract_aliases_from_sql(sql)
    aliases_lower = {k.lower(): v for k, v in aliases.items()}
    
    # Check qualified columns (table.column references)
    # Note: extract_columns_from_sql only returns qualified columns
    columns_used = extract_columns_from_sql(sql)
    
    for table_or_alias, col_name in columns_used:
        # Qualified column: resolve alias to actual table name
        table_name = table_or_alias
        
        # Check if it's an alias and resolve to actual table
        if table_or_alias.lower() in aliases_lower:
            table_name = aliases_lower[table_or_alias.lower()]
        
        # Find table in schema (case-insensitive)
        schema_table = None
        for t in schema.tables.keys():
            if t.lower() == table_name.lower():
                schema_table = t
                break
        
        if schema_table and not schema.column_exists_in_table(col_name, schema_table):
            errors.append(f"Column '{col_name}' not in table '{schema_table}'")

    # Check unqualified columns against the tables actually used in the query.
    bare_columns = extract_unqualified_columns(sql)
    used_tables = []
    for table in tables_used:
        for schema_table in schema.tables.keys():
            if schema_table.lower() == table.lower():
                used_tables.append(schema_table)
                break

    for col_name in bare_columns:
        matching_tables = [table for table in used_tables if schema.column_exists_in_table(col_name, table)]
        if not matching_tables:
            # Bare identifiers may also be SELECT aliases; flag them for caution.
            errors.append(f"Unqualified identifier '{col_name}' does not resolve in used tables")
        elif len(matching_tables) > 1:
            errors.append(
                f"Unqualified identifier '{col_name}' is ambiguous across tables {matching_tables}"
            )
    
    return len(errors) == 0, errors


# =============================================================================
# Output Hygiene
# =============================================================================

def validate_output_hygiene(
    raw_output: str,
) -> Tuple[bool, str, List[str]]:
    """
    Clean output and check for issues.
    
    Args:
        raw_output: Raw model output
    
    Returns:
        (is_valid, cleaned_sql, issues)
    """
    issues = []
    
    if not raw_output:
        return False, "", ["Empty output"]
    
    # Strip thinking tags
    sql = strip_thinking_tags(raw_output)
    
    # Strip markdown code fences
    if "```" in sql:
        match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            issues.append("Contained markdown code fences (stripped)")
    
    # Check for prose before SQL
    sql_start = re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b', sql, re.IGNORECASE)
    if sql_start and sql_start.start() > 20:
        # Significant text before SQL
        issues.append("Contained prose before SQL (stripped)")
        sql = sql[sql_start.start():]
    
    # Normalize SQL
    sql = normalize_sql(sql)
    
    # Check for repeated patterns (degenerate output)
    if has_repeated_patterns(sql):
        issues.append("Contains repeated broken patterns")
        sql = truncate_at_repeat(sql)
    
    # Check for excessive clause count (degenerate explosion)
    clause_count = count_clauses(sql)
    if clause_count > 20:
        issues.append(f"Excessive clause count ({clause_count}) - likely degenerate")
        return False, sql, issues
    
    # Check for incomplete SQL
    if not sql or len(sql) < 10:
        issues.append("SQL too short or empty")
        return False, sql, issues
    
    # Check for SQL keywords
    if not re.search(r'\b(SELECT|INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE):
        issues.append("No SQL statement found")
        return False, sql, issues
    
    return True, sql, issues


def has_repeated_patterns(sql: str) -> bool:
    """Check if SQL has repeated broken patterns."""
    if not sql or len(sql) < 50:
        return False
    
    # Check for repeated substrings
    for length in [20, 30, 40]:
        for i in range(len(sql) - 2 * length):
            pattern = sql[i:i + length]
            rest = sql[i + length:]
            if pattern in rest:
                # Check if it appears multiple times
                count = rest.count(pattern)
                if count >= 2:
                    return True
    
    return False


def truncate_at_repeat(sql: str) -> str:
    """Truncate SQL at first repeated pattern."""
    if not sql or len(sql) < 50:
        return sql
    
    for length in [20, 30, 40]:
        for i in range(len(sql) - 2 * length):
            pattern = sql[i:i + length]
            rest = sql[i + length:]
            if pattern in rest:
                return sql[:i + length].strip()
    
    return sql


def count_clauses(sql: str) -> int:
    """Count SQL clauses (approximate)."""
    if not sql:
        return 0
    
    clause_keywords = [
        'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'JOIN', 'INNER JOIN',
        'LEFT JOIN', 'RIGHT JOIN', 'ON', 'GROUP BY', 'HAVING', 'ORDER BY',
        'LIMIT', 'UNION', 'EXCEPT', 'INTERSECT'
    ]
    
    count = 0
    sql_upper = sql.upper()
    for keyword in clause_keywords:
        count += len(re.findall(rf'\b{keyword}\b', sql_upper))
    
    return count


# =============================================================================
# Execution Validation
# =============================================================================

def validate_execution(
    sql: str,
    db_path: str,
    timeout: int = 30,
) -> Tuple[bool, Any]:
    """
    Execute SQL and return (success, result_or_error).
    """
    return execute_sql(db_path, sql, timeout)


def results_match(gold_results: Any, pred_results: Any) -> bool:
    """Compare SQL result sets using the same set-based semantics as evaluator."""
    if gold_results is None or pred_results is None:
        return False

    try:
        gold_set = set(tuple(row) for row in gold_results)
        pred_set = set(tuple(row) for row in pred_results)
    except Exception:
        return False

    return gold_set == pred_set


def extract_clause(sql: str, clause: str, stop_clauses: List[str]) -> str:
    """Extract a top-level clause body using a lightweight SQL regex."""
    if not sql:
        return ""

    stop_pattern = "|".join(re.escape(stop) for stop in stop_clauses)
    match = re.search(
        rf'\b{re.escape(clause)}\b(.*?)(?=\b(?:{stop_pattern})\b|$)',
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def extract_select_targets(sql: str) -> List[str]:
    """Extract normalized SELECT targets."""
    body = extract_clause(sql, "SELECT", ["FROM"])
    if not body:
        return []

    parts: List[str] = []
    current = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            target = "".join(current).strip()
            if target:
                parts.append(target)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)

    return [re.sub(r"\s+", " ", part.strip().lower()) for part in parts]


def structure_diff_ratio(left: str, right: str) -> float:
    """Simple normalized diff ratio for SQL subclauses."""
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return compute_sql_diff_ratio(left, right)


def validate_structure_preservation(
    original_sql: str,
    repaired_sql: str,
    failure_type: str,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Reject unexpected rewrites for tiny-fix repair classes."""
    original_tables = set(t.lower() for t in extract_tables_from_sql(original_sql))
    repaired_tables = set(t.lower() for t in extract_tables_from_sql(repaired_sql))
    original_join_count = len(re.findall(r'\bJOIN\b', original_sql, re.IGNORECASE))
    repaired_join_count = len(re.findall(r'\bJOIN\b', repaired_sql, re.IGNORECASE))
    original_select = extract_select_targets(original_sql)
    repaired_select = extract_select_targets(repaired_sql)
    original_where = extract_clause(original_sql, "WHERE", ["GROUP BY", "HAVING", "ORDER BY", "LIMIT", "UNION"])
    repaired_where = extract_clause(repaired_sql, "WHERE", ["GROUP BY", "HAVING", "ORDER BY", "LIMIT", "UNION"])

    metrics = {
        "new_tables": sorted(repaired_tables - original_tables),
        "original_join_count": original_join_count,
        "repaired_join_count": repaired_join_count,
        "select_diff_ratio": structure_diff_ratio(", ".join(original_select), ", ".join(repaired_select)),
        "where_diff_ratio": structure_diff_ratio(original_where, repaired_where),
    }

    issues: List[str] = []
    tiny_fix_types = {
        "exact_identifier_error",
        "wrong_table_side_error",
        "alias_error",
    }
    if failure_type not in tiny_fix_types:
        return True, issues, metrics

    if metrics["new_tables"]:
        issues.append(f"Unexpected new tables added: {metrics['new_tables']}")
    if original_join_count != repaired_join_count:
        issues.append(f"Join count changed from {original_join_count} to {repaired_join_count}")
    if metrics["select_diff_ratio"] > 0.35:
        issues.append(f"SELECT targets changed too much ({metrics['select_diff_ratio']:.0%})")
    if original_where and repaired_where and metrics["where_diff_ratio"] > 0.4:
        issues.append(f"WHERE logic changed too much ({metrics['where_diff_ratio']:.0%})")

    return len(issues) == 0, issues, metrics


# =============================================================================
# Acceptance Decision
# =============================================================================

def should_accept_repair(
    original_sql: str,
    repaired_sql: str,
    gold_sql: str,
    original_exec_result: Tuple[bool, Any],
    gold_exec_result: Tuple[bool, Any],
    repair_exec_result: Tuple[bool, Any],
    schema_validation: Tuple[bool, List[str]],
    failure_type: str,
    structure_validation: Tuple[bool, List[str]],
) -> Tuple[bool, str]:
    """
    Decide whether to accept the repair.
    
    Acceptance policy:
    - Accept only when repaired SQL matches gold SQL execution results.
    
    Args:
        original_sql: The original predicted SQL
        repaired_sql: The repaired SQL
        original_exec_result: (success, result) for original SQL
        repair_exec_result: (success, result) for repaired SQL
        schema_validation: (is_valid, errors) from schema validation
        failure_type: The classified failure type
    
    Returns:
        (accept, reason)
    """
    _ = original_exec_result
    repair_succeeded, repair_result = repair_exec_result
    gold_succeeded, gold_result = gold_exec_result
    schema_valid, schema_errors = schema_validation
    structure_valid, structure_issues = structure_validation
    
    # Must be schema-valid
    if not schema_valid:
        return False, f"Schema invalid: {'; '.join(schema_errors)}"

    if not structure_valid:
        return False, f"Structure guard failed: {'; '.join(structure_issues)}"
    
    # Must execute
    if not repair_succeeded:
        return False, f"Execution still failed: {repair_result}"

    # Gold query must execute to score semantic correctness.
    if not gold_succeeded:
        return False, f"Gold SQL execution failed during validation: {gold_result}"

    # Must match gold results exactly using set-based comparison.
    if not results_match(gold_result, repair_result):
        return False, "Execution result mismatch vs gold"
    
    # Repair is identical (shouldn't happen but guard against)
    if repaired_sql.strip().lower() == original_sql.strip().lower():
        return False, "Repair identical to original"
    
    diff_thresholds = {
        "exact_identifier_error": 0.2,
        "wrong_table_side_error": 0.2,
        "alias_error": 0.15,
        "degenerate_or_truncated_sql": 0.4,
        "derived_metric_error": 0.3,
        "filter_value_mapping_error": 0.25,
        "join_backbone_error": 0.3,
        "generic_exec_error": 0.15,
    }

    # SUCCESS: Original failed, repair executes and is schema-valid.
    diff_ratio = compute_sql_diff_ratio(original_sql, repaired_sql)
    hard_limit = 0.5
    if diff_ratio > hard_limit:
        return False, f"rejected: repair changes >50% of SQL (diff_ratio={diff_ratio:.0%})"

    class_limit = diff_thresholds.get(failure_type, 0.2)
    if diff_ratio > class_limit:
        return False, (
            f"rejected: {failure_type} repair exceeds diff threshold "
            f"{class_limit:.0%} (diff_ratio={diff_ratio:.0%})"
        )
    
    return True, "Execution result matches gold"


# =============================================================================
# Full Validation Pipeline
# =============================================================================

def validate_repair(
    original_sql: str,
    gold_sql: str,
    raw_repair_output: str,
    db_path: str,
    schema: SchemaInfo,
    failure_type: str,
    identifier_confidence: Optional[float] = None,
    original_exec_result: Optional[Tuple[bool, Any]] = None,
    gold_exec_result: Optional[Tuple[bool, Any]] = None,
) -> Dict[str, Any]:
    """
    Run full validation pipeline on a repair.
    
    Args:
        original_sql: The original SQL that failed
        raw_repair_output: Raw model output from repair attempt
        db_path: Path to database file
        schema: SchemaInfo for the database
        failure_type: Classified failure type
    
    Returns:
        Dict with:
        - accepted: bool
        - reason: str
        - cleaned_sql: str
        - hygiene_issues: list
        - schema_errors: list
        - original_exec_result: tuple
        - repair_exec_result: tuple
        - diff_ratio: float
        - high_diff_quarantine: bool (True if diff_ratio > 0.5)
    """
    result = {
        "accepted": False,
        "reason": "",
        "cleaned_sql": "",
        "hygiene_issues": [],
        "schema_errors": [],
        "original_exec_result": (False, "not executed"),
        "gold_exec_result": (False, "not executed"),
        "repair_exec_result": (False, "not executed"),
        "diff_ratio": 0.0,
        "high_diff_quarantine": False,
        "quarantine": False,
        "quarantine_reasons": [],
        "structure_issues": [],
        "structure_metrics": {},
        "matches_gold": False,
    }
    
    # Step 1: Clean output
    hygiene_valid, cleaned_sql, hygiene_issues = validate_output_hygiene(raw_repair_output)
    result["cleaned_sql"] = cleaned_sql
    result["hygiene_issues"] = hygiene_issues
    
    if not hygiene_valid:
        result["reason"] = f"Output hygiene failed: {'; '.join(hygiene_issues)}"
        return result
    
    # Step 2: Schema validation
    schema_valid, schema_errors = validate_schema_usage(cleaned_sql, schema)
    result["schema_errors"] = schema_errors
    
    # Step 3: Execute original SQL (or reuse cached result from prior attempts)
    if original_exec_result is None:
        original_exec_result = validate_execution(original_sql, db_path)
    result["original_exec_result"] = original_exec_result

    # Step 3.5: Execute gold SQL (or reuse cached result from prior attempts)
    if gold_exec_result is None:
        gold_exec_result = validate_execution(gold_sql, db_path)
    result["gold_exec_result"] = gold_exec_result
    
    # Step 4: Execute repaired SQL
    repair_exec_result = validate_execution(cleaned_sql, db_path)
    result["repair_exec_result"] = repair_exec_result
    
    # Step 5: Compute diff ratio
    diff_ratio = compute_sql_diff_ratio(original_sql, cleaned_sql)
    result["diff_ratio"] = diff_ratio
    result["high_diff_quarantine"] = diff_ratio > 0.5

    # Step 5.5: Structure preservation checks
    structure_valid, structure_issues, structure_metrics = validate_structure_preservation(
        original_sql, cleaned_sql, failure_type
    )
    result["structure_issues"] = structure_issues
    result["structure_metrics"] = structure_metrics
    if structure_issues:
        result["quarantine_reasons"].append("structure_change")

    if diff_ratio > 0.5:
        result["quarantine_reasons"].append("high_diff")
    if identifier_confidence is not None and identifier_confidence < 0.8:
        result["quarantine_reasons"].append("low_confidence_identifier_match")
    result["quarantine"] = bool(result["quarantine_reasons"])

    # Step 6: Acceptance decision
    gold_ok, gold_rows = result["gold_exec_result"]
    repair_ok, repair_rows = result["repair_exec_result"]
    result["matches_gold"] = bool(gold_ok and repair_ok and results_match(gold_rows, repair_rows))

    accepted, reason = should_accept_repair(
        original_sql=original_sql,
        repaired_sql=cleaned_sql,
        gold_sql=gold_sql,
        original_exec_result=original_exec_result,
        gold_exec_result=gold_exec_result,
        repair_exec_result=repair_exec_result,
        schema_validation=(schema_valid, schema_errors),
        failure_type=failure_type,
        structure_validation=(structure_valid, structure_issues),
    )
    
    result["accepted"] = accepted
    result["reason"] = reason
    
    return result


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Repair Validation Module")
    print("=" * 60)
    print("Use validate_repair() to validate a repair attempt.")

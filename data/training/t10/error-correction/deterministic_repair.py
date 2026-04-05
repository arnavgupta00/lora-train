#!/usr/bin/env python3
"""
Deterministic Repair Module

Fast-path repair stage that runs before LLM repair for safest, smallest fixes:
- Exact identifier typos with high-confidence fuzzy match
- Pure alias swaps / alias qualification
- Missing backticks around special identifiers
- Obvious wrong-table-side references

Each function:
1. Detects the failed identifier/alias
2. Chooses top local schema candidate
3. Patches SQL directly
4. Validates schema
5. Executes SQL
6. Returns success/failure with repaired SQL
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from repair_utils import (
    SchemaInfo,
    apply_identifier_replacement,
    apply_backtick_wrapper,
    validate_and_execute_sql,
    extract_column_from_error,
    extract_table_from_error,
    extract_alias_from_error,
    extract_tables_from_sql,
    extract_aliases_from_sql,
    rank_column_candidates,
    rank_table_candidates,
    get_candidate_scope,
    has_unquoted_special_column,
)


# =============================================================================
# Deterministic Repair Result
# =============================================================================

class DeterministicRepairResult:
    """Result of a deterministic repair attempt."""
    
    def __init__(
        self,
        success: bool,
        method: Optional[str] = None,
        repaired_sql: Optional[str] = None,
        failed_identifier: Optional[str] = None,
        top_candidates: Optional[List[Dict[str, Any]]] = None,
        chosen_candidate: Optional[Dict[str, Any]] = None,
        alias_map: Optional[Dict[str, str]] = None,
        candidate_scope: Optional[str] = None,
        backtick_repair_used: bool = False,
        reason: str = "",
    ):
        self.success = success
        self.method = method  # "exact_identifier" | "alias" | "backtick" | "wrong_table_side"
        self.repaired_sql = repaired_sql
        self.failed_identifier = failed_identifier
        self.top_candidates = top_candidates or []
        self.chosen_candidate = chosen_candidate
        self.alias_map = alias_map
        self.candidate_scope = candidate_scope
        self.backtick_repair_used = backtick_repair_used
        self.reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tried": True,
            "succeeded": self.success,
            "method": self.method,
            "repaired_sql": self.repaired_sql,
            "failed_identifier": self.failed_identifier,
            "top_identifier_candidates": self.top_candidates,
            "chosen_identifier_candidate": self.chosen_candidate,
            "alias_map": self.alias_map,
            "candidate_scope": self.candidate_scope,
            "backtick_repair_used": self.backtick_repair_used,
            "reason": self.reason,
        }


# =============================================================================
# Exact Identifier Fix
# =============================================================================

def try_deterministic_exact_identifier_fix(
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
    db_path: str,
    classification: Any,
) -> DeterministicRepairResult:
    """
    Try deterministic fix for exact identifier errors (column/table name typos).
    
    Strategy:
    1. Use same-table candidate search first
    2. Then local selected-table search
    3. Then local bridge-neighborhood search
    4. Only accept if top candidate has high confidence (>0.85)
    """
    failed_identifier = classification.failed_identifier
    if not failed_identifier:
        return DeterministicRepairResult(
            success=False,
            reason="No failed identifier extracted from classification"
        )
    
    # Check if it's a column or table error
    is_column_error = "no such column" in error.lower()
    is_table_error = "no such table" in error.lower()
    
    if not (is_column_error or is_table_error):
        return DeterministicRepairResult(
            success=False,
            reason="Error is not a simple column/table not found error"
        )
    
    if is_column_error:
        return _fix_column_identifier(
            predicted_sql, error, failed_identifier, schema, db_path, classification
        )
    else:
        return _fix_table_identifier(
            predicted_sql, failed_identifier, schema, db_path, classification
        )


def _fix_column_identifier(
    predicted_sql: str,
    error: str,
    col_name: str,
    schema: SchemaInfo,
    db_path: str,
    classification: Any,
) -> DeterministicRepairResult:
    """Fix a column identifier error deterministically."""
    
    # Get candidates from classification if available
    candidates = classification.identifier_candidates or []
    
    # If no candidates from classification, compute them
    if not candidates:
        sql_tables = extract_tables_from_sql(predicted_sql)
        local_tables = schema.get_neighbor_tables(sql_tables, hops=1)
        
        # Extract alias info if available
        alias_info = extract_alias_from_error(error)
        preferred_tables = []
        if alias_info:
            wrong_alias, _ = alias_info
            alias_map = extract_aliases_from_sql(predicted_sql)
            aliased_table = alias_map.get(wrong_alias)
            if aliased_table:
                preferred_tables.append(aliased_table)
        
        if not preferred_tables:
            preferred_tables = sql_tables
        
        candidates = rank_column_candidates(
            col_name,
            schema,
            preferred_tables=preferred_tables,
            secondary_tables=sorted(local_tables),
            threshold=0.6,
            top_k=5,
        )
    
    if not candidates:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=col_name,
            reason="No fuzzy match candidates found"
        )
    
    # Only accept top candidate if confidence is high (>0.85)
    top_candidate = candidates[0]
    if top_candidate["score"] < 0.85:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=col_name,
            top_candidates=candidates,
            reason=f"Top candidate score {top_candidate['score']:.2f} below threshold 0.85"
        )
    
    # Apply the replacement
    new_col = top_candidate["name"]
    
    # Check if we need table qualification
    alias_info = extract_alias_from_error(error)
    table_qualifier = None
    if alias_info:
        table_qualifier = alias_info[0]
    
    repaired_sql = apply_identifier_replacement(
        predicted_sql,
        old_identifier=col_name,
        new_identifier=new_col,
        identifier_type="column",
        table_qualifier=table_qualifier,
    )
    
    # Validate and execute
    is_valid, exec_error, _ = validate_and_execute_sql(repaired_sql, db_path, schema)
    
    if not is_valid:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=col_name,
            top_candidates=candidates,
            chosen_candidate=top_candidate,
            candidate_scope=get_candidate_scope(top_candidate),
            reason=f"Repaired SQL failed validation/execution: {exec_error}"
        )
    
    return DeterministicRepairResult(
        success=True,
        method="exact_identifier",
        repaired_sql=repaired_sql,
        failed_identifier=col_name,
        top_candidates=candidates,
        chosen_candidate=top_candidate,
        candidate_scope=get_candidate_scope(top_candidate),
        reason=f"Replaced '{col_name}' with '{new_col}' (score={top_candidate['score']:.2f})"
    )


def _fix_table_identifier(
    predicted_sql: str,
    table_name: str,
    schema: SchemaInfo,
    db_path: str,
    classification: Any,
) -> DeterministicRepairResult:
    """Fix a table identifier error deterministically."""
    
    # Get candidates from classification if available
    candidates = classification.identifier_candidates or []
    
    # If no candidates, compute them
    if not candidates:
        sql_tables = extract_tables_from_sql(predicted_sql)
        local_tables = schema.get_neighbor_tables(sql_tables, hops=1)
        
        candidates = rank_table_candidates(
            table_name,
            schema,
            preferred_tables=sql_tables,
            secondary_tables=sorted(local_tables),
            threshold=0.6,
            top_k=5,
        )
    
    if not candidates:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=table_name,
            reason="No fuzzy match candidates found"
        )
    
    # Only accept if confidence is high (>0.85)
    top_candidate = candidates[0]
    if top_candidate["score"] < 0.85:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=table_name,
            top_candidates=candidates,
            reason=f"Top candidate score {top_candidate['score']:.2f} below threshold 0.85"
        )
    
    # Apply the replacement
    new_table = top_candidate["name"]
    repaired_sql = apply_identifier_replacement(
        predicted_sql,
        old_identifier=table_name,
        new_identifier=new_table,
        identifier_type="table",
    )
    
    # Validate and execute
    is_valid, exec_error, _ = validate_and_execute_sql(repaired_sql, db_path, schema)
    
    if not is_valid:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=table_name,
            top_candidates=candidates,
            chosen_candidate=top_candidate,
            candidate_scope=get_candidate_scope(top_candidate),
            reason=f"Repaired SQL failed validation/execution: {exec_error}"
        )
    
    return DeterministicRepairResult(
        success=True,
        method="exact_identifier",
        repaired_sql=repaired_sql,
        failed_identifier=table_name,
        top_candidates=candidates,
        chosen_candidate=top_candidate,
        candidate_scope=get_candidate_scope(top_candidate),
        reason=f"Replaced table '{table_name}' with '{new_table}' (score={top_candidate['score']:.2f})"
    )


# =============================================================================
# Pure Alias Fix
# =============================================================================

def try_deterministic_alias_fix(
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
    db_path: str,
    classification: Any,
) -> DeterministicRepairResult:
    """
    Try deterministic fix for pure alias errors.
    
    Only attempts fix if:
    - alias map is unambiguous
    - column exists in schema
    - fix is only alias/table qualification
    - no join rewrite required
    """
    failed_identifier = classification.failed_identifier
    wrong_alias = classification.wrong_alias
    correct_table = classification.correct_table
    
    if not all([failed_identifier, wrong_alias, correct_table]):
        return DeterministicRepairResult(
            success=False,
            reason="Missing required alias fix information"
        )
    
    # Verify column exists in the correct table
    if not schema.column_exists_in_table(failed_identifier, correct_table):
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            reason=f"Column '{failed_identifier}' does not exist in table '{correct_table}'"
        )
    
    # Get alias map
    alias_map = extract_aliases_from_sql(predicted_sql)
    
    # Find the correct alias for the correct_table
    correct_alias = None
    for alias, table in alias_map.items():
        if table == correct_table:
            correct_alias = alias
            break
    
    if not correct_alias:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            alias_map=alias_map,
            reason=f"No alias found for correct table '{correct_table}'"
        )
    
    # Replace wrong_alias with correct_alias for this column
    # Pattern: wrong_alias.column -> correct_alias.column
    pattern = rf'\b{re.escape(wrong_alias)}\s*\.\s*`?{re.escape(failed_identifier)}`?'
    replacement = f'{correct_alias}.{failed_identifier}'
    
    repaired_sql = re.sub(pattern, replacement, predicted_sql, flags=re.IGNORECASE)
    
    if repaired_sql == predicted_sql:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            alias_map=alias_map,
            reason="No alias replacement occurred"
        )
    
    # Validate and execute
    is_valid, exec_error, _ = validate_and_execute_sql(repaired_sql, db_path, schema)
    
    if not is_valid:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            alias_map=alias_map,
            reason=f"Repaired SQL failed validation/execution: {exec_error}"
        )
    
    return DeterministicRepairResult(
        success=True,
        method="alias",
        repaired_sql=repaired_sql,
        failed_identifier=failed_identifier,
        alias_map=alias_map,
        candidate_scope="same_table",
        reason=f"Fixed alias: {wrong_alias}.{failed_identifier} -> {correct_alias}.{failed_identifier}"
    )


# =============================================================================
# Backtick Fix
# =============================================================================

def try_deterministic_backtick_fix(
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
    db_path: str,
) -> DeterministicRepairResult:
    """
    Try deterministic fix for missing backticks around special identifiers.
    
    Detects columns with spaces/punctuation/parentheses/percent signs and wraps them.
    """
    # Check if error is a syntax error that might be due to unquoted special chars
    error_lower = error.lower()
    is_syntax_error = "syntax error" in error_lower
    
    if not is_syntax_error:
        # Also check if we have unquoted special columns
        if not has_unquoted_special_column(predicted_sql, schema):
            return DeterministicRepairResult(
                success=False,
                reason="No syntax error and no unquoted special columns detected"
            )
    
    # Find columns with special characters in schema
    special_cols = []
    for table_info in schema.tables.values():
        for col in table_info.columns:
            if re.search(r'[\s\(\)\-%/]', col.name):
                special_cols.append(col.name)
    
    if not special_cols:
        return DeterministicRepairResult(
            success=False,
            reason="No special character columns in schema"
        )
    
    # Try wrapping each special column that appears unquoted
    repaired_sql = predicted_sql
    wrapped_identifiers = []
    
    for col in special_cols:
        # Skip if already backticked
        if f'`{col}`' in repaired_sql:
            continue
        
        # Check if column appears in SQL
        # For multi-word columns, check if first word appears
        words = col.split()
        if len(words) > 1:
            first_word = words[0]
            if re.search(rf'\b{re.escape(first_word)}\b', repaired_sql):
                repaired_sql = apply_backtick_wrapper(repaired_sql, col, context="column")
                wrapped_identifiers.append(col)
    
    if not wrapped_identifiers:
        return DeterministicRepairResult(
            success=False,
            reason="No special columns needed wrapping"
        )
    
    # Validate and execute
    is_valid, exec_error, _ = validate_and_execute_sql(repaired_sql, db_path, schema)
    
    if not is_valid:
        return DeterministicRepairResult(
            success=False,
            backtick_repair_used=True,
            reason=f"Backtick wrapping failed validation/execution: {exec_error}"
        )
    
    return DeterministicRepairResult(
        success=True,
        method="backtick",
        repaired_sql=repaired_sql,
        backtick_repair_used=True,
        reason=f"Wrapped {len(wrapped_identifiers)} special identifier(s) in backticks: {', '.join(wrapped_identifiers)}"
    )


# =============================================================================
# Wrong Table Side Fix
# =============================================================================

def try_deterministic_wrong_table_side_fix(
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
    db_path: str,
    classification: Any,
) -> DeterministicRepairResult:
    """
    Try deterministic fix for wrong-table-side errors.
    
    Only attempts when there's one clear candidate table.
    """
    failed_identifier = classification.failed_identifier
    correct_table = classification.correct_table
    
    if not failed_identifier or not correct_table:
        return DeterministicRepairResult(
            success=False,
            reason="Missing failed identifier or correct table"
        )
    
    # Verify column exists in correct_table
    if not schema.column_exists_in_table(failed_identifier, correct_table):
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            reason=f"Column '{failed_identifier}' does not exist in '{correct_table}'"
        )
    
    # Check if there are multiple tables with this column
    tables_with_col = schema.find_column_tables(failed_identifier)
    if len(tables_with_col) > 1:
        # Ambiguous - not safe for deterministic fix
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            reason=f"Column '{failed_identifier}' exists in multiple tables: {tables_with_col}"
        )
    
    # Get alias map
    alias_map = extract_aliases_from_sql(predicted_sql)
    
    # Find alias for correct_table
    correct_alias = None
    for alias, table in alias_map.items():
        if table == correct_table:
            correct_alias = alias
            break
    
    if not correct_alias:
        # Try using table name directly
        correct_alias = correct_table
    
    # Extract the problematic reference from error
    alias_info = extract_alias_from_error(error)
    
    if alias_info:
        wrong_alias, col = alias_info
        # Replace wrong_alias.col with correct_alias.col
        pattern = rf'\b{re.escape(wrong_alias)}\s*\.\s*`?{re.escape(col)}`?'
        replacement = f'{correct_alias}.{col}'
        repaired_sql = re.sub(pattern, replacement, predicted_sql, flags=re.IGNORECASE)
    else:
        # Try qualifying the bare column reference
        # This is trickier - need to add table qualification
        # For now, just add the alias/table prefix
        pattern = rf'\b{re.escape(failed_identifier)}\b'
        replacement = f'{correct_alias}.{failed_identifier}'
        repaired_sql = re.sub(pattern, replacement, predicted_sql)
    
    if repaired_sql == predicted_sql:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            alias_map=alias_map,
            reason="No replacement occurred"
        )
    
    # Validate and execute
    is_valid, exec_error, _ = validate_and_execute_sql(repaired_sql, db_path, schema)
    
    if not is_valid:
        return DeterministicRepairResult(
            success=False,
            failed_identifier=failed_identifier,
            alias_map=alias_map,
            reason=f"Repaired SQL failed validation/execution: {exec_error}"
        )
    
    return DeterministicRepairResult(
        success=True,
        method="wrong_table_side",
        repaired_sql=repaired_sql,
        failed_identifier=failed_identifier,
        alias_map=alias_map,
        candidate_scope="same_table",
        reason=f"Fixed table reference for '{failed_identifier}' -> '{correct_alias}.{failed_identifier}'"
    )


# =============================================================================
# Main Dispatcher
# =============================================================================

def attempt_deterministic_repair(
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
    db_path: str,
    classification: Any,
) -> DeterministicRepairResult:
    """
    Main dispatcher for deterministic repair attempts.
    
    Tries repairs in order of safety and confidence:
    1. Backtick fix (safest - just adding quotes)
    2. Pure alias fix (safe - just swapping alias)
    3. Exact identifier fix (moderate - high-confidence fuzzy match)
    4. Wrong table side fix (moderate - when unambiguous)
    
    Returns on first success.
    """
    failure_type = classification.failure_type
    
    # Try backtick fix first (safest, works across failure types)
    result = try_deterministic_backtick_fix(predicted_sql, error, schema, db_path)
    if result.success:
        return result
    
    # Route to specific fix based on failure type
    if failure_type == "alias_error":
        return try_deterministic_alias_fix(
            predicted_sql, error, schema, db_path, classification
        )
    
    elif failure_type == "exact_identifier_error":
        return try_deterministic_exact_identifier_fix(
            predicted_sql, error, schema, db_path, classification
        )
    
    elif failure_type == "wrong_table_side_error":
        return try_deterministic_wrong_table_side_fix(
            predicted_sql, error, schema, db_path, classification
        )
    
    # No deterministic repair available for this failure type
    return DeterministicRepairResult(
        success=False,
        reason=f"No deterministic repair strategy for failure type '{failure_type}'"
    )

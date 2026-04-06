#!/usr/bin/env python3
"""
Failure Classification Module

Classifies T10 prediction failures into categories for targeted repair.

Failure Taxonomy:
1. exact_identifier_error     - Column/table name misspelled or invented
2. wrong_table_side_error     - Column exists but in wrong table alias
3. alias_error                - Alias confusion (T1.col when col is in T2)
4. filter_value_mapping_error - Wrong string literal or filter condition
5. derived_metric_error       - Wrong calculation formula
6. join_backbone_error        - Wrong join structure
7. degenerate_or_truncated_sql- Incomplete/malformed SQL
8. generic_exec_error         - Other execution errors
9. wrong_result_non_exec_failure - Executes but wrong result (NOT repaired)
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from repair_utils import (
    SchemaCache,
    SchemaInfo,
    extract_column_from_error,
    extract_table_from_error,
    extract_alias_from_error,
    extract_tables_from_sql,
    extract_aliases_from_sql,
    fuzzy_find_column,
    fuzzy_find_table,
    rank_column_candidates,
    rank_table_candidates,
    sql_appears_truncated,
    has_unquoted_special_column,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FailureClassification:
    """Result of classifying a failure."""
    question_id: int
    db_id: str
    failure_type: str
    confidence: float
    reason: str
    repairability_score: float
    failed_identifier: Optional[str] = None
    suggested_fix: Optional[str] = None
    wrong_alias: Optional[str] = None
    correct_table: Optional[str] = None
    failed_identifier_scope: Optional[str] = None
    identifier_candidates: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Classification Functions
# =============================================================================

def classify_failure(
    example: Dict[str, Any],
    schema: SchemaInfo,
) -> Optional[FailureClassification]:
    """
    Classify a prediction failure.
    
    Args:
        example: Dict with keys:
            - question_id
            - db_id
            - predicted_sql
            - exec_failed (bool)
            - pred_error (str or None)
            - correct (bool)
            - wrong_result (bool)
        schema: SchemaInfo for the database
    
    Returns:
        FailureClassification or None if not a failure
    """
    question_id = example.get("question_id")
    db_id = example.get("db_id")
    predicted_sql = example.get("predicted_sql", "")
    exec_failed = example.get("exec_failed", False)
    pred_error = example.get("pred_error", "")
    correct = example.get("correct", False)
    wrong_result = example.get("wrong_result", False)
    
    # Not a failure
    if correct:
        return None
    
    # Wrong result - SQL executes but produces wrong output
    # We DO NOT repair these because we cannot verify correctness
    if not exec_failed and wrong_result:
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="wrong_result_non_exec_failure",
            confidence=0.5,
            reason="SQL executes but produces wrong result - cannot repair without gold",
            repairability_score=0.0,  # DO NOT REPAIR
        )
    
    # Execution failure - these are our repair targets
    if exec_failed:
        return classify_exec_failure(
            question_id=question_id,
            db_id=db_id,
            predicted_sql=predicted_sql,
            error=pred_error or "",
            schema=schema,
        )
    
    # Unknown state
    return FailureClassification(
        question_id=question_id,
        db_id=db_id,
        failure_type="generic_exec_error",
        confidence=0.3,
        reason="Unknown failure state",
        repairability_score=0.1,
    )


def classify_exec_failure(
    question_id: int,
    db_id: str,
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
) -> FailureClassification:
    """
    Classify an execution failure based on error message.
    """
    error_lower = error.lower()
    sql_tables = extract_tables_from_sql(predicted_sql)
    local_tables = schema.get_neighbor_tables(sql_tables, hops=1)
    
    # =========================================================================
    # Column Not Found
    # =========================================================================
    if "no such column" in error_lower:
        return classify_column_error(
            question_id, db_id, predicted_sql, error, schema
        )
    
    # =========================================================================
    # Table Not Found
    # =========================================================================
    if "no such table" in error_lower:
        table_name = extract_table_from_error(error)
        if table_name:
            table_candidates = rank_table_candidates(
                table_name,
                schema,
                preferred_tables=sql_tables,
                secondary_tables=sorted(local_tables),
            )
            best_match = table_candidates[0]["name"] if table_candidates else None
            match_score = table_candidates[0]["score"] if table_candidates else 0.0
            if match_score > 0.8:
                return FailureClassification(
                    question_id=question_id,
                    db_id=db_id,
                    failure_type="exact_identifier_error",
                    confidence=0.95,
                    reason=f"Table '{table_name}' typo, likely '{best_match}'",
                    repairability_score=0.85,
                    failed_identifier=table_name,
                    suggested_fix=best_match,
                    failed_identifier_scope=table_candidates[0]["scope"] if table_candidates else None,
                    identifier_candidates=table_candidates,
                )
            elif match_score > 0.6:
                return FailureClassification(
                    question_id=question_id,
                    db_id=db_id,
                    failure_type="exact_identifier_error",
                    confidence=0.75,
                    reason=f"Table '{table_name}' possibly '{best_match}'",
                    repairability_score=0.55,
                    failed_identifier=table_name,
                    suggested_fix=best_match,
                    failed_identifier_scope=table_candidates[0]["scope"] if table_candidates else None,
                    identifier_candidates=table_candidates,
                )
        
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="exact_identifier_error",
            confidence=0.7,
            reason=f"Table not in schema: {error}",
            repairability_score=0.4,
            failed_identifier=table_name,
            identifier_candidates=table_candidates if table_name else [],
        )
    
    # =========================================================================
    # Syntax Error
    # =========================================================================
    if "syntax error" in error_lower:
        return classify_syntax_error(
            question_id, db_id, predicted_sql, error, schema
        )
    
    # =========================================================================
    # Aggregate Misuse
    # =========================================================================
    if "misuse of aggregate" in error_lower:
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="derived_metric_error",
            confidence=0.8,
            reason="Aggregate function misuse (e.g., COUNT in WHERE without GROUP BY)",
            repairability_score=0.5,
        )
    
    # =========================================================================
    # Type Mismatch / Filter Value Errors
    # =========================================================================
    if any(pattern in error_lower for pattern in [
        "type mismatch", "datatype mismatch", "cannot compare",
        "invalid literal", "no such function: like",
        "like pattern", "argument of where must be type boolean"
    ]):
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="filter_value_mapping_error",
            confidence=0.8,
            reason=f"Type mismatch or filter value error: {error[:100]}",
            repairability_score=0.5,
        )
    
    # Check for comparison errors that suggest wrong filter value types
    if "cannot apply" in error_lower and any(op in error_lower for op in ["=", "<", ">", "like"]):
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="filter_value_mapping_error",
            confidence=0.7,
            reason=f"Comparison operator type error: {error[:100]}",
            repairability_score=0.4,
        )
    
    # =========================================================================
    # Join Backbone Errors (Ambiguous Column, Cartesian Product, Missing ON)
    # =========================================================================
    if "ambiguous column" in error_lower:
        col_match = re.search(r'ambiguous column name:\s*(\w+)', error, re.IGNORECASE)
        col_name = col_match.group(1) if col_match else None
        
        # Find which tables have this column
        tables_with_col = schema.find_column_tables(col_name) if col_name else []
        
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="join_backbone_error",
            confidence=0.8,
            reason=f"Ambiguous column '{col_name}' - missing table qualifier in join. Found in: {tables_with_col}",
            repairability_score=0.6,
            failed_identifier=col_name,
            correct_table=tables_with_col[0] if tables_with_col else None,
        )
    
    # Cartesian product / missing ON clause
    if any(pattern in error_lower for pattern in [
        "cartesian product", "cross join", "missing on clause",
        "join condition", "no join condition"
    ]):
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="join_backbone_error",
            confidence=0.75,
            reason=f"Join structure error (possible cartesian product): {error[:100]}",
            repairability_score=0.5,
        )
    
    # =========================================================================
    # Generic/Other Error
    # =========================================================================
    return FailureClassification(
        question_id=question_id,
        db_id=db_id,
        failure_type="generic_exec_error",
        confidence=0.5,
        reason=f"Unknown execution error: {error[:100]}",
        repairability_score=0.2,
    )


def classify_column_error(
    question_id: int,
    db_id: str,
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
) -> FailureClassification:
    """
    Classify a 'no such column' error.
    """
    col_name = extract_column_from_error(error)
    alias_info = extract_alias_from_error(error)
    sql_tables = extract_tables_from_sql(predicted_sql)
    local_tables = schema.get_neighbor_tables(sql_tables, hops=1)
    
    if not col_name:
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="exact_identifier_error",
            confidence=0.6,
            reason="Column error but couldn't extract column name",
            repairability_score=0.3,
        )
    
    # Check if column exists in a different table
    tables_with_col = schema.find_column_tables(col_name)
    
    if tables_with_col:
        # Column exists but possibly wrong alias
        if alias_info:
            wrong_alias, col = alias_info
            aliases = extract_aliases_from_sql(predicted_sql)
            
            # Find which table the alias refers to
            aliased_table = aliases.get(wrong_alias)
            
            if aliased_table and aliased_table not in tables_with_col:
                # Alias points to wrong table - clear alias confusion
                return FailureClassification(
                    question_id=question_id,
                    db_id=db_id,
                    failure_type="alias_error",
                    confidence=0.9,
                    reason=f"Column '{col_name}' exists in {tables_with_col} but alias '{wrong_alias}' points to '{aliased_table}'",
                    repairability_score=0.85,
                    failed_identifier=col_name,
                    wrong_alias=wrong_alias,
                    correct_table=tables_with_col[0],
                    failed_identifier_scope="same_table" if aliased_table else None,
                )
            else:
                confidence = 0.85 if len(tables_with_col) == 1 else 0.6
                repairability = 0.75 if len(tables_with_col) == 1 else 0.45
                return FailureClassification(
                    question_id=question_id,
                    db_id=db_id,
                    failure_type="wrong_table_side_error",
                    confidence=confidence,
                    reason=f"Column '{col_name}' exists in {tables_with_col} - alias issue",
                    repairability_score=repairability,
                    failed_identifier=col_name,
                    wrong_alias=wrong_alias,
                    correct_table=tables_with_col[0],
                    failed_identifier_scope="same_table" if len(tables_with_col) == 1 else "ambiguous_multi_table",
                )
        
        # Column exists, no alias in error - might be ambiguous or unqualified
        confidence = 0.8 if len(tables_with_col) == 1 else 0.55
        repairability = 0.7 if len(tables_with_col) == 1 else 0.4
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="wrong_table_side_error",
            confidence=confidence,
            reason=f"Column '{col_name}' exists in {tables_with_col} but reference is wrong",
            repairability_score=repairability,
            failed_identifier=col_name,
            correct_table=tables_with_col[0],
            failed_identifier_scope="same_table" if len(tables_with_col) == 1 else "ambiguous_multi_table",
        )
    
    # Column doesn't exist - try fuzzy matching
    preferred_tables: List[str] = []
    if alias_info:
        wrong_alias, _ = alias_info
        alias_table = extract_aliases_from_sql(predicted_sql).get(wrong_alias)
        if alias_table:
            preferred_tables.append(alias_table)
    if not preferred_tables:
        preferred_tables.extend(sql_tables)

    column_candidates = rank_column_candidates(
        col_name,
        schema,
        preferred_tables=preferred_tables,
        secondary_tables=sorted(local_tables),
    )
    best_match = column_candidates[0]["name"] if column_candidates else None
    match_score = column_candidates[0]["score"] if column_candidates else 0.0
    best_table = column_candidates[0]["table"] if column_candidates else None
    best_scope = column_candidates[0]["scope"] if column_candidates else None
    
    if match_score > 0.8:
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="exact_identifier_error",
            confidence=0.95,
            reason=f"Column '{col_name}' typo, likely '{best_match}' in table '{best_table}' (score: {match_score:.2f})",
            repairability_score=0.9,
            failed_identifier=col_name,
            suggested_fix=best_match,
            correct_table=best_table,
            failed_identifier_scope=best_scope,
            identifier_candidates=column_candidates,
        )
    elif match_score > 0.6:
        repairability = 0.65 if best_scope != "global_schema" else 0.45
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="exact_identifier_error",
            confidence=0.75,
            reason=f"Column '{col_name}' possibly '{best_match}' in table '{best_table}' (score: {match_score:.2f})",
            repairability_score=repairability,
            failed_identifier=col_name,
            suggested_fix=best_match,
            correct_table=best_table,
            failed_identifier_scope=best_scope,
            identifier_candidates=column_candidates,
        )
    else:
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="exact_identifier_error",
            confidence=0.6,
            reason=f"Column '{col_name}' not in schema, no close match found",
            repairability_score=0.3,
            failed_identifier=col_name,
            identifier_candidates=column_candidates,
        )


def classify_syntax_error(
    question_id: int,
    db_id: str,
    predicted_sql: str,
    error: str,
    schema: SchemaInfo,
) -> FailureClassification:
    """
    Classify a syntax error.
    """
    # Check for missing backticks on special columns
    if has_unquoted_special_column(predicted_sql, schema):
        # Try to identify which column
        near_match = re.search(r'near\s+"([^"]+)"', error, re.IGNORECASE)
        failed_word = near_match.group(1) if near_match else None
        
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="exact_identifier_error",
            confidence=0.85,
            reason=f"Missing backticks on special column near '{failed_word}'",
            repairability_score=0.8,
            failed_identifier=failed_word,
        )
    
    # Check for truncation
    if sql_appears_truncated(predicted_sql):
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="degenerate_or_truncated_sql",
            confidence=0.9,
            reason="SQL appears truncated or incomplete",
            repairability_score=0.6,
        )
    
    # Check for missing parenthesis
    if predicted_sql.count('(') != predicted_sql.count(')'):
        return FailureClassification(
            question_id=question_id,
            db_id=db_id,
            failure_type="degenerate_or_truncated_sql",
            confidence=0.85,
            reason="Unbalanced parentheses",
            repairability_score=0.5,
        )
    
    # Generic syntax error
    near_match = re.search(r'near\s+"([^"]+)"', error, re.IGNORECASE)
    near_word = near_match.group(1) if near_match else "unknown"
    
    return FailureClassification(
        question_id=question_id,
        db_id=db_id,
        failure_type="generic_exec_error",
        confidence=0.6,
        reason=f"Syntax error near '{near_word}'",
        repairability_score=0.3,
    )


# =============================================================================
# Batch Classification
# =============================================================================

def classify_all_failures(
    eval_results: List[Dict[str, Any]],
    schema_cache: SchemaCache,
    min_repairability: float = 0.0,
) -> Tuple[List[FailureClassification], Dict[str, Any]]:
    """
    Classify all failures in evaluation results.
    
    Args:
        eval_results: List of per-example evaluation results
        schema_cache: SchemaCache for loading database schemas
        min_repairability: Only return failures with score >= this
    
    Returns:
        (classifications, summary_stats)
    """
    classifications = []
    stats = {
        "total_examples": len(eval_results),
        "correct": 0,
        "exec_failed": 0,
        "wrong_result": 0,
        "by_failure_type": {},
        "repairable_count": 0,
        "avg_repairability": 0.0,
    }
    
    repairability_sum = 0.0
    repairability_count = 0
    
    for example in eval_results:
        if example.get("correct"):
            stats["correct"] += 1
            continue
        
        db_id = example.get("db_id")
        schema = schema_cache.get_schema(db_id)
        
        if not schema:
            print(f"Warning: Could not load schema for {db_id}")
            continue
        
        classification = classify_failure(example, schema)
        
        if classification:
            if example.get("exec_failed"):
                stats["exec_failed"] += 1
            if example.get("wrong_result"):
                stats["wrong_result"] += 1
            
            failure_type = classification.failure_type
            stats["by_failure_type"][failure_type] = stats["by_failure_type"].get(failure_type, 0) + 1
            
            repairability_sum += classification.repairability_score
            repairability_count += 1
            
            if classification.repairability_score >= min_repairability:
                classifications.append(classification)
                if classification.repairability_score > 0:
                    stats["repairable_count"] += 1
    
    if repairability_count > 0:
        stats["avg_repairability"] = repairability_sum / repairability_count
    
    return classifications, stats


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify T10 prediction failures")
    parser.add_argument("--eval_results", required=True, help="Path to per_example_results.jsonl")
    parser.add_argument("--db_dir", required=True, help="Path to database directory")
    parser.add_argument("--output", required=True, help="Output path for classified failures")
    parser.add_argument("--min_repairability", type=float, default=0.0, help="Minimum repairability score")
    
    args = parser.parse_args()
    
    # Load evaluation results
    eval_results = []
    with open(args.eval_results, 'r') as f:
        for line in f:
            if line.strip():
                eval_results.append(json.loads(line))
    
    print(f"Loaded {len(eval_results)} evaluation results")
    
    # Initialize schema cache
    schema_cache = SchemaCache(args.db_dir)
    
    # Classify failures
    classifications, stats = classify_all_failures(
        eval_results, 
        schema_cache,
        min_repairability=args.min_repairability,
    )
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for c in classifications:
            f.write(json.dumps(c.to_dict()) + '\n')
    
    # Write stats
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nClassification Summary:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Exec failed: {stats['exec_failed']}")
    print(f"  Wrong result: {stats['wrong_result']}")
    print(f"  Repairable (score > 0): {stats['repairable_count']}")
    print(f"  Avg repairability: {stats['avg_repairability']:.2f}")
    print(f"\nBy failure type:")
    for ftype, count in sorted(stats['by_failure_type'].items(), key=lambda x: -x[1]):
        print(f"  {ftype}: {count}")
    
    print(f"\nOutput written to: {output_path}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()

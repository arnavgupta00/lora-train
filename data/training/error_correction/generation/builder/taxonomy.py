#!/usr/bin/env python3
"""
Taxonomy Module

Defines the failure type taxonomy and target failure modes
for error-correction examples.
"""

from enum import Enum
from typing import Dict, List, Optional, Set


class FailureType(str, Enum):
    """Controlled failure type categories."""
    
    # Identifier and naming errors
    EXACT_IDENTIFIER_ERROR = "exact_identifier_error"
    WRONG_TABLE_SIDE_ERROR = "wrong_table_side_error"
    ALIAS_ERROR = "alias_error"
    
    # Return value errors
    WRONG_RETURN_FIELD = "wrong_return_field"
    WRONG_COUNT_GRANULARITY = "wrong_count_granularity"
    MISSING_DISTINCT = "missing_distinct"
    
    # Aggregation and scope errors
    WRONG_DENOMINATOR = "wrong_denominator"
    WRONG_COHORT_DEFINITION = "wrong_cohort_definition"
    AGGREGATE_LOGIC_ERROR = "aggregate_logic_error"
    
    # Join and relationship errors
    JOIN_PATH_ERROR = "join_path_error"
    TABLE_FAMILY_CONFUSION = "table_family_confusion"
    
    # Filter and condition errors
    WRONG_FILTER_CONDITION = "wrong_filter_condition"
    WRONG_LITERAL_MAPPING = "wrong_literal_mapping"
    TEMPORAL_ANCHOR_ERROR = "temporal_anchor_error"
    
    # Output shape errors
    OUTPUT_SHAPE_ERROR = "output_shape_error"
    
    # Structural and syntax errors
    DEGENERATE_OR_TRUNCATED_SQL = "degenerate_or_truncated_sql"
    SYNTAX_LOCAL_ERROR = "syntax_local_error"


class TargetFailureMode(str, Enum):
    """Operational target failure modes."""
    
    # Field selection
    RETURN_FIELD_MISMATCH = "return_field_mismatch"
    WRONG_COUNT_GRANULARITY = "wrong_count_granularity"
    WRONG_TABLE_OWNERSHIP = "wrong_table_ownership"
    
    # Aggregation
    WRONG_DENOMINATOR = "wrong_denominator"
    AGGREGATION_LEVEL_ERROR = "aggregation_level_error"
    DISTINCT_SHAPE_ERROR = "distinct_shape_error"
    
    # Schema navigation
    PHASE_TABLE_CONFUSION = "phase_table_confusion"
    JOIN_PATH_ERROR = "join_path_error"
    
    # Time and anchoring
    TEMPORAL_ANCHOR_ERROR = "temporal_anchor_error"
    
    # Output
    OUTPUT_FORMAT_ERROR = "output_format_error"
    
    # Syntax
    COLUMN_ERROR = "column_error"
    TABLE_ERROR = "table_error"
    SYNTAX_ERROR = "syntax_error"
    TRUNCATION_ERROR = "truncation_error"


# Mapping from failure types to suggested target failure modes
FAILURE_TYPE_TO_TARGET_MODES: Dict[FailureType, List[TargetFailureMode]] = {
    FailureType.EXACT_IDENTIFIER_ERROR: [
        TargetFailureMode.COLUMN_ERROR,
        TargetFailureMode.TABLE_ERROR,
    ],
    FailureType.WRONG_TABLE_SIDE_ERROR: [
        TargetFailureMode.WRONG_TABLE_OWNERSHIP,
    ],
    FailureType.ALIAS_ERROR: [
        TargetFailureMode.COLUMN_ERROR,
        TargetFailureMode.TABLE_ERROR,
    ],
    FailureType.WRONG_RETURN_FIELD: [
        TargetFailureMode.RETURN_FIELD_MISMATCH,
    ],
    FailureType.WRONG_COUNT_GRANULARITY: [
        TargetFailureMode.WRONG_COUNT_GRANULARITY,
        TargetFailureMode.DISTINCT_SHAPE_ERROR,
    ],
    FailureType.MISSING_DISTINCT: [
        TargetFailureMode.DISTINCT_SHAPE_ERROR,
    ],
    FailureType.WRONG_DENOMINATOR: [
        TargetFailureMode.WRONG_DENOMINATOR,
        TargetFailureMode.AGGREGATION_LEVEL_ERROR,
    ],
    FailureType.WRONG_COHORT_DEFINITION: [
        TargetFailureMode.WRONG_DENOMINATOR,
        TargetFailureMode.AGGREGATION_LEVEL_ERROR,
    ],
    FailureType.AGGREGATE_LOGIC_ERROR: [
        TargetFailureMode.AGGREGATION_LEVEL_ERROR,
    ],
    FailureType.JOIN_PATH_ERROR: [
        TargetFailureMode.JOIN_PATH_ERROR,
    ],
    FailureType.TABLE_FAMILY_CONFUSION: [
        TargetFailureMode.PHASE_TABLE_CONFUSION,
        TargetFailureMode.JOIN_PATH_ERROR,
    ],
    FailureType.WRONG_FILTER_CONDITION: [
        TargetFailureMode.WRONG_DENOMINATOR,
    ],
    FailureType.WRONG_LITERAL_MAPPING: [
        TargetFailureMode.RETURN_FIELD_MISMATCH,
    ],
    FailureType.TEMPORAL_ANCHOR_ERROR: [
        TargetFailureMode.TEMPORAL_ANCHOR_ERROR,
    ],
    FailureType.OUTPUT_SHAPE_ERROR: [
        TargetFailureMode.OUTPUT_FORMAT_ERROR,
        TargetFailureMode.RETURN_FIELD_MISMATCH,
    ],
    FailureType.DEGENERATE_OR_TRUNCATED_SQL: [
        TargetFailureMode.TRUNCATION_ERROR,
    ],
    FailureType.SYNTAX_LOCAL_ERROR: [
        TargetFailureMode.SYNTAX_ERROR,
        TargetFailureMode.COLUMN_ERROR,
        TargetFailureMode.TABLE_ERROR,
    ],
}


# High-priority failure types (need more coverage)
HIGH_PRIORITY_FAILURE_TYPES: Set[FailureType] = {
    FailureType.WRONG_RETURN_FIELD,
    FailureType.WRONG_COUNT_GRANULARITY,
    FailureType.WRONG_DENOMINATOR,
    FailureType.WRONG_COHORT_DEFINITION,
    FailureType.JOIN_PATH_ERROR,
    FailureType.TABLE_FAMILY_CONFUSION,
    FailureType.TEMPORAL_ANCHOR_ERROR,
}


# Mapping from error categories (from per_example_results) to failure types
ERROR_CATEGORY_TO_FAILURE_TYPE: Dict[str, FailureType] = {
    "column_error": FailureType.EXACT_IDENTIFIER_ERROR,
    "table_error": FailureType.EXACT_IDENTIFIER_ERROR,
    "syntax_error": FailureType.SYNTAX_LOCAL_ERROR,
    "value_error": FailureType.WRONG_LITERAL_MAPPING,
}


def infer_failure_type(
    exec_failed: bool,
    wrong_result: bool,
    error_category: Optional[str] = None,
    error_message: Optional[str] = None,
    broken_sql: Optional[str] = None,
    gold_sql: Optional[str] = None,
) -> FailureType:
    """
    Infer failure type from available signals.
    
    This is a heuristic inference - actual classification may require
    deeper analysis.
    """
    # Execution failures with error category
    if exec_failed and error_category:
        if error_category in ERROR_CATEGORY_TO_FAILURE_TYPE:
            return ERROR_CATEGORY_TO_FAILURE_TYPE[error_category]
    
    # Execution failures with error message
    if exec_failed and error_message:
        error_lower = error_message.lower()
        if "no such column" in error_lower:
            return FailureType.EXACT_IDENTIFIER_ERROR
        if "no such table" in error_lower:
            return FailureType.EXACT_IDENTIFIER_ERROR
        if "syntax error" in error_lower:
            return FailureType.SYNTAX_LOCAL_ERROR
        if "ambiguous" in error_lower:
            return FailureType.ALIAS_ERROR
    
    # Truncated or degenerate SQL
    if broken_sql:
        stripped = broken_sql.strip()
        if len(stripped) < 10:
            return FailureType.DEGENERATE_OR_TRUNCATED_SQL
        if not stripped.upper().startswith("SELECT"):
            return FailureType.DEGENERATE_OR_TRUNCATED_SQL
    
    # Wrong result - need SQL comparison to determine type
    if wrong_result:
        # Default to wrong_return_field as most common semantic error
        return FailureType.WRONG_RETURN_FIELD
    
    # Fallback
    if exec_failed:
        return FailureType.SYNTAX_LOCAL_ERROR
    
    return FailureType.WRONG_RETURN_FIELD


def get_suggested_target_mode(failure_type: FailureType) -> TargetFailureMode:
    """Get the primary suggested target failure mode for a failure type."""
    modes = FAILURE_TYPE_TO_TARGET_MODES.get(failure_type, [])
    if modes:
        return modes[0]
    return TargetFailureMode.RETURN_FIELD_MISMATCH


def get_failure_type_priority(failure_type: FailureType) -> int:
    """Get priority score for a failure type (higher = more important)."""
    if failure_type in HIGH_PRIORITY_FAILURE_TYPES:
        return 2
    return 1


def get_all_failure_types() -> List[str]:
    """Get all failure type values as strings."""
    return [ft.value for ft in FailureType]


def get_all_target_modes() -> List[str]:
    """Get all target failure mode values as strings."""
    return [tm.value for tm in TargetFailureMode]


# Priority weak database families
PRIORITY_WEAK_FAMILIES: Set[str] = {
    "california_schools",
    "financial",
    "formula_1",
    "thrombosis_prediction",
    "debit_card_specializing",
    "card_games",
    "toxicology",
}


# Robustness database families (strong, but need coverage for generalization)
ROBUSTNESS_FAMILIES: Set[str] = {
    "codebase_community",
    "european_football_2",
    "student_club",
    "superhero",
}


def get_db_family_priority(db_family: str) -> int:
    """Get priority score for a database family (higher = more important)."""
    if db_family in PRIORITY_WEAK_FAMILIES:
        return 2
    if db_family in ROBUSTNESS_FAMILIES:
        return 1
    return 0


def get_sampling_weight(
    failure_type: FailureType,
    db_family: str,
    source_type: str,
) -> float:
    """
    Calculate sampling weight for an example.
    
    Weights are multiplicative factors for weighted sampling.
    """
    weight = 1.0
    
    # Source type weights
    source_weights = {
        "real_failure": 1.3,
        "contrastive": 1.2,
        "manual_fix": 1.4,
        "deterministic_fix": 1.1,
        "synthetic_corruption": 1.0,
    }
    weight *= source_weights.get(source_type, 1.0)
    
    # Failure type priority boost
    if failure_type in HIGH_PRIORITY_FAILURE_TYPES:
        weight *= 1.2
    
    # Database family priority boost
    if db_family in PRIORITY_WEAK_FAMILIES:
        weight *= 1.15
    elif db_family in ROBUSTNESS_FAMILIES:
        weight *= 1.05
    
    return round(weight, 2)

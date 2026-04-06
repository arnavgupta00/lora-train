#!/usr/bin/env python3
"""
Corruption Module

Synthetic SQL corruption transforms for generating training examples.
"""

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .schema_builder import SchemaInfo, TableInfo


class CorruptionSeverity(str, Enum):
    """Severity of corruption."""
    MINOR = "minor"      # Easy to detect/fix
    MODERATE = "moderate"  # Requires some analysis
    MAJOR = "major"      # Significant structural change


class CorruptionFamily(str, Enum):
    """Family of corruption transforms."""
    IDENTIFIER = "identifier"
    AGGREGATION = "aggregation"
    FILTER = "filter"
    JOIN = "join"
    OUTPUT = "output"
    STRUCTURAL = "structural"


@dataclass
class CorruptionResult:
    """Result of applying a corruption transform."""
    
    success: bool
    broken_sql: str
    transform_name: str
    family: str
    severity: str
    description: str
    error_message: Optional[str] = None  # Synthetic error message if applicable


@dataclass
class CorruptionTransform:
    """A corruption transform."""
    
    name: str
    family: CorruptionFamily
    severity: CorruptionSeverity
    description: str
    apply_fn: Callable[[str, SchemaInfo], CorruptionResult]
    applicable_check: Optional[Callable[[str, SchemaInfo], bool]] = None
    failure_type: str = ""  # Corresponding failure type


def extract_select_columns(sql: str) -> List[str]:
    """Extract column expressions from SELECT clause."""
    match = re.search(r'\bSELECT\s+(.*?)\s+FROM\b', sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    
    select_part = match.group(1)
    # Split by comma, but be careful about nested parens
    columns = []
    depth = 0
    current = []
    
    for char in select_part:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            columns.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
    
    if current:
        columns.append(''.join(current).strip())
    
    return columns


def extract_table_aliases(sql: str) -> Dict[str, str]:
    """Extract table aliases from SQL."""
    aliases = {}
    
    # FROM table AS alias or FROM table alias
    for match in re.finditer(
        r'\b(?:FROM|JOIN)\s+`?(\w+)`?\s+(?:AS\s+)?`?(\w+)`?',
        sql, re.IGNORECASE
    ):
        table = match.group(1)
        alias = match.group(2)
        if alias.upper() not in ('ON', 'WHERE', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'JOIN', 'AND', 'OR'):
            aliases[alias] = table
    
    return aliases


def get_all_columns(schema: SchemaInfo, table_name: str) -> List[str]:
    """Get all column names for a table."""
    table_lower = table_name.lower()
    for t_name, t_info in schema.tables.items():
        if t_name.lower() == table_lower:
            return [c.name for c in t_info.columns]
    return []


# =============================================================================
# Corruption Transforms
# =============================================================================

def corrupt_wrong_column(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Replace a column with a similar but wrong column from the same table."""
    aliases = extract_table_aliases(sql)
    
    # Find qualified column references
    qualified_refs = list(re.finditer(r'`?(\w+)`?\s*\.\s*`?([^`\s,\)]+)`?', sql))
    
    if not qualified_refs:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_column",
            family=CorruptionFamily.IDENTIFIER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="No qualified column references found",
        )
    
    # Pick a random reference to corrupt
    ref = random.choice(qualified_refs)
    alias_or_table = ref.group(1)
    column = ref.group(2)
    
    # Find the actual table
    table_name = aliases.get(alias_or_table, alias_or_table)
    columns = get_all_columns(schema, table_name)
    
    if len(columns) < 2:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_column",
            family=CorruptionFamily.IDENTIFIER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="Table has fewer than 2 columns",
        )
    
    # Pick a different column
    other_columns = [c for c in columns if c.lower() != column.lower()]
    if not other_columns:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_column",
            family=CorruptionFamily.IDENTIFIER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="No alternative columns found",
        )
    
    new_column = random.choice(other_columns)
    
    # Replace the column
    start, end = ref.span()
    broken_sql = sql[:start] + f"{alias_or_table}.{new_column}" + sql[end:]
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="wrong_column",
        family=CorruptionFamily.IDENTIFIER.value,
        severity=CorruptionSeverity.MODERATE.value,
        description=f"Changed {alias_or_table}.{column} to {alias_or_table}.{new_column}",
    )


def corrupt_remove_distinct(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Remove DISTINCT from a query that has it."""
    if 'DISTINCT' not in sql.upper():
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="remove_distinct",
            family=CorruptionFamily.AGGREGATION.value,
            severity=CorruptionSeverity.MINOR.value,
            description="No DISTINCT found",
        )
    
    # Remove DISTINCT
    broken_sql = re.sub(r'\bDISTINCT\s+', '', sql, count=1, flags=re.IGNORECASE)
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="remove_distinct",
        family=CorruptionFamily.AGGREGATION.value,
        severity=CorruptionSeverity.MINOR.value,
        description="Removed DISTINCT keyword",
    )


def corrupt_count_distinct_to_count(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Change COUNT(DISTINCT x) to COUNT(x)."""
    pattern = r'\bCOUNT\s*\(\s*DISTINCT\s+'
    
    if not re.search(pattern, sql, re.IGNORECASE):
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="count_distinct_to_count",
            family=CorruptionFamily.AGGREGATION.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="No COUNT(DISTINCT ...) found",
        )
    
    broken_sql = re.sub(pattern, 'COUNT(', sql, count=1, flags=re.IGNORECASE)
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="count_distinct_to_count",
        family=CorruptionFamily.AGGREGATION.value,
        severity=CorruptionSeverity.MODERATE.value,
        description="Changed COUNT(DISTINCT x) to COUNT(x)",
    )


def corrupt_count_to_count_star(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Change COUNT(column) to COUNT(*)."""
    pattern = r'\bCOUNT\s*\(\s*`?(\w+)`?\s*\)'
    
    match = re.search(pattern, sql, re.IGNORECASE)
    if not match or match.group(1) == '*':
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="count_to_count_star",
            family=CorruptionFamily.AGGREGATION.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="No COUNT(column) found",
        )
    
    broken_sql = re.sub(pattern, 'COUNT(*)', sql, count=1, flags=re.IGNORECASE)
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="count_to_count_star",
        family=CorruptionFamily.AGGREGATION.value,
        severity=CorruptionSeverity.MODERATE.value,
        description=f"Changed COUNT({match.group(1)}) to COUNT(*)",
    )


def corrupt_drop_filter(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Drop a filter condition from WHERE clause."""
    where_match = re.search(r'\bWHERE\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', 
                           sql, re.IGNORECASE | re.DOTALL)
    
    if not where_match:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="drop_filter",
            family=CorruptionFamily.FILTER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="No WHERE clause found",
        )
    
    where_clause = where_match.group(1)
    
    # Try to find AND conditions
    and_parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
    
    if len(and_parts) < 2:
        # Remove entire WHERE clause
        broken_sql = re.sub(r'\bWHERE\s+.+?(?=\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)',
                          '', sql, count=1, flags=re.IGNORECASE | re.DOTALL)
        description = "Removed entire WHERE clause"
    else:
        # Remove one condition
        removed_idx = random.randrange(len(and_parts))
        remaining = [p for i, p in enumerate(and_parts) if i != removed_idx]
        new_where = " AND ".join(remaining)
        
        start = where_match.start(1)
        end = where_match.end(1)
        broken_sql = sql[:start] + new_where + sql[end:]
        description = f"Removed condition: {and_parts[removed_idx].strip()}"
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="drop_filter",
        family=CorruptionFamily.FILTER.value,
        severity=CorruptionSeverity.MODERATE.value,
        description=description,
    )


def corrupt_wrong_table_alias(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Swap table aliases in a column reference."""
    aliases = extract_table_aliases(sql)
    
    if len(aliases) < 2:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_table_alias",
            family=CorruptionFamily.IDENTIFIER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="Query has fewer than 2 table aliases",
        )
    
    alias_list = list(aliases.keys())
    
    # Find a column reference to corrupt
    qualified_refs = list(re.finditer(r'`?(\w+)`?\s*\.\s*`?([^`\s,\)]+)`?', sql))
    
    if not qualified_refs:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_table_alias",
            family=CorruptionFamily.IDENTIFIER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="No qualified column references found",
        )
    
    ref = random.choice(qualified_refs)
    old_alias = ref.group(1)
    column = ref.group(2)
    
    if old_alias not in alias_list:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_table_alias",
            family=CorruptionFamily.IDENTIFIER.value,
            severity=CorruptionSeverity.MODERATE.value,
            description="Alias not in alias list",
        )
    
    # Pick a different alias
    other_aliases = [a for a in alias_list if a != old_alias]
    new_alias = random.choice(other_aliases)
    
    start, end = ref.span()
    broken_sql = sql[:start] + f"{new_alias}.{column}" + sql[end:]
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="wrong_table_alias",
        family=CorruptionFamily.IDENTIFIER.value,
        severity=CorruptionSeverity.MODERATE.value,
        description=f"Changed {old_alias}.{column} to {new_alias}.{column}",
    )


def corrupt_truncate(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Truncate SQL at a random point."""
    min_length = 20
    if len(sql) < min_length * 2:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="truncate",
            family=CorruptionFamily.STRUCTURAL.value,
            severity=CorruptionSeverity.MAJOR.value,
            description="SQL too short to truncate",
        )
    
    # Truncate at 50-80% of length
    truncate_ratio = random.uniform(0.5, 0.8)
    truncate_point = int(len(sql) * truncate_ratio)
    
    broken_sql = sql[:truncate_point]
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="truncate",
        family=CorruptionFamily.STRUCTURAL.value,
        severity=CorruptionSeverity.MAJOR.value,
        description=f"Truncated at position {truncate_point}",
    )


def corrupt_wrong_order_direction(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Change ASC to DESC or vice versa."""
    if 'ASC' in sql.upper():
        broken_sql = re.sub(r'\bASC\b', 'DESC', sql, count=1, flags=re.IGNORECASE)
        description = "Changed ASC to DESC"
    elif 'DESC' in sql.upper():
        broken_sql = re.sub(r'\bDESC\b', 'ASC', sql, count=1, flags=re.IGNORECASE)
        description = "Changed DESC to ASC"
    else:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="wrong_order_direction",
            family=CorruptionFamily.OUTPUT.value,
            severity=CorruptionSeverity.MINOR.value,
            description="No ASC/DESC found",
        )
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="wrong_order_direction",
        family=CorruptionFamily.OUTPUT.value,
        severity=CorruptionSeverity.MINOR.value,
        description=description,
    )


def corrupt_change_limit(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Change LIMIT value."""
    match = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
    
    if not match:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="change_limit",
            family=CorruptionFamily.OUTPUT.value,
            severity=CorruptionSeverity.MINOR.value,
            description="No LIMIT found",
        )
    
    old_limit = int(match.group(1))
    
    # Change limit to a different value
    if old_limit == 1:
        new_limit = random.choice([3, 5, 10])
    else:
        new_limit = random.choice([1, old_limit + 1, old_limit * 2])
    
    broken_sql = sql[:match.start(1)] + str(new_limit) + sql[match.end(1):]
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="change_limit",
        family=CorruptionFamily.OUTPUT.value,
        severity=CorruptionSeverity.MINOR.value,
        description=f"Changed LIMIT {old_limit} to LIMIT {new_limit}",
    )


def corrupt_remove_backticks(sql: str, schema: SchemaInfo) -> CorruptionResult:
    """Remove backticks from identifiers that need them."""
    # Find backtick-quoted identifiers with special chars
    special_ids = list(re.finditer(r'`([^`]*[\s\-\(\)%/][^`]*)`', sql))
    
    if not special_ids:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name="remove_backticks",
            family=CorruptionFamily.STRUCTURAL.value,
            severity=CorruptionSeverity.MINOR.value,
            description="No backtick-quoted special identifiers found",
        )
    
    # Remove backticks from one identifier
    match = random.choice(special_ids)
    identifier = match.group(1)
    
    start, end = match.span()
    broken_sql = sql[:start] + identifier + sql[end:]
    
    return CorruptionResult(
        success=True,
        broken_sql=broken_sql,
        transform_name="remove_backticks",
        family=CorruptionFamily.STRUCTURAL.value,
        severity=CorruptionSeverity.MINOR.value,
        description=f"Removed backticks from `{identifier}`",
        error_message=f"near \"{identifier.split()[0]}\": syntax error",
    )


# =============================================================================
# Transform Registry
# =============================================================================

CORRUPTION_TRANSFORMS: List[CorruptionTransform] = [
    CorruptionTransform(
        name="wrong_column",
        family=CorruptionFamily.IDENTIFIER,
        severity=CorruptionSeverity.MODERATE,
        description="Replace column with similar wrong column",
        apply_fn=corrupt_wrong_column,
        failure_type="wrong_return_field",
    ),
    CorruptionTransform(
        name="remove_distinct",
        family=CorruptionFamily.AGGREGATION,
        severity=CorruptionSeverity.MINOR,
        description="Remove DISTINCT keyword",
        apply_fn=corrupt_remove_distinct,
        failure_type="missing_distinct",
    ),
    CorruptionTransform(
        name="count_distinct_to_count",
        family=CorruptionFamily.AGGREGATION,
        severity=CorruptionSeverity.MODERATE,
        description="Change COUNT(DISTINCT x) to COUNT(x)",
        apply_fn=corrupt_count_distinct_to_count,
        failure_type="wrong_count_granularity",
    ),
    CorruptionTransform(
        name="count_to_count_star",
        family=CorruptionFamily.AGGREGATION,
        severity=CorruptionSeverity.MODERATE,
        description="Change COUNT(column) to COUNT(*)",
        apply_fn=corrupt_count_to_count_star,
        failure_type="wrong_count_granularity",
    ),
    CorruptionTransform(
        name="drop_filter",
        family=CorruptionFamily.FILTER,
        severity=CorruptionSeverity.MODERATE,
        description="Drop a filter condition",
        apply_fn=corrupt_drop_filter,
        failure_type="wrong_cohort_definition",
    ),
    CorruptionTransform(
        name="wrong_table_alias",
        family=CorruptionFamily.IDENTIFIER,
        severity=CorruptionSeverity.MODERATE,
        description="Swap table aliases",
        apply_fn=corrupt_wrong_table_alias,
        failure_type="wrong_table_side_error",
    ),
    CorruptionTransform(
        name="truncate",
        family=CorruptionFamily.STRUCTURAL,
        severity=CorruptionSeverity.MAJOR,
        description="Truncate SQL",
        apply_fn=corrupt_truncate,
        failure_type="degenerate_or_truncated_sql",
    ),
    CorruptionTransform(
        name="wrong_order_direction",
        family=CorruptionFamily.OUTPUT,
        severity=CorruptionSeverity.MINOR,
        description="Change ASC/DESC",
        apply_fn=corrupt_wrong_order_direction,
        failure_type="output_shape_error",
    ),
    CorruptionTransform(
        name="change_limit",
        family=CorruptionFamily.OUTPUT,
        severity=CorruptionSeverity.MINOR,
        description="Change LIMIT value",
        apply_fn=corrupt_change_limit,
        failure_type="output_shape_error",
    ),
    CorruptionTransform(
        name="remove_backticks",
        family=CorruptionFamily.STRUCTURAL,
        severity=CorruptionSeverity.MINOR,
        description="Remove required backticks",
        apply_fn=corrupt_remove_backticks,
        failure_type="syntax_local_error",
    ),
]


def get_transform_by_name(name: str) -> Optional[CorruptionTransform]:
    """Get a transform by name."""
    for t in CORRUPTION_TRANSFORMS:
        if t.name == name:
            return t
    return None


def apply_random_corruption(
    sql: str,
    schema: SchemaInfo,
    exclude_transforms: Optional[Set[str]] = None,
    prefer_family: Optional[CorruptionFamily] = None,
) -> Optional[CorruptionResult]:
    """
    Apply a random applicable corruption transform.
    
    Args:
        sql: Original SQL
        schema: Schema information
        exclude_transforms: Transform names to exclude
        prefer_family: Prefer transforms from this family
    
    Returns:
        CorruptionResult if successful, None if no transform applicable
    """
    exclude = exclude_transforms or set()
    
    # Filter and optionally prefer by family
    candidates = [t for t in CORRUPTION_TRANSFORMS if t.name not in exclude]
    
    if prefer_family:
        preferred = [t for t in candidates if t.family == prefer_family]
        if preferred:
            candidates = preferred
    
    # Shuffle and try transforms
    random.shuffle(candidates)
    
    for transform in candidates:
        result = transform.apply_fn(sql, schema)
        if result.success:
            return result
    
    return None


def apply_specific_corruption(
    sql: str,
    schema: SchemaInfo,
    transform_name: str,
) -> CorruptionResult:
    """Apply a specific corruption transform by name."""
    transform = get_transform_by_name(transform_name)
    
    if not transform:
        return CorruptionResult(
            success=False,
            broken_sql=sql,
            transform_name=transform_name,
            family="unknown",
            severity="unknown",
            description=f"Unknown transform: {transform_name}",
        )
    
    return transform.apply_fn(sql, schema)

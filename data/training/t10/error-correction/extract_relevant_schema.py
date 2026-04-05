#!/usr/bin/env python3
"""
Relevant Schema Extraction Module

Extracts a minimal, targeted schema block for SQL repair.

Uses signals from:
- Question text
- Hints/evidence
- Predicted SQL (table/column references)
- Execution error message

Does NOT use gold SQL.
"""

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from repair_utils import (
    SchemaInfo,
    TableInfo,
    ColumnInfo,
    ForeignKey,
    extract_tables_from_sql,
    extract_columns_from_sql,
    extract_aliases_from_sql,
    extract_column_from_error,
    fuzzy_match,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractionContext:
    """Context for schema extraction."""
    question: str
    hints: str
    predicted_sql: str
    error: str
    error_column: Optional[str] = None
    
    def __post_init__(self):
        # Pre-compute tokens for matching
        self.question_tokens = set(tokenize(self.question))
        self.hints_tokens = set(tokenize(self.hints)) if self.hints else set()
        self.sql_tables = set(t.lower() for t in extract_tables_from_sql(self.predicted_sql))
        self.sql_columns = extract_columns_from_sql(self.predicted_sql)
        self.sql_aliases = extract_aliases_from_sql(self.predicted_sql)
        
        # Extract error column if not provided
        if not self.error_column and self.error:
            self.error_column = extract_column_from_error(self.error)


@dataclass
class RelevantSchemaBlock:
    """Result of schema extraction."""
    tables: List[str]
    columns_by_table: Dict[str, List[str]]
    relations: List[str]
    notes: List[str]
    ddl_block: str = ""
    required_tables: List[str] = field(default_factory=list)
    optional_tables: List[str] = field(default_factory=list)
    
    def format(self) -> str:
        """Format as text block for prompt."""
        lines = []
        
        # Tables and columns
        for table in self.tables:
            cols = self.columns_by_table.get(table, [])
            if cols:
                cols_str = ", ".join(f"`{c}`" if needs_backticks(c) else c for c in cols)
                lines.append(f"- {table}: {cols_str}")
            else:
                lines.append(f"- {table}")
        
        return "\n".join(lines)
    
    def format_ddl(self) -> str:
        """Return DDL-style schema block."""
        return self.ddl_block
    
    def format_relations(self) -> str:
        """Format relations as text."""
        if not self.relations:
            return "None"
        return "\n".join(f"- {r}" for r in self.relations)
    
    def format_notes(self) -> str:
        """Format notes as text."""
        if not self.notes:
            return "None"
        return "\n".join(f"- {n}" for n in self.notes)


# =============================================================================
# Helper Functions
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Tokenize text into words for matching."""
    if not text:
        return []
    # Split on non-alphanumeric, lowercase
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return tokens


def needs_backticks(name: str) -> bool:
    """Check if identifier needs backticks."""
    return bool(re.search(r'[\s\(\)\-%/]', name))


def token_overlap(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard-style overlap between token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def text_in_text(needle: str, haystack: str) -> bool:
    """Check if needle appears in haystack (case-insensitive)."""
    if not needle or not haystack:
        return False
    return needle.lower() in haystack.lower()


# =============================================================================
# Table Scoring
# =============================================================================

def score_table(
    table_name: str,
    table_info: TableInfo,
    context: ExtractionContext,
    schema: SchemaInfo,
) -> float:
    """
    Score a table's relevance for repair.
    Higher score = more relevant.
    """
    score = 0.0
    table_lower = table_name.lower()
    table_tokens = set(tokenize(table_name))
    
    # Question overlap
    if text_in_text(table_name, context.question):
        score += 3.0
    score += token_overlap(table_tokens, context.question_tokens) * 2.0
    
    # Column name overlap with question
    col_tokens = set()
    for col in table_info.columns:
        col_tokens.update(tokenize(col.name))
    score += token_overlap(col_tokens, context.question_tokens) * 1.5
    
    # Hints overlap (strong signal)
    if context.hints:
        if text_in_text(table_name, context.hints):
            score += 4.0
        score += token_overlap(table_tokens, context.hints_tokens) * 2.5
        score += token_overlap(col_tokens, context.hints_tokens) * 2.0
    
    # Used in predicted SQL (very strong signal)
    if table_lower in context.sql_tables:
        score += 6.0
    
    # Check if any alias maps to this table
    for alias, tbl in context.sql_aliases.items():
        if tbl.lower() == table_lower:
            score += 5.0
            break
    
    # Execution error clues
    if context.error:
        if text_in_text(table_name, context.error):
            score += 4.0
        
        # Check if error column might be in this table
        if context.error_column:
            for col in table_info.columns:
                if fuzzy_match(col.name, context.error_column) > 0.7:
                    score += 5.0
                    break
    
    return score


def compute_required_bridge_tables(
    anchor_tables: List[str],
    schema: SchemaInfo,
) -> List[str]:
    """Return bridge tables required to keep anchor tables FK-connected."""
    if len(anchor_tables) < 2:
        return []

    canonical = {name.lower(): name for name in schema.tables.keys()}
    graph = build_fk_graph(schema)
    required: List[str] = []
    required_lower: Set[str] = set()
    anchor_lower = [t.lower() for t in anchor_tables if t.lower() in canonical]

    connected_component: List[str] = []
    for table_lower in anchor_lower:
        if table_lower in required_lower:
            continue
        if not connected_component:
            connected_component.append(table_lower)
            required.append(canonical[table_lower])
            required_lower.add(table_lower)
            continue

        candidate_paths = []
        for existing in connected_component:
            path = shortest_fk_path(graph, existing, table_lower)
            if path:
                candidate_paths.append(path)

        if candidate_paths:
            best_path = min(candidate_paths, key=len)
            for node in best_path:
                if node not in required_lower and node in canonical:
                    required.append(canonical[node])
                    required_lower.add(node)
            connected_component.extend(node for node in best_path if node not in connected_component)
        else:
            required.append(canonical[table_lower])
            required_lower.add(table_lower)
            connected_component.append(table_lower)

    return required


def select_tables(
    schema: SchemaInfo,
    context: ExtractionContext,
    max_tables: int = 4,
    min_score: float = 1.0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Select the most relevant tables for repair, preserving required bridge tables.
    """
    scores = []
    for table_name, table_info in schema.tables.items():
        s = score_table(table_name, table_info, context, schema)
        scores.append((table_name, s))
    
    # Sort by score descending
    scores.sort(key=lambda x: -x[1])
    
    ranked_tables = [t for t, s in scores if s >= min_score]

    sql_tables_in_schema = [t for t in context.sql_tables if t in schema.tables]
    if sql_tables_in_schema:
        anchor_tables = list(dict.fromkeys(sql_tables_in_schema))
    else:
        anchor_tables = ranked_tables[: min(2, max_tables)]

    required_tables = compute_required_bridge_tables(anchor_tables, schema)
    required_lower = {t.lower() for t in required_tables}

    optional_tables: List[str] = []
    for table_name, _ in scores:
        if table_name.lower() not in required_lower:
            optional_tables.append(table_name)

    if len(required_tables) >= max_tables:
        selected = required_tables
    else:
        remaining = max_tables - len(required_tables)
        selected = required_tables + optional_tables[:remaining]

    if not selected:
        selected = [t for t, _ in scores[:max_tables]]

    return selected, required_tables, optional_tables


def build_fk_graph(schema: SchemaInfo) -> Dict[str, Set[str]]:
    """Build an undirected FK connectivity graph over schema tables."""
    graph: Dict[str, Set[str]] = defaultdict(set)

    for table_name in schema.tables:
        graph[table_name.lower()]

    for fk in schema.foreign_keys:
        from_tbl = fk.from_table.lower()
        to_tbl = fk.to_table.lower()
        graph[from_tbl].add(to_tbl)
        graph[to_tbl].add(from_tbl)

    return graph


def shortest_fk_path(
    graph: Dict[str, Set[str]],
    start: str,
    goal: str,
) -> Optional[List[str]]:
    """Return the shortest FK path between two tables, if one exists."""
    start = start.lower()
    goal = goal.lower()

    if start == goal:
        return [start]
    if start not in graph or goal not in graph:
        return None

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))

    return None


# =============================================================================
# Column Scoring
# =============================================================================

def score_column(
    col: ColumnInfo,
    table_name: str,
    context: ExtractionContext,
    schema: SchemaInfo,
    is_primary_table: bool,
) -> float:
    """
    Score a column's relevance for repair.
    """
    score = 0.0
    col_tokens = set(tokenize(col.name))
    
    # Always keep PK and FK columns
    if col.is_pk:
        score += 10.0
    if col.is_fk:
        score += 8.0
    
    # Referenced in predicted SQL (very strong)
    for alias, col_name in context.sql_columns:
        if col_name.lower() == col.name.lower():
            # Check if alias matches this table
            if alias:
                alias_table = context.sql_aliases.get(alias, alias)
                if alias_table.lower() == table_name.lower():
                    score += 10.0
                else:
                    # Column referenced but maybe wrong table
                    score += 6.0
            else:
                score += 8.0
    
    # Execution error mentions this column
    if context.error_column:
        if col.name.lower() == context.error_column.lower():
            score += 10.0
        elif fuzzy_match(col.name, context.error_column) > 0.7:
            score += 8.0
        elif fuzzy_match(col.name, context.error_column) > 0.5:
            score += 4.0
    
    # Question overlap
    if text_in_text(col.name, context.question):
        score += 4.0
    score += token_overlap(col_tokens, context.question_tokens) * 2.0
    
    # Hints overlap (strong signal)
    if context.hints:
        if text_in_text(col.name, context.hints):
            score += 5.0
        score += token_overlap(col_tokens, context.hints_tokens) * 3.0
    
    # Bonus for primary table
    if is_primary_table:
        score *= 1.2
    
    return score


def select_columns(
    table_name: str,
    table_info: TableInfo,
    context: ExtractionContext,
    schema: SchemaInfo,
    is_primary_table: bool,
    is_bridge_table: bool,
    expanded: bool = False,
) -> List[str]:
    """
    Select the most relevant columns from a table.
    
    Args:
        expanded: If True, increase column limits for retry attempts.
    """
    # Set max columns based on table role (expanded mode increases limits)
    if is_primary_table:
        max_cols = 16 if expanded else 12
    elif is_bridge_table:
        max_cols = 7 if expanded else 5
    else:
        max_cols = 12 if expanded else 8
    
    # Score all columns
    scores = []
    for col in table_info.columns:
        s = score_column(col, table_name, context, schema, is_primary_table)
        scores.append((col.name, s, col.is_pk or col.is_fk))
    
    # Sort by score descending
    scores.sort(key=lambda x: (-x[1], not x[2]))
    
    # Must keep: PK, FK, high scores
    must_keep = [name for name, score, is_key in scores if is_key or score >= 8.0]
    optional = [name for name, score, is_key in scores if not is_key and score < 8.0 and score > 0]
    
    # Take must_keep + fill with optional
    result = list(must_keep)
    for name in optional:
        if name not in result:
            result.append(name)
            if len(result) >= max_cols:
                break
    
    # Ensure we have at least some columns
    if len(result) < 3:
        for col in table_info.columns[:5]:
            if col.name not in result:
                result.append(col.name)
    
    return result[:max_cols]


# =============================================================================
# Relation Extraction
# =============================================================================

def extract_relations(
    selected_tables: List[str],
    schema: SchemaInfo,
) -> List[str]:
    """
    Extract FK relations between selected tables.
    """
    relations = []
    selected_set = set(t.lower() for t in selected_tables)
    
    for fk in schema.foreign_keys:
        if fk.from_table.lower() in selected_set and fk.to_table.lower() in selected_set:
            rel = f"{fk.from_table}.{fk.from_col} = {fk.to_table}.{fk.to_col}"
            if rel not in relations:
                relations.append(rel)
    
    return relations


# =============================================================================
# Notes Generation
# =============================================================================

def generate_notes(
    context: ExtractionContext,
    selected_columns: Dict[str, List[str]],
    schema: SchemaInfo,
) -> List[str]:
    """
    Generate helpful notes for repair.
    Only high-confidence notes.
    """
    notes = []
    
    # Extract hints that look like formulas/mappings
    if context.hints:
        # Pattern: "X = Y / Z" or "X refers to Y"
        if "=" in context.hints and any(c in context.hints for c in ['/', '*', '+', '-']):
            notes.append(context.hints.strip())
        elif "refers to" in context.hints.lower():
            notes.append(context.hints.strip())
    
    # If error column has a suggested match
    if context.error_column:
        best_match = None
        best_score = 0.0
        best_table = None
        
        for table, cols in selected_columns.items():
            for col in cols:
                score = fuzzy_match(col, context.error_column)
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = col
                    best_table = table
        
        if best_match and best_match.lower() != context.error_column.lower():
            notes.append(f"The column '{context.error_column}' should likely be `{best_match}` (in {best_table})")
    
    return notes[:3]


# =============================================================================
# DDL Formatting
# =============================================================================

def format_ddl_block(
    selected_tables: List[str],
    columns_by_table: Dict[str, List[str]],
    schema: SchemaInfo,
) -> str:
    """
    Format selected schema as DDL-style block.
    """
    lines = []
    
    for table_name in selected_tables:
        if table_name not in schema.tables:
            continue
        
        table_info = schema.tables[table_name]
        cols = columns_by_table.get(table_name, table_info.get_column_names())
        
        # Build CREATE TABLE
        col_lines = []
        for col_name in cols:
            col_info = next((c for c in table_info.columns if c.name == col_name), None)
            if col_info:
                name_str = f"`{col_name}`" if needs_backticks(col_name) else col_name
                col_line = f"    {name_str} {col_info.col_type}"
                if col_info.is_pk:
                    col_line += " PRIMARY KEY"
                col_lines.append(col_line)
        
        lines.append(f"CREATE TABLE {table_name}")
        lines.append("(")
        lines.append(",\n".join(col_lines))
        lines.append(")")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_relevant_schema(
    schema: SchemaInfo,
    question: str,
    hints: str,
    predicted_sql: str,
    error: str,
    expanded: bool = False,
) -> RelevantSchemaBlock:
    """
    Extract a minimal, relevant schema block for SQL repair.
    
    Args:
        schema: Full database schema
        question: Natural language question
        hints: Hints/evidence text
        predicted_sql: The SQL that failed
        error: Execution error message
        expanded: If True, expand context with more columns per table (for retry attempts)
    
    Returns:
        RelevantSchemaBlock with selected tables, columns, relations, notes
    """
    # Build context
    context = ExtractionContext(
        question=question,
        hints=hints,
        predicted_sql=predicted_sql,
        error=error,
    )
    
    # Select tables (expanded mode allows one extra table)
    max_tables = 5 if expanded else 4
    selected_tables, required_tables, optional_tables = select_tables(schema, context, max_tables=max_tables)
    
    # Determine primary table (highest scored, or first from SQL)
    primary_table = selected_tables[0] if selected_tables else None
    
    # Determine bridge tables
    bridge_tables = set()
    for fk in schema.foreign_keys:
        # A table is a bridge if it connects two other selected tables
        if fk.from_table in selected_tables and fk.to_table in selected_tables:
            # Check if this table only connects others
            if len(schema.tables.get(fk.from_table, TableInfo("", [], [])).columns) <= 5:
                bridge_tables.add(fk.from_table)
    
    # Select columns for each table
    columns_by_table = {}
    for table_name in selected_tables:
        if table_name not in schema.tables:
            continue
        
        table_info = schema.tables[table_name]
        is_primary = (table_name == primary_table)
        is_bridge = (table_name in bridge_tables)
        
        columns_by_table[table_name] = select_columns(
            table_name, table_info, context, schema, is_primary, is_bridge, expanded
        )
    
    # Extract relations
    relations = extract_relations(selected_tables, schema)
    
    # Generate notes
    notes = generate_notes(context, columns_by_table, schema)
    
    # Format DDL block
    ddl_block = format_ddl_block(selected_tables, columns_by_table, schema)
    
    return RelevantSchemaBlock(
        tables=selected_tables,
        columns_by_table=columns_by_table,
        relations=relations,
        notes=notes,
        ddl_block=ddl_block,
        required_tables=required_tables,
        optional_tables=optional_tables,
    )


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Relevant Schema Extraction Module")
    print("=" * 60)
    print("Use extract_relevant_schema() to extract schema for repair.")

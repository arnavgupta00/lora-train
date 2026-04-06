#!/usr/bin/env python3
"""
Schema Builder Module

Builds schema context for error-correction examples.
Supports compact (relevant) and full schema modes.
"""

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import PathConfig, SchemaConfig


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: str
    is_pk: bool = False
    is_fk: bool = False
    fk_ref: Optional[Tuple[str, str]] = None  # (table, column)


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    pk_columns: List[str] = field(default_factory=list)
    
    def get_column_names(self) -> List[str]:
        return [c.name for c in self.columns]


@dataclass
class SchemaInfo:
    """Complete schema information for a database."""
    db_id: str
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    foreign_keys: List[Tuple[str, str, str, str]] = field(default_factory=list)  # (src_table, src_col, dst_table, dst_col)
    raw_ddl: str = ""
    
    def get_table_names(self) -> List[str]:
        return list(self.tables.keys())
    
    def get_all_columns(self) -> Dict[str, List[str]]:
        return {t: info.get_column_names() for t, info in self.tables.items()}


def load_schema_from_db(db_path: Path) -> SchemaInfo:
    """Load schema information from a SQLite database file."""
    db_id = db_path.parent.name
    schema = SchemaInfo(db_id=db_id)
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get raw DDL
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        ddl_parts = [row[0] for row in cursor.fetchall() if row[0]]
        schema.raw_ddl = "\n".join(ddl_parts)
        
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
                col_type = row[2] or "TEXT"
                is_pk = row[5] > 0
                
                columns.append(ColumnInfo(
                    name=col_name,
                    type=col_type,
                    is_pk=is_pk,
                ))
                
                if is_pk:
                    pk_columns.append(col_name)
            
            schema.tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                pk_columns=pk_columns,
            )
        
        # Get foreign keys
        for table_name in table_names:
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            for row in cursor.fetchall():
                fk_table = row[2]
                fk_from = row[3]
                fk_to = row[4]
                schema.foreign_keys.append((table_name, fk_from, fk_table, fk_to))
                
                # Mark column as FK
                if table_name in schema.tables:
                    for col in schema.tables[table_name].columns:
                        if col.name == fk_from:
                            col.is_fk = True
                            col.fk_ref = (fk_table, fk_to)
        
        conn.close()
    except Exception as e:
        print(f"Error loading schema from {db_path}: {e}")
    
    return schema


def extract_identifiers_from_text(text: str) -> Set[str]:
    """Extract potential table/column identifiers from question or hints."""
    if not text:
        return set()
    
    identifiers = set()
    
    # Extract quoted identifiers
    for match in re.finditer(r"'([^']+)'", text):
        identifiers.add(match.group(1).lower())
    
    for match in re.finditer(r'"([^"]+)"', text):
        identifiers.add(match.group(1).lower())
    
    # Extract potential column names (with underscores)
    for match in re.finditer(r'\b([a-z_][a-z0-9_]*)\b', text.lower()):
        word = match.group(1)
        if len(word) > 2 and '_' in word:
            identifiers.add(word)
    
    return identifiers


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from SQL query."""
    if not sql:
        return set()
    
    tables = set()
    
    # FROM clause tables
    for match in re.finditer(r'\bFROM\s+`?([A-Za-z_]\w*)`?', sql, re.IGNORECASE):
        tables.add(match.group(1))
    
    # JOIN tables
    for match in re.finditer(r'\bJOIN\s+`?([A-Za-z_]\w*)`?', sql, re.IGNORECASE):
        tables.add(match.group(1))
    
    return tables


def extract_columns_from_sql(sql: str) -> Set[str]:
    """Extract column names from SQL query."""
    if not sql:
        return set()
    
    columns = set()
    
    # Qualified columns (table.column)
    for match in re.finditer(r'`?(\w+)`?\s*\.\s*`?([^`\s,\)]+)`?', sql):
        columns.add(match.group(2).strip('`'))
    
    # Backtick-quoted columns
    for match in re.finditer(r'`([^`]+)`', sql):
        col = match.group(1)
        if '.' not in col:
            columns.add(col)
    
    return columns


def find_relevant_tables(
    schema: SchemaInfo,
    question: str,
    hints: str,
    broken_sql: Optional[str] = None,
) -> Set[str]:
    """
    Find tables relevant to the question using heuristic matching.
    """
    relevant = set()
    all_text = f"{question} {hints}".lower()
    
    # Tables mentioned in broken SQL
    if broken_sql:
        sql_tables = extract_tables_from_sql(broken_sql)
        for table in sql_tables:
            for schema_table in schema.tables:
                if table.lower() == schema_table.lower():
                    relevant.add(schema_table)
    
    # Tables with name matches in question/hints
    for table_name in schema.tables:
        # Direct match
        if table_name.lower() in all_text:
            relevant.add(table_name)
        
        # Match without underscores
        normalized = table_name.lower().replace('_', ' ')
        if normalized in all_text:
            relevant.add(table_name)
        
        # Match individual words
        words = table_name.lower().split('_')
        if len(words) > 1 and all(w in all_text for w in words if len(w) > 3):
            relevant.add(table_name)
    
    return relevant


def find_bridge_tables(
    schema: SchemaInfo,
    selected_tables: Set[str],
) -> Set[str]:
    """Find bridge tables that connect selected tables."""
    bridges = set()
    selected_lower = {t.lower() for t in selected_tables}
    
    # Build FK graph
    fk_graph = defaultdict(set)
    for src_table, src_col, dst_table, dst_col in schema.foreign_keys:
        fk_graph[src_table.lower()].add(dst_table.lower())
        fk_graph[dst_table.lower()].add(src_table.lower())
    
    # Find tables that connect two selected tables
    for table_name in schema.tables:
        if table_name.lower() in selected_lower:
            continue
        
        connected = fk_graph.get(table_name.lower(), set())
        connected_to_selected = connected & selected_lower
        
        if len(connected_to_selected) >= 2:
            bridges.add(table_name)
    
    return bridges


def find_relevant_columns(
    schema: SchemaInfo,
    table_name: str,
    question: str,
    hints: str,
    broken_sql: Optional[str] = None,
    max_columns: int = 15,
) -> List[str]:
    """
    Find relevant columns for a table.
    
    Priority order:
    1. PK/FK columns (always included)
    2. Columns mentioned in question/hints/SQL
    3. Columns with semantic matches
    """
    if table_name not in schema.tables:
        return []
    
    table_info = schema.tables[table_name]
    all_text = f"{question} {hints}".lower()
    
    # SQL columns
    sql_columns = set()
    if broken_sql:
        sql_columns = {c.lower() for c in extract_columns_from_sql(broken_sql)}
    
    # Score columns
    scored_columns = []
    for col in table_info.columns:
        score = 0
        
        # PK/FK always high priority
        if col.is_pk:
            score += 100
        if col.is_fk:
            score += 90
        
        # Mentioned in SQL
        if col.name.lower() in sql_columns:
            score += 80
        
        # Direct text match
        if col.name.lower() in all_text:
            score += 60
        
        # Normalized match
        normalized = col.name.lower().replace('_', ' ')
        if normalized in all_text:
            score += 50
        
        # Word match
        words = col.name.lower().split('_')
        matching_words = sum(1 for w in words if len(w) > 3 and w in all_text)
        score += matching_words * 10
        
        scored_columns.append((col.name, score))
    
    # Sort by score descending
    scored_columns.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top columns
    result = [name for name, score in scored_columns[:max_columns]]
    
    return result


def build_compact_schema(
    schema: SchemaInfo,
    question: str,
    hints: str,
    broken_sql: Optional[str] = None,
    config: Optional[SchemaConfig] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a compact schema context.
    
    Returns:
        (schema_text, metadata)
    """
    if config is None:
        config = SchemaConfig()
    
    # Find relevant tables
    relevant_tables = find_relevant_tables(schema, question, hints, broken_sql)
    
    # Add bridge tables
    bridges = find_bridge_tables(schema, relevant_tables)
    selected_tables = relevant_tables | bridges
    
    # Limit table count
    if len(selected_tables) > config.max_compact_tables:
        # Keep most relevant, prioritize tables mentioned in SQL
        sql_tables = set()
        if broken_sql:
            sql_tables = {t.lower() for t in extract_tables_from_sql(broken_sql)}
        
        scored = []
        for t in selected_tables:
            score = 0
            if t.lower() in sql_tables:
                score += 10
            if t in relevant_tables:
                score += 5
            if t in bridges:
                score += 2
            scored.append((t, score))
        
        scored.sort(key=lambda x: -x[1])
        selected_tables = {t for t, _ in scored[:config.max_compact_tables]}
    
    # Fallback: if no tables found, use all tables
    if not selected_tables:
        selected_tables = set(schema.tables.keys())
    
    # Build schema text
    parts = []
    columns_kept = 0
    tables_kept = []
    
    for table_name in sorted(selected_tables):
        if table_name not in schema.tables:
            continue
        
        table_info = schema.tables[table_name]
        
        # Find relevant columns
        relevant_cols = find_relevant_columns(
            schema, table_name, question, hints, broken_sql,
            max_columns=config.max_compact_columns_per_table,
        )
        
        if not relevant_cols:
            relevant_cols = [c.name for c in table_info.columns][:config.max_compact_columns_per_table]
        
        # Build column list
        col_defs = []
        for col in table_info.columns:
            if col.name in relevant_cols:
                col_def = f"    {col.name} {col.type}"
                if col.is_pk:
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)
                columns_kept += 1
        
        # Build table DDL
        table_ddl = f"CREATE TABLE {table_name}\n(\n"
        table_ddl += ",\n".join(col_defs)
        table_ddl += "\n)"
        
        parts.append(table_ddl)
        tables_kept.append(table_name)
    
    schema_text = "\n".join(parts)
    
    # Build metadata
    metadata = {
        "schema_context_type": "compact_relevant",
        "tables_kept": tables_kept,
        "columns_kept": columns_kept,
        "total_tables": len(schema.tables),
        "total_columns": sum(len(t.columns) for t in schema.tables.values()),
    }
    
    return schema_text, metadata


def build_full_schema(schema: SchemaInfo) -> Tuple[str, Dict[str, Any]]:
    """
    Build full schema context (DDL).
    
    Returns:
        (schema_text, metadata)
    """
    tables_kept = list(schema.tables.keys())
    columns_kept = sum(len(t.columns) for t in schema.tables.values())
    
    metadata = {
        "schema_context_type": "full_schema",
        "tables_kept": tables_kept,
        "columns_kept": columns_kept,
        "total_tables": len(schema.tables),
        "total_columns": columns_kept,
    }
    
    return schema.raw_ddl, metadata


class SchemaBuilder:
    """Builder for schema contexts."""
    
    def __init__(self, paths: PathConfig, config: Optional[SchemaConfig] = None):
        self.paths = paths
        self.config = config or SchemaConfig()
        self._schema_cache: Dict[str, SchemaInfo] = {}
    
    def get_schema(self, db_id: str) -> SchemaInfo:
        """Get schema for a database (cached)."""
        if db_id not in self._schema_cache:
            db_path = self.paths.database_path(db_id)
            self._schema_cache[db_id] = load_schema_from_db(db_path)
        return self._schema_cache[db_id]
    
    def build_context(
        self,
        db_id: str,
        question: str,
        hints: str,
        broken_sql: Optional[str] = None,
        use_compact: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build schema context for an example.
        
        Args:
            db_id: Database identifier
            question: Question text
            hints: Hints/evidence text
            broken_sql: The broken SQL (optional, used for relevance)
            use_compact: Whether to use compact schema (default True)
        
        Returns:
            (schema_text, metadata)
        """
        schema = self.get_schema(db_id)
        
        if use_compact:
            return build_compact_schema(
                schema, question, hints, broken_sql, self.config
            )
        else:
            return build_full_schema(schema)
    
    def should_use_compact(self) -> bool:
        """
        Decide whether to use compact schema based on config ratio.
        
        Call this to randomly select schema type according to the compact_ratio.
        """
        import random
        return random.random() < self.config.compact_ratio

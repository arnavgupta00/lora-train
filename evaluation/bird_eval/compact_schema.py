#!/usr/bin/env python3
"""
Compact Schema Module for BIRD Eval

Schema compaction that works WITHOUT gold SQL, using only:
- Question text
- Hints (evidence)
- Full schema

Key Features:
1. Heuristic extraction from question+hints only (no gold SQL leakage)
2. Primary-table bonus - most relevant table keeps more columns
3. Bridge-table preservation - FK-connected tables between selected tables
4. Stricter fallback thresholds (target 65-80% reduction)
5. Widen-once-before-full-fallback
6. Per-example compaction metadata

Adapted from t11_1_utils.py but modified for question-only extraction.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


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
class QuestionExtractionResult:
    """Result of question/hint identifier extraction."""
    tables: Set[str]
    columns: Set[str]
    table_mentions: Dict[str, int]  # table -> mention count (for primary table detection)
    is_confident: bool
    fallback_reason: Optional[str] = None


@dataclass
class CompactionMetadata:
    """Per-example compaction metadata for auditability."""
    original_schema_length: int
    compact_schema_length: int
    reduction_percent: float
    primary_table: Optional[str]
    tables_kept: List[str]
    tables_dropped: List[str]
    compaction_status: str  # "success", "widened", "fallback"
    fallback_reason: Optional[str]
    pass_number: int  # 1 = standard, 2 = widened


@dataclass
class CompactionResult:
    """Result of schema compaction."""
    compact_schema: str
    metadata: CompactionMetadata
    kept_columns: Dict[str, List[str]] = field(default_factory=dict)


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
    name_match = re.match(r'"([^"]+)"\s*\(', stmt)
    if name_match:
        table_name = name_match.group(1)
    else:
        name_match = re.match(r'`([^`]+)`\s*\(', stmt)
        if name_match:
            table_name = name_match.group(1)
        else:
            name_match = re.match(r'([A-Za-z_][A-Za-z0-9_\-]*)\s*\(', stmt)
            if not name_match:
                return None
            table_name = name_match.group(1)

    table_info = TableInfo(name=table_name, raw_ddl=f"CREATE TABLE {stmt.strip()}")

    # Find the column definitions section
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
    column_defs = _split_by_comma(columns_str)

    for col_def in column_defs:
        col_def = col_def.strip()
        if not col_def:
            continue

        col_def_upper = col_def.upper()

        # Check for PRIMARY KEY constraint
        if col_def_upper.startswith('PRIMARY KEY'):
            pk_match = re.search(r'PRIMARY\s+KEY\s*\(([^)]+)\)', col_def, re.IGNORECASE)
            if pk_match:
                pk_cols = [c.strip().strip('`"') for c in pk_match.group(1).split(',')]
                table_info.pk_columns.extend(pk_cols)
            continue

        # Check for FOREIGN KEY constraint
        if col_def_upper.startswith('FOREIGN KEY'):
            fk_match = re.search(
                r'FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                col_def, re.IGNORECASE
            )
            if fk_match:
                fk_cols = _parse_column_list(fk_match.group(1))
                ref_table = fk_match.group(2).strip()
                ref_cols = _parse_column_list(fk_match.group(3))
                for fk_col, ref_col in zip(fk_cols, ref_cols):
                    table_info.fk_columns[fk_col] = (ref_table, ref_col)
            continue

        # Skip other constraints
        if col_def_upper.startswith(('UNIQUE', 'CHECK', 'CONSTRAINT')):
            continue

        # Regular column definition
        backtick_match = re.match(r'`([^`]+)`', col_def)
        if backtick_match:
            col_name = backtick_match.group(1)
            table_info.columns.append(f"`{col_name}`")
            if 'PRIMARY KEY' in col_def_upper:
                table_info.pk_columns.append(f"`{col_name}`")
            ref_match = re.search(
                r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                col_def, re.IGNORECASE
            )
            if ref_match:
                ref_table = ref_match.group(1)
                ref_col = ref_match.group(2).strip().strip('`"')
                table_info.fk_columns[f"`{col_name}`"] = (ref_table, ref_col)
            continue

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

        col_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', col_def)
        if col_match:
            col_name = col_match.group(1)
            if col_name.upper() not in ('PRIMARY', 'FOREIGN', 'UNIQUE', 'CHECK', 'CONSTRAINT', 'KEY'):
                if not col_name.startswith('-') and not col_name.startswith('#'):
                    table_info.columns.append(col_name)
                    if 'PRIMARY KEY' in col_def_upper:
                        table_info.pk_columns.append(col_name)
                    ref_match = re.search(
                        r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)',
                        col_def, re.IGNORECASE
                    )
                    if ref_match:
                        ref_table = ref_match.group(1)
                        ref_col = ref_match.group(2).strip().strip('`"')
                        table_info.fk_columns[col_name] = (ref_table, ref_col)

    return table_info


def _parse_column_list(cols_str: str) -> List[str]:
    """Parse comma-separated column names, handling quoted identifiers."""
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

    col = current.strip()
    if col:
        cols.append(col)

    return cols


def _split_by_comma(s: str) -> List[str]:
    """Split string by comma, respecting nested parentheses and stripping SQL comments."""
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
# Question/Hint Identifier Extraction (NO GOLD SQL)
# =============================================================================

def extract_question_identifiers(
    question: str,
    hints: Optional[str],
    schema_info: SchemaInfo
) -> QuestionExtractionResult:
    """
    Extract table and column identifiers from question and hints.
    Uses heuristic text matching - NO gold SQL access.
    
    Strategy:
    1. Exact match against schema identifiers
    2. Fuzzy word matching (split on underscores, spaces)
    3. Case-insensitive matching
    """
    all_tables, all_columns = get_all_schema_identifiers(schema_info)
    
    tables: Set[str] = set()
    columns: Set[str] = set()
    table_mentions: Dict[str, int] = defaultdict(int)
    
    # Combine question and hints for matching
    search_text = question.lower()
    if hints and hints.strip().lower() != 'none':
        search_text += " " + hints.lower()
    
    # Remove punctuation for word extraction but keep original for exact matching
    search_words = set(re.findall(r'[a-z0-9_]+', search_text))
    
    # --- Match tables ---
    for table_name in all_tables:
        table_lower = table_name.lower().strip('`"')
        table_words = set(re.split(r'[_\s]+', table_lower))
        
        # Exact match (table name appears in text)
        if table_lower in search_text:
            tables.add(table_name)
            # Count mentions for primary table detection
            table_mentions[table_name] += search_text.count(table_lower)
            continue
        
        # Word overlap match (e.g., "schools" matches "california_schools")
        # Require at least one significant word (>3 chars) to match
        significant_matches = [w for w in table_words if len(w) > 3 and w in search_words]
        if significant_matches:
            tables.add(table_name)
            table_mentions[table_name] += len(significant_matches)
    
    # --- Match columns ---
    for col_name in all_columns:
        col_lower = col_name.lower().strip('`"')
        col_words = set(re.split(r'[_\s()]+', col_lower))
        
        # Exact match
        if col_lower in search_text:
            columns.add(col_name)
            continue
        
        # Word overlap match (require significant word)
        significant_matches = [w for w in col_words if len(w) > 3 and w in search_words]
        if significant_matches:
            columns.add(col_name)
    
    # --- Add tables that own matched columns ---
    for col_name in columns:
        for table_name, table_info in schema_info.tables.items():
            if col_name in table_info.columns:
                if table_name not in tables:
                    tables.add(table_name)
                    table_mentions[table_name] += 1  # Lower priority than direct mention
    
    # Determine confidence
    is_confident = len(tables) > 0
    fallback_reason = None if is_confident else "No tables matched from question/hints"
    
    return QuestionExtractionResult(
        tables=tables,
        columns=columns,
        table_mentions=dict(table_mentions),
        is_confident=is_confident,
        fallback_reason=fallback_reason
    )


# =============================================================================
# FK Bridge Table Detection
# =============================================================================

def find_bridge_tables(
    selected_tables: Set[str],
    fk_graph: Dict[Tuple[str, str], Tuple[str, str]],
    all_tables: Set[str]
) -> Set[str]:
    """
    Find bridge tables needed to connect selected tables.
    Uses BFS on FK graph to find intermediate tables.
    
    EXPLICIT RULE: If selected tables are disconnected, include
    FK-connected bridge tables and their key columns.
    """
    # Build adjacency: table -> set of directly FK-connected tables
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for (src_tbl, _), (dst_tbl, _) in fk_graph.items():
        adjacency[src_tbl].add(dst_tbl)
        adjacency[dst_tbl].add(src_tbl)

    bridges: Set[str] = set()
    selected_list = list(selected_tables & all_tables)

    # For each pair of selected tables, find shortest path
    for i, t1 in enumerate(selected_list):
        for t2 in selected_list[i+1:]:
            if t2 in adjacency[t1]:
                continue  # Direct connection exists

            # BFS to find shortest path
            visited = {t1}
            queue: List[Tuple[str, List[str]]] = [(t1, [])]
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
# Safety Margin Application
# =============================================================================

def apply_safety_margin(
    table_name: str,
    required_columns: Set[str],
    all_columns: List[str],
    question: str,
    hints: Optional[str],
    is_primary_table: bool = False,
    is_bridge_table: bool = False,
    widened: bool = False,
) -> Set[str]:
    """
    Apply deterministic safety margin rules.
    Returns final set of columns to keep.
    
    EXPLICIT RULES:
    - Primary table: keeps more columns (8+ extra)
    - Bridge tables: keep PK/FK + minimal margin
    - Other tables: standard margin
    
    Parameters (standard / widened):
    - SMALL_TABLE_THRESHOLD: 8 / 8 (keep all for small tables)
    - MAX_QUESTION_EXTRAS: 5 / 8
    - PRIMARY_TABLE_EXTRAS: 8 / 12
    - BRIDGE_TABLE_EXTRAS: 2 / 3
    - MIN_COLS_PER_TABLE: 4 / 6
    """
    final_columns = set(required_columns)

    # Tunable params based on mode and table type
    SMALL_TABLE_THRESHOLD = 8
    if widened:
        MAX_QUESTION_EXTRAS = 8
        PRIMARY_TABLE_EXTRAS = 12
        BRIDGE_TABLE_EXTRAS = 3
        MIN_COLS_PER_TABLE = 6
    else:
        MAX_QUESTION_EXTRAS = 5
        PRIMARY_TABLE_EXTRAS = 8
        BRIDGE_TABLE_EXTRAS = 2
        MIN_COLS_PER_TABLE = 4

    # Rule A: Small tables (≤8 columns) - keep everything
    if len(all_columns) <= SMALL_TABLE_THRESHOLD:
        return set(all_columns)

    # Rule B: Bridge tables - minimal margin
    if is_bridge_table:
        # Just ensure minimum columns
        min_cols = min(MIN_COLS_PER_TABLE, len(all_columns))
        if len(final_columns) < min_cols:
            for col in all_columns:
                if col not in final_columns:
                    final_columns.add(col)
                if len(final_columns) >= min_cols:
                    break
        return final_columns

    # Rule C: Add columns matching question/hints
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

    max_extras = MAX_QUESTION_EXTRAS
    for col in candidates[:max_extras]:
        final_columns.add(col)

    # Rule D: Primary table bonus - keep adjacent columns in DDL order
    if is_primary_table and len(all_columns) > len(final_columns):
        # Find indices of currently-kept columns
        required_indices = set()
        for col in final_columns:
            if col in all_columns:
                required_indices.add(all_columns.index(col))

        # Collect adjacent columns (within ±2 positions)
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

    # Rule E: Ensure minimum columns per table
    min_cols = min(MIN_COLS_PER_TABLE, len(all_columns))
    if len(final_columns) < min_cols:
        for col in all_columns:
            if col not in final_columns:
                final_columns.add(col)
            if len(final_columns) >= min_cols:
                break

    return final_columns


# =============================================================================
# Over-Compaction Checks (STRICTER THRESHOLDS)
# =============================================================================

def check_over_compaction(
    compact_len: int,
    original_len: int,
    kept_tables: List[str],
    kept_columns: Dict[str, List[str]],
    schema_info: SchemaInfo,
) -> Tuple[bool, Optional[str]]:
    """
    Check if compaction is too aggressive.
    
    STRICTER THRESHOLDS (target 65-80% reduction):
    1. Compact < 400 chars AND original > 2000 chars → too small
    2. Reduction > 85% on schemas > 3000 chars → too aggressive
    3. Any kept table has < 3 columns → too thin
    """
    reduction_pct = 100 * (1 - compact_len / original_len) if original_len > 0 else 0
    
    # Guard 1: Absolute minimum size for large schemas
    if original_len > 2000 and compact_len < 400:
        return True, f"Compact too small ({compact_len} chars) for large schema ({original_len} chars)"
    
    # Guard 2: Reduction percentage cap for very large schemas
    if original_len > 3000 and reduction_pct > 85:
        return True, f"Reduction too aggressive ({reduction_pct:.1f}%) for large schema ({original_len} chars)"

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


# =============================================================================
# Primary Table Detection
# =============================================================================

def find_primary_table(
    kept_tables: List[str],
    question_result: QuestionExtractionResult
) -> Optional[str]:
    """
    Find the primary table (most relevant to the question).
    
    EXPLICIT RULE: Primary table is determined by:
    1. Most mentions in question/hints
    2. First mentioned if tie
    """
    if not kept_tables:
        return None
    
    # Sort by mention count (descending), then by name for determinism
    sorted_tables = sorted(
        kept_tables,
        key=lambda t: (-question_result.table_mentions.get(t, 0), t)
    )
    
    return sorted_tables[0] if sorted_tables else None


# =============================================================================
# Column Selection for Tables
# =============================================================================

def build_columns_for_tables(
    schema_info: SchemaInfo,
    kept_tables: List[str],
    question_result: QuestionExtractionResult,
    question: str,
    hints: Optional[str],
    primary_table: Optional[str],
    bridge_tables: Set[str],
    widened: bool,
) -> Dict[str, List[str]]:
    """
    Build the column sets for each kept table, applying safety margins.
    Returns {table_name: [sorted columns]}.
    """
    kept_columns: Dict[str, List[str]] = {}
    
    for table_name in kept_tables:
        table_info = schema_info.tables[table_name]

        # Start with required columns
        required_cols: Set[str] = set()

        # Add columns matched from question/hints
        for col in question_result.columns:
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
        is_bridge = (table_name in bridge_tables)
        
        final_cols = apply_safety_margin(
            table_name, required_cols, table_info.columns,
            question, hints,
            is_primary_table=is_primary,
            is_bridge_table=is_bridge,
            widened=widened,
        )

        # Sort in original DDL order for determinism
        kept_columns[table_name] = sorted(
            final_cols,
            key=lambda c: table_info.columns.index(c) if c in table_info.columns else 999
        )

    return kept_columns


# =============================================================================
# DDL Rendering
# =============================================================================

def render_compact_schema(
    schema_info: SchemaInfo,
    kept_tables: List[str],
    kept_columns: Dict[str, List[str]]
) -> str:
    """Render compact schema as multiline DDL, preserving exact identifiers."""
    ddl_statements = []

    for table_name in kept_tables:
        table_info = schema_info.tables[table_name]
        cols = kept_columns.get(table_name, table_info.columns)

        # Build column definitions
        col_defs = []
        for col in cols:
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
        def needs_quoting(name: str) -> bool:
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
        table_ddl += "\n)"

        ddl_statements.append(table_ddl)

    return "\n".join(ddl_statements)


def _find_column_def(raw_ddl: str, column_name: str) -> Optional[str]:
    """Find column definition in raw DDL, preserving exact formatting."""
    # Clean up DDL by removing inline comments
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
        col_for_pattern = column_name[1:-1]

    # Handle various quoting styles
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
            # Clean up whitespace
            col_def = ' '.join(col_def.split())
            return col_def

    return None


# =============================================================================
# Main Compaction Function
# =============================================================================

def compact_schema(
    full_schema: str,
    question: str,
    hints: Optional[str]
) -> CompactionResult:
    """
    Compact schema using question and hints only (NO gold SQL).
    
    Flow:
    1. Parse schema + extract identifiers from question/hints
    2. Build compact with normal margins
    3. If over-compacted → retry with widened margins
    4. If still over-compacted → fallback to full schema
    
    Returns CompactionResult with detailed per-example metadata.
    """
    original_len = len(full_schema)
    all_tables_in_schema: List[str] = []

    # --- Parse full schema ---
    try:
        schema_info = parse_schema(full_schema)
        all_tables_in_schema = list(schema_info.tables.keys())
    except Exception as e:
        return CompactionResult(
            compact_schema=full_schema,
            metadata=CompactionMetadata(
                original_schema_length=original_len,
                compact_schema_length=original_len,
                reduction_percent=0.0,
                primary_table=None,
                tables_kept=all_tables_in_schema,
                tables_dropped=[],
                compaction_status="fallback",
                fallback_reason=f"Schema parse error: {e}",
                pass_number=1,
            )
        )

    if not schema_info.tables:
        return CompactionResult(
            compact_schema=full_schema,
            metadata=CompactionMetadata(
                original_schema_length=original_len,
                compact_schema_length=original_len,
                reduction_percent=0.0,
                primary_table=None,
                tables_kept=[],
                tables_dropped=[],
                compaction_status="fallback",
                fallback_reason="No tables parsed from schema",
                pass_number=1,
            )
        )

    # --- Extract identifiers from question/hints ---
    all_tables, all_columns = get_all_schema_identifiers(schema_info)
    question_result = extract_question_identifiers(question, hints, schema_info)

    if not question_result.is_confident:
        return CompactionResult(
            compact_schema=full_schema,
            metadata=CompactionMetadata(
                original_schema_length=original_len,
                compact_schema_length=original_len,
                reduction_percent=0.0,
                primary_table=None,
                tables_kept=list(all_tables),
                tables_dropped=[],
                compaction_status="fallback",
                fallback_reason=question_result.fallback_reason,
                pass_number=1,
            )
        )

    # --- Identify required tables (matched + bridge) ---
    kept_tables_set = set(question_result.tables)
    bridge_tables = find_bridge_tables(kept_tables_set, schema_info.fk_graph, all_tables)
    kept_tables_set.update(bridge_tables)

    # Filter to tables that exist in schema
    kept_tables = [t for t in schema_info.tables.keys() if t in kept_tables_set]
    tables_dropped = [t for t in schema_info.tables.keys() if t not in kept_tables_set]

    if not kept_tables:
        return CompactionResult(
            compact_schema=full_schema,
            metadata=CompactionMetadata(
                original_schema_length=original_len,
                compact_schema_length=original_len,
                reduction_percent=0.0,
                primary_table=None,
                tables_kept=list(all_tables),
                tables_dropped=[],
                compaction_status="fallback",
                fallback_reason="No matching tables found",
                pass_number=1,
            )
        )

    # --- Determine primary table ---
    primary_table = find_primary_table(kept_tables, question_result)

    # ================================================================
    # ATTEMPT 1: Normal margins
    # ================================================================
    kept_columns = build_columns_for_tables(
        schema_info, kept_tables, question_result, question, hints,
        primary_table, bridge_tables, widened=False
    )
    compact_schema_text = render_compact_schema(schema_info, kept_tables, kept_columns)
    compact_len = len(compact_schema_text)
    reduction_pct = 100 * (1 - compact_len / original_len) if original_len > 0 else 0

    # Sanity: compact should be smaller
    if compact_len >= original_len:
        return CompactionResult(
            compact_schema=full_schema,
            metadata=CompactionMetadata(
                original_schema_length=original_len,
                compact_schema_length=original_len,
                reduction_percent=0.0,
                primary_table=primary_table,
                tables_kept=list(all_tables),
                tables_dropped=[],
                compaction_status="fallback",
                fallback_reason="Compact schema not smaller than original",
                pass_number=1,
            )
        )

    # Check over-compaction guards
    over_compacted, reason = check_over_compaction(
        compact_len, original_len, kept_tables, kept_columns, schema_info
    )

    if over_compacted:
        # ============================================================
        # ATTEMPT 2: Widened margins (widen-once-before-full-fallback)
        # ============================================================
        kept_columns = build_columns_for_tables(
            schema_info, kept_tables, question_result, question, hints,
            primary_table, bridge_tables, widened=True
        )
        compact_schema_text = render_compact_schema(schema_info, kept_tables, kept_columns)
        compact_len = len(compact_schema_text)
        reduction_pct = 100 * (1 - compact_len / original_len) if original_len > 0 else 0

        # Sanity: compact should still be smaller
        if compact_len >= original_len:
            return CompactionResult(
                compact_schema=full_schema,
                metadata=CompactionMetadata(
                    original_schema_length=original_len,
                    compact_schema_length=original_len,
                    reduction_percent=0.0,
                    primary_table=primary_table,
                    tables_kept=list(all_tables),
                    tables_dropped=[],
                    compaction_status="fallback",
                    fallback_reason="Widened compact schema not smaller than original",
                    pass_number=2,
                )
            )

        # Re-check guards after widening
        still_over, reason2 = check_over_compaction(
            compact_len, original_len, kept_tables, kept_columns, schema_info
        )

        if still_over:
            # ========================================================
            # FALLBACK: Full schema
            # ========================================================
            return CompactionResult(
                compact_schema=full_schema,
                metadata=CompactionMetadata(
                    original_schema_length=original_len,
                    compact_schema_length=original_len,
                    reduction_percent=0.0,
                    primary_table=primary_table,
                    tables_kept=list(all_tables),
                    tables_dropped=[],
                    compaction_status="fallback",
                    fallback_reason=f"Over-compacted after widening: {reason2}",
                    pass_number=2,
                ),
                kept_columns=kept_columns,
            )

        # Widened attempt succeeded
        return CompactionResult(
            compact_schema=compact_schema_text,
            metadata=CompactionMetadata(
                original_schema_length=original_len,
                compact_schema_length=compact_len,
                reduction_percent=round(reduction_pct, 1),
                primary_table=primary_table,
                tables_kept=kept_tables,
                tables_dropped=tables_dropped,
                compaction_status="widened",
                fallback_reason=None,
                pass_number=2,
            ),
            kept_columns=kept_columns,
        )

    # Normal attempt succeeded
    return CompactionResult(
        compact_schema=compact_schema_text,
        metadata=CompactionMetadata(
            original_schema_length=original_len,
            compact_schema_length=compact_len,
            reduction_percent=round(reduction_pct, 1),
            primary_table=primary_table,
            tables_kept=kept_tables,
            tables_dropped=tables_dropped,
            compaction_status="success",
            fallback_reason=None,
            pass_number=1,
        ),
        kept_columns=kept_columns,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def extract_schema_from_t10_prompt(user_content: str) -> str:
    """Extract schema section from T10 user prompt."""
    schema_match = re.search(r'Schema:\s*(.*?)(?=\n\nHints:)', user_content, re.DOTALL)
    if schema_match:
        return schema_match.group(1).strip()
    return ""


def extract_hints_from_t10_prompt(user_content: str) -> Optional[str]:
    """Extract hints section from T10 user prompt."""
    hints_match = re.search(r'Hints:\s*(.*?)(?=\n\nQuestion:)', user_content, re.DOTALL)
    if hints_match:
        hints = hints_match.group(1).strip()
        if hints.lower() == 'none':
            return None
        return hints
    return None


def extract_question_from_t10_prompt(user_content: str) -> str:
    """Extract question section from T10 user prompt."""
    q_match = re.search(r'Question:\s*(.*)', user_content, re.DOTALL)
    if q_match:
        return q_match.group(1).strip()
    return ""


def replace_schema_in_t10_prompt(user_content: str, new_schema: str) -> str:
    """Replace schema section in T10 user content."""
    schema_match = re.search(r'(Schema:\s*)(.*?)(\n\nHints:)', user_content, re.DOTALL)
    if schema_match:
        return (
            user_content[:schema_match.start()] +
            schema_match.group(1) + new_schema + schema_match.group(3) +
            user_content[schema_match.end():]
        )
    return user_content


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Compact Schema Module")
    print("=" * 60)
    
    # Test schema
    test_schema = """CREATE TABLE frpm
(
    CDSCode TEXT not null primary key,
    `County Name` TEXT null,
    `District Name` TEXT null,
    `School Name` TEXT null,
    `Free Meal Count (K-12)` INTEGER null,
    `Enrollment (K-12)` INTEGER null,
    `FRPM Count (K-12)` REAL null
)
CREATE TABLE schools
(
    CDSCode TEXT not null primary key,
    School TEXT null,
    MailStreet TEXT null,
    City TEXT null,
    State TEXT null,
    Zip TEXT null
)
CREATE TABLE satscores
(
    cds TEXT not null primary key,
    sname TEXT null,
    AvgScrMath INTEGER null,
    AvgScrRead INTEGER null,
    AvgScrWrite INTEGER null
)"""
    
    test_question = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
    test_hints = "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`"
    
    result = compact_schema(test_schema, test_question, test_hints)
    
    print(f"\nOriginal length: {result.metadata.original_schema_length}")
    print(f"Compact length: {result.metadata.compact_schema_length}")
    print(f"Reduction: {result.metadata.reduction_percent}%")
    print(f"Status: {result.metadata.compaction_status}")
    print(f"Primary table: {result.metadata.primary_table}")
    print(f"Tables kept: {result.metadata.tables_kept}")
    print(f"Tables dropped: {result.metadata.tables_dropped}")
    print(f"\nCompact schema:\n{result.compact_schema}")

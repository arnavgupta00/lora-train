from __future__ import annotations

import datetime as _dt
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .schema_spec import SchemaSpec


def build_ddl(spec: SchemaSpec) -> str:
    stmts: List[str] = []
    for t in spec.tables:
        col_defs: List[str] = []
        pk_cols = [c.name for c in t.columns if c.pk]
        handled_inline_pk = False
        for c in t.columns:
            if c.pk and c.name == "id" and c.type.upper().startswith("INT") and len(pk_cols) == 1:
                col_defs.append(f"{c.name} {c.type} PRIMARY KEY")
                handled_inline_pk = True
            else:
                col_defs.append(f"{c.name} {c.type}")
        if pk_cols and not handled_inline_pk:
            col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
        for fk in t.fks:
            col_defs.append(f"FOREIGN KEY ({fk.column}) REFERENCES {fk.ref_table}({fk.ref_column})")
        stmts.append(f"CREATE TABLE {t.name} (\n  " + ",\n  ".join(col_defs) + "\n);")
    return "\n\n".join(stmts)


@dataclass
class BuiltSchema:
    spec: SchemaSpec
    ddl_sql: str
    conn: sqlite3.Connection


def _rand_iso_date(start: _dt.date, end: _dt.date) -> str:
    days = (end - start).days
    if days <= 0:
        return start.isoformat()
    offset = random.randint(0, days)
    return (start + _dt.timedelta(days=offset)).isoformat()


def _col_value(col_name: str, col_type: str, enum: Optional[List[str]], semantic_type: Optional[str], row_idx: int) -> object:
    if enum:
        return enum[row_idx % len(enum)]
    st = (semantic_type or "").lower()
    if st in ("datetime", "date", "timestamp"):
        today = _dt.date.today()
        start = today - _dt.timedelta(days=365)
        return _rand_iso_date(start, today)
    if col_type.upper().startswith("INT"):
        return row_idx + 1
    if col_type.upper().startswith("REAL") or col_type.upper().startswith("NUM"):
        return float((row_idx % 97) + 1) * 1.13
    # TEXT default
    if "email" in col_name.lower():
        return f"user_{row_idx}@example.com"
    if "name" in col_name.lower():
        return f"{col_name}_{row_idx}"
    return f"{col_name}_{row_idx}"


def seed_sqlite(spec: SchemaSpec, conn: sqlite3.Connection) -> None:
    # Very simple deterministic seeding: insert rows_per_table into each table.
    # For FK columns, we reference earlier tables by picking a valid id in range.
    conn.execute("PRAGMA foreign_keys = ON;")
    rows_per = spec.seed_profile.rows_per_table

    # Determine insert order: assume tables are listed roughly parent->child.
    for t in spec.tables:
        cols = [c for c in t.columns]
        col_names = [c.name for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO {t.name} ({', '.join(col_names)}) VALUES ({placeholders})"

        # Precompute FK column indices
        fk_by_col: Dict[str, Tuple[str, str]] = {fk.column: (fk.ref_table, fk.ref_column) for fk in t.fks}

        for i in range(rows_per):
            values: List[object] = []
            for c in cols:
                if c.name in fk_by_col:
                    # reference id from parent table (assume integer ids 1..rows_per)
                    values.append((i % rows_per) + 1)
                else:
                    # If single-column pk named id, keep it stable 1..N
                    if c.pk and c.name == "id" and c.type.upper().startswith("INT"):
                        values.append(i + 1)
                    else:
                        values.append(_col_value(c.name, c.type, c.enum, c.semantic_type, i))
            conn.execute(sql, values)
    conn.commit()


def build_sqlite_schema(spec: SchemaSpec) -> BuiltSchema:
    ddl = build_ddl(spec)
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    seed_sqlite(spec, conn)
    return BuiltSchema(spec=spec, ddl_sql=ddl, conn=conn)

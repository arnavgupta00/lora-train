from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .schema_spec import SchemaSpec, TableSpec


@dataclass(frozen=True)
class TemplateExample:
    schema_id: str
    sql: str
    tags: List[str]
    output_contract_hint: str


def _table_by_name(spec: SchemaSpec) -> Dict[str, TableSpec]:
    return {t.name: t for t in spec.tables}


def _pick_numeric_cols(t: TableSpec) -> List[str]:
    out = []
    for c in t.columns:
        typ = c.type.upper()
        if typ.startswith("REAL") or typ.startswith("NUM"):
            out.append(c.name)
        if typ.startswith("INT") and c.name.endswith(("_count", "_qty", "_units")):
            out.append(c.name)
    return out


def _pick_text_enum_cols(t: TableSpec) -> List[Tuple[str, Sequence[str]]]:
    out = []
    for c in t.columns:
        if c.enum and c.type.upper().startswith("TEXT"):
            out.append((c.name, c.enum))
    return out


def _pick_date_cols(t: TableSpec) -> List[str]:
    out = []
    for c in t.columns:
        if (c.semantic_type or "").lower() in ("date", "datetime", "timestamp") or c.name.endswith(("_at", "_date")):
            out.append(c.name)
    return out


def _has_col(t: TableSpec, name: str) -> bool:
    return any(c.name == name for c in t.columns)


def _select_one(conn: sqlite3.Connection, sql: str) -> Optional[object]:
    cur = conn.execute(sql)
    row = cur.fetchone()
    return row[0] if row else None


def _safe_identifier(name: str) -> str:
    # Specs should already be SQL-safe. Keep this in case.
    return "".join(ch for ch in name if ch.isalnum() or ch == "_")


def generate_templates(spec: SchemaSpec, conn: sqlite3.Connection, target: int = 50, seed: int = 42) -> List[TemplateExample]:
    random.seed(seed)
    tables = spec.tables
    t_by = _table_by_name(spec)
    examples: List[TemplateExample] = []

    # Basic single-table families
    for t in tables:
        if len(examples) >= target:
            break

        # 1) Count by enum filter
        enums = _pick_text_enum_cols(t)
        if enums:
            col, vals = random.choice(enums)
            v = random.choice(list(vals))
            sql = f"SELECT COUNT(*) AS row_count FROM {_safe_identifier(t.name)} WHERE {_safe_identifier(col)} = '{v}'"
            examples.append(
                TemplateExample(
                    schema_id=spec.schema_id,
                    sql=sql,
                    tags=["filter", "count"],
                    output_contract_hint="Return a single integer count.",
                )
            )

        # 2) Top-k by numeric measure with GROUP BY (if has a name-ish column)
        nums = _pick_numeric_cols(t)
        if nums:
            num = random.choice(nums)
            group_col = None
            for cand in ("name", "status", "category", "type"):
                if _has_col(t, cand):
                    group_col = cand
                    break
            if group_col:
                sql = (
                    f"SELECT {_safe_identifier(group_col)}, SUM({_safe_identifier(num)}) AS total_value "
                    f"FROM {_safe_identifier(t.name)} "
                    f"GROUP BY {_safe_identifier(group_col)} "
                    f"ORDER BY total_value DESC, {_safe_identifier(group_col)} ASC LIMIT 5"
                )
                examples.append(
                    TemplateExample(
                        schema_id=spec.schema_id,
                        sql=sql,
                        tags=["aggregation", "group_by", "ranking"],
                        output_contract_hint=f"Return {group_col} and total_value, sorted by total_value desc.",
                    )
                )

        # 3) Month trend if date column exists
        dates = _pick_date_cols(t)
        if dates:
            d = random.choice(dates)
            sql = (
                f"SELECT strftime('%Y-%m', {_safe_identifier(d)}) AS month, COUNT(*) AS row_count "
                f"FROM {_safe_identifier(t.name)} "
                f"GROUP BY strftime('%Y-%m', {_safe_identifier(d)}) "
                f"ORDER BY month ASC"
            )
            examples.append(
                TemplateExample(
                    schema_id=spec.schema_id,
                    sql=sql,
                    tags=["time", "aggregation", "trend"],
                    output_contract_hint="Return month and row_count ordered by month ascending.",
                )
            )

    # FK join families (many-to-one)
    for t in tables:
        if len(examples) >= target:
            break
        if not t.fks:
            continue
        fk = random.choice(t.fks)
        parent = t_by.get(fk.ref_table)
        if not parent:
            continue

        # Choose a parent grouping column
        parent_group = None
        for cand in ("name", "org_name", "title", "category", "segment"):
            if _has_col(parent, cand):
                parent_group = cand
                break
        if not parent_group:
            continue

        # Count children per parent, top 10
        sql = (
            f"SELECT p.{_safe_identifier(parent_group)} AS parent_name, COUNT(c.id) AS child_count "
            f"FROM {_safe_identifier(parent.name)} p "
            f"JOIN {_safe_identifier(t.name)} c ON c.{_safe_identifier(fk.column)} = p.{_safe_identifier(fk.ref_column)} "
            f"GROUP BY p.{_safe_identifier(parent_group)} "
            f"ORDER BY child_count DESC, parent_name ASC LIMIT 10"
        )
        examples.append(
            TemplateExample(
                schema_id=spec.schema_id,
                sql=sql,
                tags=["join", "aggregation", "ranking"],
                output_contract_hint="Return parent_name and child_count sorted by child_count desc.",
            )
        )

    # Ensure unique SQL strings
    uniq: Dict[str, TemplateExample] = {}
    for ex in examples:
        uniq.setdefault(ex.sql, ex)
    out = list(uniq.values())[:target]
    return out


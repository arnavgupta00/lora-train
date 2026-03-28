from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    pk: bool = False
    enum: Optional[List[str]] = None
    semantic_type: Optional[str] = None


@dataclass(frozen=True)
class ForeignKeySpec:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: List[ColumnSpec]
    fks: List[ForeignKeySpec]


@dataclass(frozen=True)
class SeedProfile:
    rows_per_table: int = 200
    datetime_range_days: int = 365


@dataclass(frozen=True)
class SchemaSpec:
    schema_id: str
    description: str
    dialect: str
    tables: List[TableSpec]
    business_rules: List[str]
    seed_profile: SeedProfile


def parse_schema_spec(obj: Dict[str, Any]) -> SchemaSpec:
    tables: List[TableSpec] = []
    for t in obj["tables"]:
        cols = [
            ColumnSpec(
                name=c["name"],
                type=c["type"],
                pk=bool(c.get("pk", False)),
                enum=c.get("enum"),
                semantic_type=c.get("semantic_type"),
            )
            for c in t["columns"]
        ]
        fks = [
            ForeignKeySpec(column=fk["column"], ref_table=fk["ref_table"], ref_column=fk["ref_column"])
            for fk in t.get("fks", [])
        ]
        tables.append(TableSpec(name=t["name"], columns=cols, fks=fks))

    sp = obj.get("seed_profile") or {}
    seed_profile = SeedProfile(
        rows_per_table=int(sp.get("rows_per_table", 200)),
        datetime_range_days=int(sp.get("datetime_range_days", 365)),
    )

    return SchemaSpec(
        schema_id=obj["schema_id"],
        description=obj.get("description", ""),
        dialect=obj.get("dialect", "sqlite"),
        tables=tables,
        business_rules=list(obj.get("business_rules") or []),
        seed_profile=seed_profile,
    )


def schema_context(spec: SchemaSpec) -> str:
    lines: List[str] = []
    for t in spec.tables:
        cols = ", ".join(c.name for c in t.columns)
        lines.append(f"{t.name}({cols})")
    return "\n".join(lines)


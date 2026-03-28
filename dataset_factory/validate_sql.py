from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


READONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    s = sql.strip()
    s = s.split(";")[0].strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class Validation:
    accepted: bool
    parse_ok: bool
    execution_ok: bool
    deterministic_ok: bool
    result_hash: Optional[str]
    error: Optional[str]


def _hash_rows(rows: List[Tuple[Any, ...]]) -> str:
    data = json.dumps(rows, ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_sql(conn: sqlite3.Connection, sql: str) -> Validation:
    sql_n = normalize_sql(sql)
    if not READONLY_RE.match(sql_n):
        return Validation(False, True, False, False, None, "not_read_only")
    try:
        cur = conn.execute(sql_n)
        rows1 = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        return Validation(False, True, False, False, None, f"exec_error:{e}")

    # Determinism check: run twice and compare exact rows
    try:
        cur2 = conn.execute(sql_n)
        rows2 = cur2.fetchall()
    except Exception as e:  # noqa: BLE001
        return Validation(False, True, False, False, None, f"exec_error:{e}")

    if rows1 != rows2:
        return Validation(False, True, True, False, None, "nondeterministic")

    rh = _hash_rows(rows1)
    return Validation(True, True, True, True, rh, None)


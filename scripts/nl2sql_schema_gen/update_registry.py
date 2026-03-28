import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--schemas", required=True, help="stage01_generated_schemas.json")
    ap.add_argument("--db_map", required=True, help="json mapping schema_id -> {db_name, db_id, binding}")
    args = ap.parse_args()

    reg_path = Path(args.registry)
    schema_path = Path(args.schemas)
    db_map_path = Path(args.db_map)

    reg = _load(reg_path, {"schemas": []})
    existing = {s.get("schema_id") for s in reg.get("schemas", []) if isinstance(s, dict)}

    gen = _load(schema_path, {"schemas": []})
    db_map = _load(db_map_path, {})

    added = 0
    for s in gen.get("schemas", []):
        sid = s.get("schema_id")
        if not sid or sid in existing:
            continue
        meta = db_map.get(sid) or {}
        reg["schemas"].append(
            {
                "schema_id": sid,
                "version": s.get("version", "v1"),
                "dialect": s.get("dialect", "sqlite"),
                "validation_binding": meta.get("binding") or s.get("validation_binding"),
                "db_name": meta.get("db_name"),
                "db_id": meta.get("db_id"),
                "description": s.get("description"),
                "source": "generated",
            }
        )
        existing.add(sid)
        added += 1

    _save(reg_path, reg)
    print(f"registry={reg_path} added={added} total={len(reg.get('schemas', []))}")


if __name__ == "__main__":
    main()

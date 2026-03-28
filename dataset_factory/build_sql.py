from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dataset_factory.schema_spec import parse_schema_spec, schema_context
from dataset_factory.sqlite_builder import build_sqlite_schema
from dataset_factory.sql_templates import generate_templates
from dataset_factory.validate_sql import validate_sql


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True, help="Directory with schema spec JSON files")
    ap.add_argument("--out_dir", required=True, help="Build output directory")
    ap.add_argument("--templates_per_schema", type=int, default=50)
    args = ap.parse_args()

    spec_dir = Path(args.spec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(spec_dir.glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        spec = parse_schema_spec(obj)
        built = build_sqlite_schema(spec)

        templates = generate_templates(spec, built.conn, target=args.templates_per_schema)
        accepted: List[Dict[str, Any]] = []
        for i, ex in enumerate(templates):
            v = validate_sql(built.conn, ex.sql)
            if not v.accepted:
                continue
            accepted.append(
                {
                    "id": f"{spec.schema_id}:{i}",
                    "schema_id": spec.schema_id,
                    "schema_context": schema_context(spec),
                    "business_rules": spec.business_rules,
                    "sql": ex.sql,
                    "tags": ex.tags,
                    "output_contract_hint": ex.output_contract_hint,
                }
            )

        schema_out = out_dir / f"{spec.schema_id}.build.json"
        schema_out.write_text(json.dumps({"schema": obj, "accepted_templates": accepted}, indent=2), encoding="utf-8")
        print(f"{spec.schema_id}: accepted {len(accepted)}/{len(templates)} templates -> {schema_out}")


if __name__ == "__main__":
    main()


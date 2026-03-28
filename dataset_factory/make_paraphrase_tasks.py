from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=25)
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    out_path = Path(args.out)

    items: List[Dict[str, Any]] = []
    for p in sorted(build_dir.glob("*.build.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        for ex in obj.get("accepted_templates", []):
            items.append(
                {
                    "id": ex["id"],
                    "schema_id": ex["schema_id"],
                    "schema_context": ex["schema_context"],
                    "business_rules": ex["business_rules"],
                    "sql": ex["sql"],
                    "output_contract_hint": ex["output_contract_hint"],
                }
            )

    # Chunk into batches that you can paste into Copilot.
    batches: List[Dict[str, Any]] = []
    for i in range(0, len(items), args.batch_size):
        batches.append({"batch_id": f"batch_{i//args.batch_size:04d}", "items": items[i : i + args.batch_size]})

    out_path.write_text(json.dumps({"batches": batches}, indent=2), encoding="utf-8")
    print(f"wrote {len(batches)} batches -> {out_path}")


if __name__ == "__main__":
    main()


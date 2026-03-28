from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


SYSTEM_PROMPT = "You are a sqlite SQL generator. Return only SQL."


def _load_build(build_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in build_dir.glob("*.build.json"):
        obj = json.loads(p.read_text(encoding="utf-8"))
        for ex in obj.get("accepted_templates", []):
            out[ex["id"]] = ex
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_dir", required=True)
    ap.add_argument("--paraphrases", required=True, help="Copilot JSON output containing items[id]->questions")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--paraphrases_per_sql", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--schema_holdout_fraction", type=float, default=0.2, help="fraction of schemas held out for test")
    args = ap.parse_args()

    random.seed(args.seed)
    build_dir = Path(args.build_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build = _load_build(build_dir)
    para_obj = json.loads(Path(args.paraphrases).read_text(encoding="utf-8"))
    para_items = {it["id"]: it["questions"] for it in para_obj.get("items", [])}

    # Group schema IDs
    schema_ids = sorted({ex["schema_id"] for ex in build.values()})
    random.shuffle(schema_ids)
    holdout_n = max(1, int(len(schema_ids) * args.schema_holdout_fraction))
    test_schemas = set(schema_ids[:holdout_n])
    dev_schemas = set(schema_ids[holdout_n : holdout_n + max(1, holdout_n // 2)])

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    for ex_id, ex in build.items():
        qs = para_items.get(ex_id)
        if not qs:
            continue
        qs = qs[: args.paraphrases_per_sql]
        user_prefix = (
            "Schema:\n"
            + ex["schema_context"]
            + "\n\nBusiness rules:\n"
            + "\n".join(ex.get("business_rules") or [])
            + "\n\nQuestion:\n"
        )
        for q in qs:
            row = {
                "type": "chatml",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prefix + q.strip()},
                    {"role": "assistant", "content": ex["sql"]},
                ],
                "source": "dataset_factory",
            }
            schema_id = ex["schema_id"]
            if schema_id in test_schemas:
                splits["test"].append(row)
            elif schema_id in dev_schemas:
                splits["dev"].append(row)
            else:
                splits["train"].append(row)

    for split, rows in splits.items():
        out_path = out_dir / f"{split}.qwen.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        print(split, len(rows), "->", out_path)


if __name__ == "__main__":
    main()


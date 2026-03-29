import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import urllib.request


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _post_json(url: str, body: Dict[str, Any], admin_key: str) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {admin_key}",
            "User-Agent": "curl/8.0",
        },
    )
    with urllib.request.urlopen(req) as resp:  # nosec - internal tool
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=50)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_url = cfg["dataset_service"]["base_url"].rstrip("/")
    admin_env = cfg["dataset_service"]["admin_api_key_env"]
    admin_key = os.environ.get(admin_env, "").strip()
    if not admin_key:
        raise SystemExit(f"Missing env {admin_env}")

    plan = _load_json(Path(args.plan))
    gen = _load_json(Path(args.examples))
    examples = gen.get("examples", [])
    if not isinstance(examples, list) or not examples:
        raise SystemExit("examples file missing examples[]")

    # Required accepted counts per (schema_id, split)
    required: Dict[Tuple[str, str], int] = {}
    for sid, meta in (plan.get("by_schema") or {}).items():
        deficit = meta.get("deficit") or {}
        for split in ("train", "dev", "test"):
            need = int(deficit.get(split, 0))
            if need > 0:
                required[(sid, split)] = need

    grouped: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        sid = ex.get("schema_id")
        split = ex.get("split")
        if not isinstance(sid, str) or not isinstance(split, str):
            continue
        if (sid, split) not in required:
            continue
        grouped[(sid, split)].append(ex)

    report: Dict[str, Any] = {
        "base_url": base_url,
        "accepted": {},
        "rejected": {},
        "attempted": {},
        "errors_sample": {},
    }

    for key, need in required.items():
        sid, split = key
        pool = grouped.get(key, [])
        if not pool:
            report["attempted"][f"{sid}:{split}"] = 0
            report["accepted"][f"{sid}:{split}"] = 0
            report["rejected"][f"{sid}:{split}"] = 0
            report["errors_sample"][f"{sid}:{split}"] = ["missing examples for this schema/split in generator output"]
            continue

        accepted = 0
        rejected = 0
        attempted = 0
        errors: List[str] = []

        i = 0
        while i < len(pool) and accepted < need:
            chunk = pool[i : i + int(args.chunk_size)]
            i += int(args.chunk_size)
            resp = _post_json(f"{base_url}/v1/examples/batch", {"examples": chunk}, admin_key)
            results = resp.get("results", [])
            if not isinstance(results, list):
                raise SystemExit(f"unexpected response: {resp}")
            for r in results:
                attempted += 1
                ok = bool(r.get("accepted"))
                if ok:
                    accepted += 1
                    if accepted >= need:
                        # Stop early; do not overshoot target for this schema/split.
                        break
                else:
                    rejected += 1
                    err = r.get("error") or (r.get("validation") or {}).get("error_message") or (r.get("validation") or {}).get("error_code")
                    if err and len(errors) < 20:
                        errors.append(str(err))
            if accepted >= need:
                break

        report["attempted"][f"{sid}:{split}"] = attempted
        report["accepted"][f"{sid}:{split}"] = accepted
        report["rejected"][f"{sid}:{split}"] = rejected
        report["errors_sample"][f"{sid}:{split}"] = errors

    _save_json(Path(args.out), report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()


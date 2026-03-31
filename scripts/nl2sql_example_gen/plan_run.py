import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import urllib.request


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req) as resp:  # nosec - internal tool
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, body: Dict[str, Any], admin_key: str) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "User-Agent": "curl/8.0",
            "Authorization": f"Bearer {admin_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:  # nosec - internal tool
        return json.loads(resp.read().decode("utf-8"))


def _context_tables(schema_context: Any) -> Set[str]:
    if not isinstance(schema_context, str):
        return set()
    out: Set[str] = set()
    for line in schema_context.splitlines():
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\)\s*$", line.strip())
        if m:
            out.add(m.group(1))
    return out


def _runtime_tables(base_url: str, admin_key: str, schema_id: str, schema_version: str, dialect: str) -> Set[str]:
    body = {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "dialect": dialect,
        "split": "train",
        "question": "List all table names.",
        "gold_sql": "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        "tags": ["probe"],
        "metadata": {"source": "probe"},
    }
    resp = _post_json(f"{base_url}/v1/validate", body, admin_key)
    rows = ((resp or {}).get("validation") or {}).get("result_preview", {}).get("rows", [])
    out: Set[str] = set()
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                name = r.get("name")
                if isinstance(name, str) and name and not name.startswith("sqlite_") and name != "_cf_KV":
                    out.add(name)
    return out


def _targets(total: int, ratio: Dict[str, int]) -> Dict[str, int]:
    keys = ["train", "dev", "test"]
    rsum = sum(int(ratio[k]) for k in keys)
    raw = {k: total * (ratio[k] / rsum) for k in keys}
    base = {k: int(math.floor(raw[k])) for k in keys}
    remainder = total - sum(base.values())
    fracs = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
    for i in range(remainder):
        base[fracs[i % len(fracs)]] += 1
    return base


def _count_examples(examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    # Count only rows that are meaningfully "available":
    # - train/dev: accepted only
    # - test: pending_review or accepted
    out: Dict[str, Dict[str, int]] = {}
    for ex in examples:
        sid = ex.get("schema_id")
        split = ex.get("split")
        status = ex.get("validation_status")
        if not isinstance(sid, str) or not isinstance(split, str) or not isinstance(status, str):
            continue
        if split not in ("train", "dev", "test"):
            continue
        ok = False
        if split in ("train", "dev"):
            ok = status == "accepted"
        else:
            ok = status in ("pending_review", "accepted")
        if not ok:
            continue
        out.setdefault(sid, {}).setdefault(split, 0)
        out[sid][split] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--schema_ids_from_plan", help="If set, only recount schemas from this stage00_plan.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_url = cfg["dataset_service"]["base_url"].rstrip("/")
    admin_env = cfg["dataset_service"].get("admin_api_key_env", "ADMIN_API_KEY")
    admin_key = os.environ.get(admin_env, "").strip()
    runs_dir = Path(cfg["state"]["runs_dir"])
    run_name = cfg["run_name"]
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    schemas_payload = _fetch_json(f"{base_url}/v1/schemas")
    schemas = schemas_payload.get("schemas", schemas_payload) if isinstance(schemas_payload, dict) else schemas_payload
    if not isinstance(schemas, list):
        raise SystemExit("Unexpected /v1/schemas response shape")

    examples_payload = _fetch_json(f"{base_url}/v1/examples")
    examples = examples_payload.get("examples", examples_payload) if isinstance(examples_payload, dict) else examples_payload
    if not isinstance(examples, list):
        raise SystemExit("Unexpected /v1/examples response shape")

    counts_by_schema = _count_examples(examples)

    default_ver = cfg["defaults"].get("schema_version", "v1")
    total_target = int(cfg["target_total_per_schema"])
    ratio = cfg["split_ratio"]
    split_targets = _targets(total_target, ratio)

    if args.schema_ids_from_plan:
        prior = _load_json(Path(args.schema_ids_from_plan), {})
        only_ids = set(prior.get("selected_schemas", []))
    else:
        only_ids = None

    rows: List[Dict[str, Any]] = []
    for s in schemas:
        if not isinstance(s, dict):
            continue
        sid = s.get("id") or s.get("schema_id")
        ver = s.get("version") or s.get("schema_version") or default_ver
        status = s.get("status")
        if not isinstance(sid, str) or not isinstance(ver, str):
            continue
        if only_ids is not None and sid not in only_ids:
            continue
        if status not in (None, "active"):
            continue
        c = counts_by_schema.get(sid, {})
        current = {k: int(c.get(k, 0)) for k in ("train", "dev", "test")}
        target = split_targets
        deficit = {k: max(0, int(target[k]) - int(current[k])) for k in ("train", "dev", "test")}
        total_current = sum(current.values())
        total_deficit = sum(deficit.values())
        rows.append(
            {
                "schema_id": sid,
                "schema_version": ver,
                "dialect": s.get("dialect", cfg["defaults"].get("dialect", "sqlite")),
                "current": current,
                "target": target,
                "deficit": deficit,
                "total_current": total_current,
                "total_deficit": total_deficit,
                "schema_context": s.get("schema_context"),
                "business_rules": s.get("business_rules"),
            }
        )

    rows.sort(key=lambda r: (r["total_deficit"], r["schema_id"]), reverse=True)
    cap = int(cfg["schemas_per_run"])
    selected: List[Dict[str, Any]] = []
    skipped_binding_mismatch: Dict[str, Any] = {}
    for r in rows:
        if r["total_deficit"] <= 0:
            continue
        if len(selected) >= cap:
            break
        # Guard against schemas whose runtime validation binding doesn't match schema_context.
        if admin_key:
            ctx_tables = _context_tables(r.get("schema_context"))
            if not ctx_tables:
                skipped_binding_mismatch[r["schema_id"]] = {"error": "missing_schema_context"}
                continue
            else:
                try:
                    rt_tables = _runtime_tables(base_url, admin_key, r["schema_id"], r["schema_version"], r["dialect"])
                    overlap = len(ctx_tables & rt_tables)
                    ratio = overlap / max(1, len(ctx_tables))
                    if not rt_tables or ratio < 0.2:
                        skipped_binding_mismatch[r["schema_id"]] = {
                            "context_tables": len(ctx_tables),
                            "runtime_tables": len(rt_tables),
                            "overlap": overlap,
                            "overlap_ratio": round(ratio, 3),
                        }
                        continue
                except Exception:
                    skipped_binding_mismatch[r["schema_id"]] = {"error": "runtime_probe_failed"}
                    continue
        selected.append(r)

    plan = {
        "run_name": run_name,
        "schemas_per_run": int(cfg["schemas_per_run"]),
        "target_total_per_schema": total_target,
        "split_ratio": ratio,
        "split_targets": split_targets,
        "selected_schemas": [r["schema_id"] for r in selected],
        "skipped_binding_mismatch": skipped_binding_mismatch,
        "by_schema": {
            r["schema_id"]: {
                "schema_version": r["schema_version"],
                "current": r["current"],
                "target": r["target"],
                "deficit": r["deficit"],
                "total_current": r["total_current"],
                "total_deficit": r["total_deficit"],
            }
            for r in selected
        },
    }

    schemas_out = {
        "schemas": [
            {
                "schema_id": r["schema_id"],
                "schema_version": r["schema_version"],
                "dialect": r["dialect"],
                "schema_context": r["schema_context"],
                "business_rules": r["business_rules"],
            }
            for r in selected
        ]
    }

    _save_json(run_dir / "stage00_plan.json", plan)
    _save_json(run_dir / "stage00_schemas.json", schemas_out)
    print(f"run_dir={run_dir} selected={len(selected)}")


if __name__ == "__main__":
    main()

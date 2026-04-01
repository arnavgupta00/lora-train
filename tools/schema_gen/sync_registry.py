import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import urllib.request


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _fetch_schemas(base_url: str) -> List[Dict[str, Any]]:
    req = urllib.request.Request(f"{base_url}/v1/schemas", headers={"User-Agent": "curl/8.0"})
    # Some Cloudflare configs block default Python user-agents; curl UA is reliably allowed.
    with urllib.request.urlopen(req) as resp:  # nosec - internal tool
        data = json.loads(resp.read().decode("utf-8"))
    # service returns {schemas:[...]} in some versions, or [...] in others
    if isinstance(data, dict) and "schemas" in data:
        return data["schemas"]
    if isinstance(data, list):
        return data
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    reg_path = Path(cfg["state"]["schema_registry_path"])
    base_url = cfg["dataset_service"]["base_url"]

    reg = _load_json(reg_path, {"schemas": []})
    existing = {s.get("schema_id") for s in reg.get("schemas", []) if isinstance(s, dict)}

    remote = _fetch_schemas(base_url)
    added = 0
    for s in remote:
        sid = s.get("schema_id") or s.get("id")
        if not sid or sid in existing:
            continue
        reg["schemas"].append(
            {
                "schema_id": sid,
                "version": s.get("version") or s.get("schema_version") or "v1",
                "dialect": s.get("dialect") or "sqlite",
                "validation_binding": s.get("validation_binding"),
                "description": s.get("description"),
                "source": "remote",
            }
        )
        existing.add(sid)
        added += 1

    _save_json(reg_path, reg)
    print(f"registry={reg_path} total={len(reg.get('schemas', []))} added_from_service={added}")


if __name__ == "__main__":
    main()

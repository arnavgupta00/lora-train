import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import urllib.request


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _upper_snake(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()


def _run(cmd: List[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}\nOutput:\n{p.stdout}")
    return p.stdout


def _wrangler_list_dbs(project_dir: Path) -> Dict[str, str]:
    """List existing D1 databases. Returns {db_name: db_uuid}."""
    out = _run(["npx", "wrangler", "d1", "list", "--json"], cwd=project_dir)
    dbs = json.loads(out)
    return {d["name"]: d["uuid"] for d in dbs}


def _wrangler_create_db(project_dir: Path, db_name: str, existing_dbs: Dict[str, str] | None = None) -> str:
    # Check if DB already exists
    if existing_dbs is not None and db_name in existing_dbs:
        print(f"  [skip] DB {db_name} already exists (id={existing_dbs[db_name][:8]}...)")
        return existing_dbs[db_name]
    out = _run(["npx", "wrangler", "d1", "create", db_name], cwd=project_dir)
    # Try JSON parse first (older wrangler versions)
    try:
        data = json.loads(out)
        if isinstance(data, dict) and "database_id" in data:
            return data["database_id"]
        if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict) and "uuid" in data["result"]:
            return data["result"]["uuid"]
    except json.JSONDecodeError:
        pass
    # Parse text output: look for database_id = "UUID"
    match = re.search(r'database_id\s*=\s*"([^"]+)"', out)
    if match:
        return match.group(1)
    raise SystemExit(f"Could not parse database_id from wrangler output: {out[:400]}")


def _wrangler_seed_db(project_dir: Path, db_name: str, seed_sql: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sql", delete=False) as f:
        f.write(seed_sql)
        tmp = f.name
    try:
        _run(["npx", "wrangler", "d1", "execute", db_name, "--file", tmp, "--remote", "-y"], cwd=project_dir)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _http_json(method: str, url: str, body: Dict[str, Any], admin_key: str) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {admin_key}",
            "User-Agent": "curl/8.0",
        },
    )
    with urllib.request.urlopen(req) as resp:  # nosec - internal tool
        return json.loads(resp.read().decode("utf-8"))


def _first_table_from_schema_context(schema_context: str) -> str:
    for line in (schema_context or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            return m.group(1)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--schemas", required=True, help="stage01_generated_schemas.json")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dry_run", action="store_true", help="Print actions without running wrangler/deploy/api writes")
    args = ap.parse_args()

    cfg = _load_json(Path(args.config))
    run_dir = Path(args.run_dir)
    schemas_path = Path(args.schemas)
    _assert(schemas_path.exists(), f"missing schemas file: {schemas_path}")

    schema_defaults = cfg.get("schema_defaults") or {}
    default_version = cfg.get("schema_version") or schema_defaults.get("schema_version") or "v1"
    default_dialect = cfg.get("dialect") or schema_defaults.get("dialect") or "sqlite"

    admin_env = cfg["dataset_service"]["admin_api_key_env"]
    admin_key = os.environ.get(admin_env, "").strip()
    _assert(admin_key or args.dry_run, f"Missing env {admin_env} (required to register schemas)")

    project_dir = Path(cfg["cloudflare"]["wrangler_project_dir"]).resolve()
    wrangler_toml = Path(cfg["cloudflare"]["wrangler_toml_path"]).resolve()
    base_url = cfg["dataset_service"]["base_url"].rstrip("/")
    prefix = cfg["cloudflare"]["d1_db_name_prefix"]
    deploy_cmd = cfg["cloudflare"]["deploy_command"].split()

    reg_path = Path(cfg["state"]["schema_registry_path"]).resolve()
    reg = {"schemas": []}
    if reg_path.exists():
        reg = _load_json(reg_path)
    existing = {s.get("schema_id") for s in reg.get("schemas", []) if isinstance(s, dict)}

    gen = _load_json(schemas_path)
    schemas = gen.get("schemas", [])
    _assert(isinstance(schemas, list) and schemas, "schemas file has no schemas[]")

    # Validate uniqueness and no duplicates.
    seen: set[str] = set()
    for s in schemas:
        sid = s.get("schema_id")
        _assert(isinstance(sid, str) and sid, "schema missing schema_id")
        _assert(sid not in existing, f"duplicate schema_id already in registry: {sid}")
        _assert(sid not in seen, f"duplicate schema_id within stage01 file: {sid}")
        seen.add(sid)
        for req in ("schema_sql", "seed_sql", "schema_context", "business_rules", "validation_binding"):
            _assert(req in s and s[req], f"{sid}: missing {req}")

    db_map: Dict[str, Dict[str, str]] = {}
    db_map_path = run_dir / "db_map.json"
    
    # Resume from partial run if db_map exists
    if db_map_path.exists():
        db_map = _load_json(db_map_path)
        print(f"Resuming from partial run ({len(db_map)} schemas already processed)")

    # List existing D1 databases to allow re-use
    existing_dbs = _wrangler_list_dbs(project_dir) if not args.dry_run else {}

    for s in schemas:
        sid = s["schema_id"]
        
        # Skip if already processed
        if sid in db_map:
            print(f"  [skip] {sid} already processed")
            continue
            
        version = s.get("version") or default_version
        db_name = f"{prefix}{sid}_{version}"
        binding = s.get("validation_binding") or f"DYNAMIC_{_upper_snake(sid)}_{_upper_snake(version)}_DB"

        if args.dry_run:
            print(f"[dry_run] create db {db_name}, binding={binding}")
            db_id = "DRY_RUN_DB_ID"
        else:
            db_already_exists = db_name in existing_dbs
            db_id = _wrangler_create_db(project_dir, db_name, existing_dbs)
            
            # Only seed if DB was just created (not already existing)
            if not db_already_exists:
                _wrangler_seed_db(project_dir, db_name, s["seed_sql"])

            # Patch wrangler.toml to include binding.
            _run(
                [
                    "python3",
                    str(Path("scripts/nl2sql_schema_gen/patch_wrangler_toml.py").resolve()),
                    "--wrangler_toml",
                    str(wrangler_toml),
                    "--binding",
                    binding,
                    "--db_name",
                    db_name,
                    "--db_id",
                    db_id,
                ],
                cwd=Path.cwd(),
            )

        db_map[sid] = {"db_name": db_name, "db_id": db_id, "binding": binding}
        
        # Save progress after each schema for resumability
        _save_json(db_map_path, db_map)

    if args.dry_run:
        print("[dry_run] deploy worker")
    else:
        _run(deploy_cmd, cwd=project_dir)

    # Register + activate schemas
    for s in schemas:
        sid = s["schema_id"]
        version = s.get("version") or default_version
        payload = {
            "schema_id": sid,
            "description": s.get("description", ""),
            "version": version,
            "dialect": s.get("dialect") or default_dialect,
            "validation_binding": db_map[sid]["binding"],
            "schema_sql": s.get("schema_sql"),
            "schema_context": s.get("schema_context"),
            "business_rules": "\n".join(s.get("business_rules") or []),
        }
        if args.dry_run:
            print(f"[dry_run] register {sid}:{version}")
        else:
            _http_json("POST", f"{base_url}/v1/schemas", payload, admin_key)
            _http_json("POST", f"{base_url}/v1/schemas/{sid}/{version}/activate", {}, admin_key)

            # Sanity-check that the schema routes to the right D1 binding and executes.
            table = _first_table_from_schema_context(s.get("schema_context") or "")
            _assert(table, f"{sid}: could not parse first table from schema_context")
            sanity = {
                "schema_id": sid,
                "schema_version": version,
                "split": "train",
                "question": f"How many rows are in the {table} table?",
                "gold_sql": f"SELECT COUNT(*) AS row_count FROM {table}",
            }
            out = _http_json("POST", f"{base_url}/v1/validate", sanity, admin_key)
            _assert(bool(out.get("accepted")), f"{sid}: sanity validate rejected: {out}")
            _assert(bool(out.get("execution_ok")), f"{sid}: sanity validate execution failed: {out}")
            _assert(bool(out.get("deterministic_ok")), f"{sid}: sanity validate nondeterministic: {out}")

    # Update registry with db ids
    if args.dry_run:
        print("[dry_run] update registry")
    else:
        _run(
            [
                "python3",
                str(Path("scripts/nl2sql_schema_gen/update_registry.py").resolve()),
                "--registry",
                str(reg_path),
                "--schemas",
                str(schemas_path),
                "--db_map",
                str(db_map_path),
            ],
            cwd=Path.cwd(),
        )

    print(f"done. db_map={db_map_path} registry={reg_path}")


if __name__ == "__main__":
    main()

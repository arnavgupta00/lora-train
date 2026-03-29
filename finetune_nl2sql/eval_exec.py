import argparse
import json
import os
import re
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_URL = "https://nl2sql-dataset-service.adv-dep-test.workers.dev"


def _load_admin_key() -> str:
    # Load from a local file next to this script to avoid Python import-path ambiguity
    # when running as `python finetune_nl2sql/eval_exec.py`.
    key_path = os.path.join(os.path.dirname(__file__), "private_key.py")
    if not os.path.exists(key_path):
        raise SystemExit(
            "Missing finetune_nl2sql/private_key.py. Create it from private_key.py.example "
            "or set NL2SQL_ADMIN_API_KEY and rerun the runner script."
        )
    ns: Dict[str, Any] = {}
    with open(key_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, key_path, "exec"), ns, ns)  # noqa: S102
    key = str(ns.get("ADMIN_API_KEY", "")).strip()
    if not key or "PASTE_KEY_HERE" in key:
        raise SystemExit("finetune_nl2sql/private_key.py has no valid ADMIN_API_KEY")
    return key


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_schema_text(user_content: str) -> str:
    # Expect "Schema:\n... \n\nBusiness rules:\n... \n\nQuestion:\n..."
    m = re.search(r"Schema:\s*(.*?)\n\nBusiness rules:\s*(.*?)\n\nQuestion:\s*(.*)$", user_content, re.S)
    if not m:
        # Fallback: return everything after Schema:
        m2 = re.search(r"Schema:\s*(.*)$", user_content, re.S)
        return (m2.group(1) if m2 else user_content).strip()
    return m.group(1).strip()


def _schema_signature(schema_block: str) -> str:
    lines = [ln.strip() for ln in (schema_block or "").splitlines() if ln.strip()]
    lines.sort()
    return "\n".join(lines)


def _fetch_schema_signature_map() -> Dict[str, str]:
    # /v1/schemas in this service includes schema_context.
    r = requests.get(f"{BASE_URL}/v1/schemas", headers={"User-Agent": "curl/8.0"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    schemas = data.get("schemas", data)
    if not isinstance(schemas, list):
        return {}
    out: Dict[str, str] = {}
    for s in schemas:
        if not isinstance(s, dict):
            continue
        sid = s.get("id") or s.get("schema_id")
        ctx = s.get("schema_context")
        if not isinstance(sid, str) or not isinstance(ctx, str):
            continue
        out.setdefault(_schema_signature(ctx), sid)
    return out


def _parse_table_names(schema_block: str) -> Set[str]:
    tables: Set[str] = set()
    for line in schema_block.splitlines():
        line = line.strip()
        if not line:
            continue
        # table_name(col1, col2)
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", line)
        if m:
            tables.add(m.group(1))
    return tables


def _schema_id_from_tables(tables: Set[str]) -> str:
    # Heuristic routing by distinctive table names.
    t = tables
    if {"organizations", "plans", "subscriptions", "invoices", "usage_events", "support_tickets"}.issubset(t):
        return "b2b_saas"
    if {"customers", "sellers", "products", "orders", "payments", "shipments", "returns"}.issubset(t):
        return "ecommerce_marketplace"
    if {"borrowers", "applications", "loans", "repayments"}.issubset(t):
        return "fintech_lending"
    if {"openings", "candidates", "interviews", "offers", "hires"}.issubset(t) or {
        "job_openings",
        "candidates",
        "interviews",
        "offers",
    }.issubset(t):
        return "hr_ats_enterprise"
    if {"routes", "shipments", "packages", "carriers", "delivery_events"}.issubset(t):
        return "logistics_fleet"
    if {"patients", "providers", "appointments", "claims", "diagnoses"}.issubset(t):
        return "healthcare_clinic_network"
    if {"courses", "enrollments", "lessons", "assessments", "attempts"}.issubset(t) or {
        "courses",
        "enrollments",
        "lessons",
        "assessments",
        "assessment_attempts",
    }.issubset(t):
        return "edtech_platform"
    if {"campaigns", "ad_groups", "spend", "leads", "conversions"}.issubset(t) or {
        "campaigns",
        "ad_groups",
        "ads",
        "spend_daily",
        "leads",
        "conversions",
    }.issubset(t):
        return "martech_campaigns"
    if {"properties", "rooms", "reservations", "folios", "housekeeping"}.issubset(t) or {
        "properties",
        "rooms",
        "reservations",
        "folios",
    }.issubset(t):
        return "hospitality_chain"
    if {"assets", "detections", "alerts", "incidents", "vulnerabilities", "patches"}.issubset(t) or {
        "assets",
        "detections",
        "alerts",
        "incidents",
        "vulnerabilities",
    }.issubset(t):
        return "cybersecurity_siem"
    if {"plants", "work_orders", "boms", "suppliers", "purchase_orders", "inventory"}.issubset(t) or {
        "plants",
        "work_orders",
        "suppliers",
        "purchase_orders",
        "inventory",
        "quality_checks",
    }.issubset(t) or {"bill_of_materials", "bom_items", "plants", "work_orders"}.issubset(t):
        return "manufacturing_supply_chain"
    if {"agents", "clients", "properties", "listings", "showings", "offers", "closings"}.issubset(t):
        return "real_estate_brokerage"
    # If we can't confidently map, default to b2b_saas (most common in many runs)
    return "b2b_saas"


def _normalize_sql(sql: str) -> str:
    s = sql.strip()
    # Remove code fences if present.
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
        if m:
            s = m.group(1).strip()
    # Only keep first statement (validator requires read-only; this avoids trailing chatter).
    s = s.split(";")[0].strip()
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _validator_validate(
    admin_key: str,
    schema_id: str,
    schema_version: str,
    question: str,
    sql: str,
) -> Dict[str, Any]:
    url = f"{BASE_URL}/v1/validate"
    headers = {"Authorization": f"Bearer {admin_key}", "Content-Type": "application/json"}
    payload = {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "split": "test",
        "question": question,
        "gold_sql": sql,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _validator_validate_batch(admin_key: str, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/v1/validate/batch"
    headers = {"Authorization": f"Bearer {admin_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json={"examples": examples}, timeout=120)
    r.raise_for_status()
    data = r.json()
    res = data.get("results", [])
    if not isinstance(res, list):
        raise RuntimeError(f"Unexpected /v1/validate/batch response: {data}")
    return res


def _validate_in_batches(
    admin_key: str,
    examples: List[Dict[str, Any]],
    batch_size: int,
    parallelism: int,
) -> List[Dict[str, Any]]:
    bs = max(1, int(batch_size))
    par = max(1, int(parallelism))
    chunks: List[List[Dict[str, Any]]] = [examples[i : i + bs] for i in range(0, len(examples), bs)]
    if not chunks:
        return []

    out: List[Optional[List[Dict[str, Any]]]] = [None] * len(chunks)

    def worker(ix: int) -> List[Dict[str, Any]]:
        for attempt in range(4):
            try:
                return _validator_validate_batch(admin_key, chunks[ix])
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    with concurrent.futures.ThreadPoolExecutor(max_workers=par) as pool:
        futs = {pool.submit(worker, i): i for i in range(len(chunks))}
        for fut in concurrent.futures.as_completed(futs):
            out[futs[fut]] = fut.result()

    flat: List[Dict[str, Any]] = []
    for part in out:
        if part is None:
            raise RuntimeError("missing validator batch result")
        flat.extend(part)
    return flat


def _result_hash(resp: Dict[str, Any]) -> Optional[str]:
    # Service response shape: { accepted: bool, validation: { result_hash: string|null, ... } }
    v = resp.get("validation")
    if isinstance(v, dict):
        rh = v.get("result_hash")
        return str(rh) if rh is not None else None
    return None


def _build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    # Drop assistant if present and add generation prompt.
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def _generate_sql(
    tokenizer,
    model,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    prompt = _build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _normalize_sql(text)


@torch.inference_mode()
def _generate_sql_batch(
    tokenizer,
    model,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    out_sql: List[str] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(messages_list), bs):
        chunk = messages_list[i : i + bs]
        prompts = [_build_prompt(tokenizer, m) for m in chunk]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        out_sql.extend([_normalize_sql(t) for t in texts])
    return out_sql


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", required=True)
    ap.add_argument("--adapter_dir", default=None, help="Optional PEFT adapter dir. Omit to eval the base model.")
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--validator_batch_size", type=int, default=50)
    ap.add_argument("--validator_parallelism", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows (0 = no limit)")
    ap.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the base model in 8-bit (bitsandbytes) to fit smaller VRAM GPUs. This is for eval only.",
    )
    args = ap.parse_args()

    admin_key = _load_admin_key()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models should left-pad for batched generation to avoid shifting logits.
    tokenizer.padding_side = "left"

    if args.load_in_8bit:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        variant = "lora"
    else:
        model = base
        variant = "base"
    if not args.load_in_8bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    rows = _read_jsonl(args.test_jsonl)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    sig_map = _fetch_schema_signature_map()

    pred_path = os.path.join(args.out_dir, f"predictions.test.{variant}.jsonl")
    report_path = os.path.join(args.out_dir, f"eval_report.{variant}.json")

    total = 0
    exact_match = 0
    exec_match = 0
    validator_fail = 0
    route_fail = 0

    per_row: List[Dict[str, Any]] = []
    messages_list: List[List[Dict[str, str]]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
        if not user_msg or not assistant_msg:
            continue

        user_content = str(user_msg.get("content") or "")
        gold_sql = _normalize_sql(str(assistant_msg.get("content") or ""))

        schema_block = _extract_schema_text(user_content)
        sig = _schema_signature(schema_block)
        schema_id = sig_map.get(sig)
        if not schema_id:
            tables = _parse_table_names(schema_block)
            schema_id = _schema_id_from_tables(tables)
        schema_version = "v1"

        qm = re.search(r"\n\nQuestion:\s*(.*)$", user_content, re.S)
        question = (qm.group(1).strip() if qm else user_content.strip())

        per_row.append(
            {
                "messages": messages,
                "question": question,
                "gold_sql": gold_sql,
                "schema_id": schema_id,
                "schema_version": schema_version,
            }
        )
        messages_list.append(messages)

    pred_sql_list = _generate_sql_batch(
        tokenizer,
        model,
        messages_list,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
    )

    gold_examples: List[Dict[str, Any]] = []
    pred_examples: List[Dict[str, Any]] = []
    for meta, pred_sql in zip(per_row, pred_sql_list):
        gold_examples.append(
            {
                "schema_id": meta["schema_id"],
                "schema_version": meta["schema_version"],
                "split": "test",
                "question": meta["question"],
                "gold_sql": meta["gold_sql"],
            }
        )
        pred_examples.append(
            {
                "schema_id": meta["schema_id"],
                "schema_version": meta["schema_version"],
                "split": "test",
                "question": meta["question"],
                "gold_sql": pred_sql,
            }
        )

    gold_res: List[Dict[str, Any]] = []
    pred_res: List[Dict[str, Any]] = []
    try:
        gold_res = _validate_in_batches(
            admin_key,
            gold_examples,
            batch_size=args.validator_batch_size,
            parallelism=args.validator_parallelism,
        )
        pred_res = _validate_in_batches(
            admin_key,
            pred_examples,
            batch_size=args.validator_batch_size,
            parallelism=args.validator_parallelism,
        )
    except requests.HTTPError:
        validator_fail += 1
    except Exception:
        route_fail += 1

    with open(pred_path, "w", encoding="utf-8") as f_out:
        for i, (meta, pred_sql) in enumerate(zip(per_row, pred_sql_list)):
            total += 1
            if _normalize_sql(pred_sql).lower() == str(meta["gold_sql"]).lower():
                exact_match += 1

            exec_ok = False
            if gold_res and pred_res and i < len(gold_res) and i < len(pred_res):
                gold_hash = _result_hash(gold_res[i])
                pred_hash = _result_hash(pred_res[i])
                if gold_hash and pred_hash and gold_hash == pred_hash:
                    exec_ok = True
            if exec_ok:
                exec_match += 1

            f_out.write(
                json.dumps(
                    {
                        "schema_id": meta["schema_id"],
                        "schema_version": meta["schema_version"],
                        "question": meta["question"],
                        "gold_sql": meta["gold_sql"],
                        "pred_sql": pred_sql,
                        "exact_match": _normalize_sql(pred_sql).lower() == str(meta["gold_sql"]).lower(),
                        "execution_match": exec_ok,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    report = {
        "variant": variant,
        "total": total,
        "exact_match": exact_match,
        "exact_match_accuracy": (exact_match / total) if total else 0.0,
        "execution_match": exec_match,
        "execution_match_accuracy": (exec_match / total) if total else 0.0,
        "validator_http_failures": validator_fail,
        "routing_or_other_failures": route_fail,
        "predictions_path": pred_path,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

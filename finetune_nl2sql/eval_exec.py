import argparse
import json
import os
import re
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    admin_key = _load_admin_key()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    rows = _read_jsonl(args.test_jsonl)

    pred_path = os.path.join(args.out_dir, "predictions.test.jsonl")
    report_path = os.path.join(args.out_dir, "eval_report.json")

    total = 0
    exact_match = 0
    exec_match = 0
    validator_fail = 0
    route_fail = 0

    with open(pred_path, "w", encoding="utf-8") as f_out:
        for row in rows:
            messages = row["messages"]
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if not user_msg or not assistant_msg:
                continue

            user_content = user_msg["content"]
            gold_sql = _normalize_sql(assistant_msg["content"])
            schema_block = _extract_schema_text(user_content)
            tables = _parse_table_names(schema_block)
            schema_id = _schema_id_from_tables(tables)
            schema_version = "v1"

            # Question text: best-effort parse from user content.
            qm = re.search(r"\n\nQuestion:\s*(.*)$", user_content, re.S)
            question = (qm.group(1).strip() if qm else user_content.strip())

            try:
                pred_sql = _generate_sql(tokenizer, model, messages, max_new_tokens=args.max_new_tokens)
            except Exception:
                pred_sql = ""

            total += 1
            if _normalize_sql(pred_sql).lower() == gold_sql.lower():
                exact_match += 1

            exec_ok = False
            try:
                gold_v = _validator_validate(admin_key, schema_id, schema_version, question, gold_sql)
                pred_v = _validator_validate(admin_key, schema_id, schema_version, question, pred_sql)

                gold_hash = _result_hash(gold_v)
                pred_hash = _result_hash(pred_v)
                if gold_hash and pred_hash and gold_hash == pred_hash:
                    exec_ok = True
            except requests.HTTPError:
                validator_fail += 1
            except Exception:
                route_fail += 1

            if exec_ok:
                exec_match += 1

            f_out.write(
                json.dumps(
                    {
                        "schema_id": schema_id,
                        "schema_version": schema_version,
                        "question": question,
                        "gold_sql": gold_sql,
                        "pred_sql": pred_sql,
                        "exact_match": _normalize_sql(pred_sql).lower() == gold_sql.lower(),
                        "execution_match": exec_ok,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    report = {
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

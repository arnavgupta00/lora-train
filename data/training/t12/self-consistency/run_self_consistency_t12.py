#!/usr/bin/env python3
"""
T12 Self-Consistency Runner (n=7, sampling-only)

This script generates 7 sampled SQL candidates per prompt, applies execution-aware
voting, and evaluates the voted SQL on BIRD dev databases.

Design goals:
- No greedy mode (sampling is enforced).
- Parallelization through batched generation and threaded evaluation.
- Execution-aware voting for text-to-SQL robustness.

References that informed the strategy:
- Self-Consistency (Wang et al., ICLR 2023): arXiv:2203.11171
- Execution-Guided Decoding for text-to-SQL: arXiv:1807.03100
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import T12 helpers from parent directory.
SCRIPT_DIR = Path(__file__).resolve().parent
T12_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(T12_DIR))

from t12_utils import find_database, get_t12_system_prompt_hash, normalize_sql


def load_prompts(prompts_file: str, limit: int) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
            if limit > 0 and len(prompts) >= limit:
                break
    return prompts


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Optional[Dict[str, Any]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_progress_snapshot(
    output_dir: Path,
    prediction_rows: List[Optional[Dict[str, Any]]],
    candidate_rows: List[Optional[Dict[str, Any]]],
    state: Dict[str, Any],
) -> None:
    write_jsonl(output_dir / "predictions_sc_t12.jsonl", prediction_rows)
    write_jsonl(output_dir / "candidates_sc_t12.jsonl", candidate_rows)
    with open(output_dir / "progress_state_sc_t12.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[bool, Any, float]:
    start = time.time()
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        rows = conn.execute(sql).fetchall()
        conn.close()
        return True, rows, time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def result_signature(rows: Any) -> str:
    """Order-independent signature for SQL result voting."""
    try:
        normalized = sorted(tuple(r) for r in rows)
        payload = json.dumps(normalized, ensure_ascii=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        payload = json.dumps(rows, ensure_ascii=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def results_match(gold_rows: Any, pred_rows: Any) -> bool:
    try:
        return set(tuple(r) for r in gold_rows) == set(tuple(r) for r in pred_rows)
    except Exception:
        return False


def build_vote(
    db_path: Optional[str],
    candidates: List[str],
    timeout: int,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Voting strategy:
    1) Execute all candidates.
    2) Group successful candidates by result signature.
    3) Pick the largest executable-result group.
    4) Tie-break by SQL text frequency, then shortest SQL.
    5) If all fail, fallback to most frequent SQL text.
    """
    attempt_records: List[Dict[str, Any]] = []

    sql_freq = Counter(candidates)

    if not db_path:
        chosen = sql_freq.most_common(1)[0][0]
        meta = {
            "method": "fallback",
            "reason": "db_not_found",
            "num_candidates": len(candidates),
            "num_executable": 0,
            "confidence": 0.0,
        }
        for sql in candidates:
            attempt_records.append(
                {
                    "sql": sql,
                    "ok": False,
                    "error": "db_not_found",
                    "latency_sec": 0.0,
                    "result_sig": None,
                }
            )
        return chosen, meta, attempt_records

    groups: Dict[str, List[int]] = defaultdict(list)
    exec_ok = 0

    for i, sql in enumerate(candidates):
        ok, payload, latency = execute_sql(db_path, sql, timeout=timeout)
        if ok:
            sig = result_signature(payload)
            groups[sig].append(i)
            exec_ok += 1
            attempt_records.append(
                {
                    "sql": sql,
                    "ok": True,
                    "error": None,
                    "latency_sec": round(latency, 6),
                    "result_sig": sig,
                }
            )
        else:
            attempt_records.append(
                {
                    "sql": sql,
                    "ok": False,
                    "error": str(payload)[:240],
                    "latency_sec": round(latency, 6),
                    "result_sig": None,
                }
            )

    if exec_ok == 0:
        chosen = sql_freq.most_common(1)[0][0]
        meta = {
            "method": "fallback",
            "reason": "all_failed",
            "num_candidates": len(candidates),
            "num_executable": 0,
            "confidence": 0.0,
        }
        return chosen, meta, attempt_records

    # Rank groups by size.
    ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    best_sig, best_idxs = ranked[0]

    # Candidate tie-break in winning group.
    winning_sqls = [candidates[i] for i in best_idxs]
    winning_freq = Counter(winning_sqls)
    best_sql = sorted(
        winning_freq.keys(),
        key=lambda s: (-winning_freq[s], len(s), s),
    )[0]

    meta = {
        "method": "execution_result_vote",
        "num_candidates": len(candidates),
        "num_executable": exec_ok,
        "num_result_groups": len(groups),
        "winning_group_size": len(best_idxs),
        "winning_result_sig": best_sig,
        "confidence": round(len(best_idxs) / float(len(candidates)), 4),
        "group_sizes": sorted([len(v) for v in groups.values()], reverse=True),
    }
    return best_sql, meta, attempt_records


def generate_candidates_batch(
    model,
    tokenizer,
    prompt_batch: List[str],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[List[str]]:
    """
    Batched generation with num_return_sequences=n_samples.

    Returns: list sized [batch_size], each element is list of n_samples SQL strings.
    """
    import torch

    inputs = tokenizer(
        prompt_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_beams=1,
            num_return_sequences=n_samples,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Map flat outputs back to per-example candidates.
    # For batch B and n_samples N, output order is:
    # ex0_s0..ex0_sN-1, ex1_s0..ex1_sN-1, ...
    bsz = len(prompt_batch)
    seqs: List[List[str]] = [[] for _ in range(bsz)]

    input_lens = inputs["attention_mask"].sum(dim=1).tolist()

    out_idx = 0
    for i in range(bsz):
        in_len = int(input_lens[i])
        for _ in range(n_samples):
            out_ids = outputs[out_idx][in_len:]
            raw = tokenizer.decode(out_ids, skip_special_tokens=True)
            seqs[i].append(normalize_sql(raw))
            out_idx += 1

    return seqs


def evaluate_selected_predictions(
    prompts: List[Dict[str, Any]],
    selected_sql: List[str],
    db_dir: str,
    eval_workers: int,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    def one(i: int) -> Dict[str, Any]:
        row = prompts[i]
        db_id = row["db_id"]
        gold_sql = normalize_sql(row.get("gold_sql", row.get("SQL", "")))
        pred_sql = normalize_sql(selected_sql[i])
        db_path = find_database(db_dir, db_id)

        out = {
            "question_id": row.get("question_id", i),
            "db_id": db_id,
            "difficulty": row.get("difficulty", "unknown"),
            "gold_sql": gold_sql,
            "predicted_sql": pred_sql,
            "correct": False,
            "exact_match": False,
            "exec_failed": False,
            "error": None,
        }

        if pred_sql.lower().strip() == gold_sql.lower().strip():
            out["exact_match"] = True

        if not db_path:
            out["exec_failed"] = True
            out["error"] = "db_not_found"
            return out

        gok, grows, _ = execute_sql(db_path, gold_sql, timeout=timeout)
        if not gok:
            out["error"] = f"gold_error: {str(grows)[:180]}"
            return out

        pok, prows, _ = execute_sql(db_path, pred_sql, timeout=timeout)
        if not pok:
            out["exec_failed"] = True
            out["error"] = str(prows)[:180]
            return out

        if results_match(grows, prows):
            out["correct"] = True

        return out

    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as pool:
        futs = {pool.submit(one, i): i for i in range(len(prompts))}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(prompts):
                print(f"  [eval {done}/{len(prompts)}]")

    final_results = [r for r in results if r is not None]

    total = len(final_results)
    correct = sum(1 for r in final_results if r["correct"])
    exact = sum(1 for r in final_results if r["exact_match"])
    exec_failed = sum(1 for r in final_results if r["exec_failed"])

    by_db: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_diff: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in final_results:
        db_id = r["db_id"]
        diff = r["difficulty"]
        by_db[db_id]["total"] += 1
        by_db[db_id]["correct"] += int(r["correct"])
        by_diff[diff]["total"] += 1
        by_diff[diff]["correct"] += int(r["correct"])

    summary = {
        "total_examples": total,
        "execution_match": correct,
        "execution_accuracy": round(100.0 * correct / total, 2) if total else 0.0,
        "exact_match": exact,
        "exact_match_pct": round(100.0 * exact / total, 2) if total else 0.0,
        "execution_errors": exec_failed,
        "execution_error_pct": round(100.0 * exec_failed / total, 2) if total else 0.0,
        "by_database": [
            {
                "db_id": k,
                "total": v["total"],
                "exec_match": v["correct"],
                "accuracy": round(100.0 * v["correct"] / v["total"], 2) if v["total"] else 0.0,
            }
            for k, v in sorted(by_db.items())
        ],
        "by_difficulty": [
            {
                "difficulty": k,
                "total": v["total"],
                "exec_match": v["correct"],
                "accuracy": round(100.0 * v["correct"] / v["total"], 2) if v["total"] else 0.0,
            }
            for k, v in sorted(by_diff.items())
        ],
    }

    return final_results, summary


def quick_match_selected(
    example: Dict[str, Any],
    pred_sql: str,
    db_dir: str,
    timeout: int,
) -> bool:
    db_id = example["db_id"]
    db_path = find_database(db_dir, db_id)
    if not db_path:
        return False
    gold_sql = normalize_sql(example.get("gold_sql", example.get("SQL", "")))
    pred_sql = normalize_sql(pred_sql)

    gok, grows, _ = execute_sql(db_path, gold_sql, timeout=timeout)
    if not gok:
        return False
    pok, prows, _ = execute_sql(db_path, pred_sql, timeout=timeout)
    if not pok:
        return False
    return results_match(grows, prows)


def main() -> None:
    parser = argparse.ArgumentParser(description="T12 self-consistency runner (sampling-only, n=7)")

    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--adapter_repo", type=str, default="Arnav3035/garuda-sql-2b")
    parser.add_argument("--prompts_file", type=str, default="data/training/t12/bird_dev_t12.jsonl")
    parser.add_argument("--db_dir", type=str, default="data/bird_eval_datasets/dev_databases")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--n_samples", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vote_workers", type=int, default=8)
    parser.add_argument("--eval_workers", type=int, default=8)
    parser.add_argument("--sql_timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from predictions_sc_t12.jsonl in output_dir")
    parser.add_argument("--min_batch_size", type=int, default=1, help="Minimum batch size when reducing after OOM")

    args = parser.parse_args()

    # Enforce sampling-only mode.
    if args.n_samples != 7:
        raise ValueError("This runner is fixed to n=7 by design. Pass --n_samples 7.")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0.0 for self-consistency sampling.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    # Reduce fragmentation risk for long runs on consumer GPUs.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("=" * 72)
    print("T12 SELF-CONSISTENCY (N=7, SAMPLING-ONLY)")
    print("=" * 72)
    print(f"Base model: {args.base_model_id}")
    print(f"Adapter: {args.adapter_repo}")
    print(f"Prompts: {args.prompts_file}")
    print(f"DB dir: {args.db_dir}")
    print(f"Output: {args.output_dir}")
    print(
        f"Sampling: n={args.n_samples}, temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, do_sample=True"
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompts = load_prompts(args.prompts_file, args.limit)
    print(f"Loaded {len(prompts)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base then apply HF adapter repo.
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter_repo)
    model = model.to(args.device)
    model.eval()

    cfg = {
        "base_model_id": args.base_model_id,
        "adapter_repo": args.adapter_repo,
        "prompts_file": args.prompts_file,
        "db_dir": args.db_dir,
        "t12_system_prompt_hash": get_t12_system_prompt_hash(),
        "generation": {
            "n_samples": args.n_samples,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "num_beams": 1,
        },
        "runtime": {
            "batch_size": args.batch_size,
            "vote_workers": args.vote_workers,
            "eval_workers": args.eval_workers,
            "sql_timeout": args.sql_timeout,
            "seed": args.seed,
            "device": args.device,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(Path(args.output_dir) / "generation_config_sc_t12.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Build rendered prompts once.
    rendered_prompts: List[str] = []
    for p in prompts:
        t12p = p["t12_prompt"]
        msgs = [
            {"role": "system", "content": t12p["system"]},
            {"role": "user", "content": t12p["user"]},
        ]
        rendered_prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    selected_sql: List[str] = [""] * len(prompts)
    prediction_rows: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    candidate_rows: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

    qid_to_idx = {p.get("question_id", i): i for i, p in enumerate(prompts)}
    running_processed = 0
    running_correct = 0
    filled_count = 0

    pred_path = output_dir / "predictions_sc_t12.jsonl"
    cand_path = output_dir / "candidates_sc_t12.jsonl"
    state_path = output_dir / "progress_state_sc_t12.json"

    if args.resume and pred_path.exists():
        existing_preds = read_jsonl(pred_path)
        existing_cands = read_jsonl(cand_path)
        cand_by_qid = {r.get("question_id"): r for r in existing_cands}

        for r in existing_preds:
            qid = r.get("question_id")
            if qid not in qid_to_idx:
                continue
            idx = qid_to_idx[qid]
            prediction_rows[idx] = r
            selected_sql[idx] = normalize_sql(r.get("predicted_sql", ""))
            if qid in cand_by_qid:
                candidate_rows[idx] = cand_by_qid[qid]

        filled_count = sum(1 for r in prediction_rows if r is not None)
        print(f"Resumed {filled_count}/{len(prompts)} rows from existing output files")

        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            running_processed = int(st.get("running_processed", 0))
            running_correct = int(st.get("running_correct", 0))
            print(
                f"Resume state: running_EX={((100.0 * running_correct / running_processed) if running_processed else 0.0):.2f}% "
                f"({running_correct}/{running_processed})"
            )
        else:
            # Backfill running metrics if old run predates state file.
            running_processed = 0
            running_correct = 0
            for i, row in enumerate(prediction_rows):
                if row is None:
                    continue
                running_processed += 1
                if quick_match_selected(prompts[i], row.get("predicted_sql", ""), args.db_dir, args.sql_timeout):
                    running_correct += 1
            print(
                f"Backfilled resume metrics: running_EX={((100.0 * running_correct / running_processed) if running_processed else 0.0):.2f}% "
                f"({running_correct}/{running_processed})"
            )

    gen_start = time.time()

    batch_start = 0
    current_batch_size = args.batch_size

    while batch_start < len(prompts):
        batch_end = min(batch_start + current_batch_size, len(prompts))
        unresolved = [i for i in range(batch_start, batch_end) if prediction_rows[i] is None]

        if not unresolved:
            batch_start = batch_end
            continue

        p_batch = [prompts[i] for i in unresolved]
        r_batch = [rendered_prompts[i] for i in unresolved]
        local_idx_by_global = {g: l for l, g in enumerate(unresolved)}

        try:
            batch_candidates = generate_candidates_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_batch=r_batch,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        except RuntimeError as e:
            emsg = str(e).lower()
            if "outofmemory" in emsg or "cuda out of memory" in emsg:
                import torch

                torch.cuda.empty_cache()
                if current_batch_size <= args.min_batch_size:
                    raise RuntimeError(
                        "OOM even at minimum batch size. Try lower max_new_tokens, lower n_samples, or use resume after restarting with more free VRAM."
                    ) from e
                new_bs = max(args.min_batch_size, current_batch_size // 2)
                print(
                    f"  [OOM] Reducing batch_size {current_batch_size} -> {new_bs} and retrying from index {batch_start}"
                )
                current_batch_size = new_bs
                continue
            raise

        def vote_one(local_i: int) -> Tuple[int, str, Dict[str, Any], List[Dict[str, Any]], bool]:
            global_i = unresolved[local_i]
            db_id = prompts[global_i]["db_id"]
            db_path = find_database(args.db_dir, db_id)
            sql, meta, attempts = build_vote(db_path, batch_candidates[local_i], args.sql_timeout)
            partial_correct = quick_match_selected(prompts[global_i], sql, args.db_dir, args.sql_timeout)
            return global_i, sql, meta, attempts, partial_correct

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.vote_workers) as pool:
            futs = [pool.submit(vote_one, i) for i in range(len(p_batch))]
            for fut in concurrent.futures.as_completed(futs):
                i, chosen, vmeta, attempts, partial_correct = fut.result()
                selected_sql[i] = chosen
                row = prompts[i]

                prediction_rows[i] = {
                    "question_id": row.get("question_id", i),
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "predicted_sql": chosen,
                    "gold_sql": row.get("gold_sql", row.get("SQL", "")),
                    "difficulty": row.get("difficulty", "unknown"),
                    "vote_metadata": vmeta,
                    "partial_correct": partial_correct,
                }
                candidate_rows[i] = {
                    "question_id": row.get("question_id", i),
                    "db_id": row["db_id"],
                    "candidates": batch_candidates[local_idx_by_global[i]],
                    "vote_metadata": vmeta,
                    "attempts": attempts,
                    "selected_sql": chosen,
                }
                filled_count += 1
                running_processed += 1
                running_correct += int(partial_correct)

        running_ex = (100.0 * running_correct / running_processed) if running_processed else 0.0
        print(
            f"  [gen+vote {filled_count}/{len(prompts)}] running_EX={running_ex:.2f}% ({running_correct}/{running_processed})"
        )

        save_progress_snapshot(
            output_dir=output_dir,
            prediction_rows=prediction_rows,
            candidate_rows=candidate_rows,
            state={
                "running_processed": running_processed,
                "running_correct": running_correct,
                "filled_count": filled_count,
                "next_batch_start": batch_end,
                "current_batch_size": current_batch_size,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        batch_start = batch_end

    gen_elapsed = time.time() - gen_start

    # Save prediction outputs one final time.
    save_progress_snapshot(
        output_dir=output_dir,
        prediction_rows=prediction_rows,
        candidate_rows=candidate_rows,
        state={
            "running_processed": running_processed,
            "running_correct": running_correct,
            "filled_count": filled_count,
            "next_batch_start": len(prompts),
            "current_batch_size": current_batch_size,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "done_generation": True,
        },
    )

    # Ensure no missing rows before final full evaluation.
    missing = [i for i, r in enumerate(prediction_rows) if r is None]
    if missing:
        raise RuntimeError(
            f"Generation incomplete ({len(missing)} missing). Re-run with --resume to continue."
        )

    # Evaluate selected SQL.
    print("Evaluating voted predictions...")
    per_example, summary = evaluate_selected_predictions(
        prompts=prompts,
        selected_sql=selected_sql,
        db_dir=args.db_dir,
        eval_workers=args.eval_workers,
        timeout=args.sql_timeout,
    )

    # Aggregate voting stats.
    voting_conf = []
    voting_used = 0
    fallback_used = 0
    for row in prediction_rows:
        if row is None:
            continue
        vm = row["vote_metadata"]
        if vm["method"] == "execution_result_vote":
            voting_used += 1
            voting_conf.append(vm.get("confidence", 0.0))
        else:
            fallback_used += 1

    summary["n_samples"] = args.n_samples
    summary["temperature"] = args.temperature
    summary["top_p"] = args.top_p
    summary["generation_time_min"] = round(gen_elapsed / 60.0, 2)
    summary["voting_stats"] = {
        "voting_used": voting_used,
        "fallback_used": fallback_used,
        "avg_confidence": round(sum(voting_conf) / len(voting_conf), 4) if voting_conf else 0.0,
    }

    with open(Path(args.output_dir) / "evaluation_report_sc_t12.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(Path(args.output_dir) / "per_example_results_sc_t12.jsonl", "w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Write a compact markdown summary.
    md_path = Path(args.output_dir) / "eval_summary_sc_t12.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# T12 Self-Consistency Evaluation Summary\n\n")
        f.write(f"- Total examples: {summary['total_examples']}\n")
        f.write(f"- Execution Accuracy: {summary['execution_match']}/{summary['total_examples']} ({summary['execution_accuracy']}%)\n")
        f.write(f"- Exact Match: {summary['exact_match']}/{summary['total_examples']} ({summary['exact_match_pct']}%)\n")
        f.write(f"- Execution Errors: {summary['execution_errors']} ({summary['execution_error_pct']}%)\n")
        f.write(f"- n_samples: {summary['n_samples']}\n")
        f.write(f"- temperature: {summary['temperature']}\n")
        f.write(f"- top_p: {summary['top_p']}\n")
        f.write(f"- generation_time_min: {summary['generation_time_min']}\n")
        f.write("\n## Voting Stats\n\n")
        f.write(f"- voting_used: {summary['voting_stats']['voting_used']}\n")
        f.write(f"- fallback_used: {summary['voting_stats']['fallback_used']}\n")
        f.write(f"- avg_confidence: {summary['voting_stats']['avg_confidence']}\n")

    print("\n" + "=" * 72)
    print("SELF-CONSISTENCY COMPLETE")
    print("=" * 72)
    print(f"Execution Accuracy: {summary['execution_accuracy']}%")
    print(f"Exact Match: {summary['exact_match_pct']}%")
    print(f"Generation time: {summary['generation_time_min']} min")
    print(f"Output dir: {args.output_dir}")
    print("Files:")
    print(f"  - {pred_path}")
    print(f"  - {cand_path}")
    print(f"  - {Path(args.output_dir) / 'evaluation_report_sc_t12.json'}")
    print(f"  - {Path(args.output_dir) / 'per_example_results_sc_t12.jsonl'}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()

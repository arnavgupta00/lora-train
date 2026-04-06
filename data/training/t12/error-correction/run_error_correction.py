#!/usr/bin/env python3
"""
T12 Error-Correction Pipeline

Main script to run the error-correction pipeline on T12 predictions.

Features:
- Classifies failures by type
- Extracts relevant schema blocks
- Generates repair prompts
- Runs repair model inference with thinking enabled
- Validates and accepts/rejects repairs
- Produces repaired predictions and detailed logs

Usage:
    python run_error_correction.py \
        --predictions runs/t12_baseline_3090/qwen3.5-2b/without-sampling/predictions/predictions_t12.jsonl \
        --eval_results runs/t12_baseline_3090/qwen3.5-2b/without-sampling/eval/per_example_results.jsonl \
        --prompts data/training/t12/bird_dev_t12.jsonl \
        --db_dir data/bird_eval_datasets/dev_databases \
        --output_dir data/training/t12/error-correction \
        --model_id Qwen/Qwen3.5-2B \
        --enable_thinking \
        --max_repair_attempts 2
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

# Auto-detect HuggingFace cache location
if "HF_HOME" not in os.environ:
    for cache_path in ["/workspace/hf", "/runpod-volume/hf", os.path.expanduser("~/.cache/huggingface")]:
        if os.path.isdir(cache_path):
            os.environ["HF_HOME"] = cache_path
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_path, "datasets")
            break

# Local imports
from repair_utils import SchemaCache, normalize_sql
from classify_failures import classify_failure, FailureClassification
from extract_relevant_schema import extract_relevant_schema, RelevantSchemaBlock
from repair_prompts import RepairPromptBuilder, EscalationPromptBuilder
from validate_repairs import validate_repair


# =============================================================================
# Data Loading
# =============================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_prompts_by_id(path: str) -> Dict[int, Dict[str, Any]]:
    """Load prompts and index by question_id."""
    prompts = load_jsonl(path)
    return {p["question_id"]: p for p in prompts}


def merge_data(
    predictions: List[Dict],
    eval_results: List[Dict],
    prompts_by_id: Dict[int, Dict],
) -> List[Dict[str, Any]]:
    """Merge predictions, eval results, and prompts into unified records."""
    # Index eval results by question_id
    eval_by_id = {e["question_id"]: e for e in eval_results}
    
    merged = []
    for pred in predictions:
        qid = pred["question_id"]
        record = {
            "question_id": qid,
            "db_id": pred["db_id"],
            "question": pred["question"],
            "predicted_sql": pred.get("predicted_sql", ""),
            "gold_sql": pred.get("gold_sql", ""),
            "difficulty": pred.get("difficulty", "unknown"),
        }
        
        # Add eval info
        if qid in eval_by_id:
            eval_rec = eval_by_id[qid]
            record["correct"] = eval_rec.get("correct", False)
            record["exec_failed"] = eval_rec.get("exec_failed", False)
            record["wrong_result"] = eval_rec.get("wrong_result", False)
            record["pred_error"] = eval_rec.get("pred_error", "")
        else:
            record["correct"] = False
            record["exec_failed"] = True
            record["wrong_result"] = False
            record["pred_error"] = "unknown"
        
        # Add prompt info (hints)
        if qid in prompts_by_id:
            prompt_rec = prompts_by_id[qid]
            record["evidence"] = prompt_rec.get("evidence", "")
        else:
            record["evidence"] = ""
        
        merged.append(record)
    
    return merged


# =============================================================================
# Model Loading
# =============================================================================

def resolve_adapter_path(adapter_path: str) -> str:
    """Resolve adapter path for local directories before falling back to Hub IDs.

    Supports:
    - Absolute local paths
    - Relative local paths
    - Common mistaken '/runs/...' style paths by trying './runs/...'
    """
    raw_path = adapter_path.strip()
    path_obj = Path(raw_path)

    # 1) As-provided path (absolute or relative)
    if path_obj.exists():
        return str(path_obj.resolve())

    # 2) Common mistake: /runs/... while repo root contains ./runs/...
    if raw_path.startswith("/runs/"):
        alt = Path.cwd() / raw_path.lstrip("/")
        if alt.exists():
            return str(alt.resolve())

    # 3) Keep original value for HF Hub resolution, if intended
    return raw_path

def load_model(model_id: str, device: str = "cuda", adapter_path: Optional[str] = None):
    """Load model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "--adapter_path was provided but peft is not installed. Install with `pip install peft`."
            ) from exc
        resolved_adapter_path = resolve_adapter_path(adapter_path)
        print(f"Loading LoRA adapter: {resolved_adapter_path}")

        # If path looks local, validate adapter_config.json before invoking PEFT.
        local_candidate = Path(resolved_adapter_path)
        if local_candidate.exists() and local_candidate.is_dir():
            config_path = local_candidate / "adapter_config.json"
            if not config_path.exists():
                raise ValueError(
                    f"Invalid adapter directory: '{resolved_adapter_path}' (missing adapter_config.json)"
                )

        model = PeftModel.from_pretrained(model, resolved_adapter_path)
    
    model = model.to(device)
    model.eval()
    if hasattr(model, "generation_config"):
        # Force clean greedy decoding and suppress warnings from model defaults.
        model.generation_config.do_sample = False
        model.generation_config.num_beams = 1
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    
    print(f"Model loaded on {device}")
    return model, tokenizer


# =============================================================================
# Repair Generation
# =============================================================================

def generate_repairs_batch(
    model,
    tokenizer,
    messages_batch: List[List[Dict[str, str]]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    device: str = "cuda",
) -> List[str]:
    """Generate repair SQL for a batch of prompts."""
    import torch

    if not messages_batch:
        return []

    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature if temperature > 0 else None,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw_outputs = []
    for row_idx in range(len(outputs)):
        input_len = inputs["input_ids"][row_idx].shape[0]
        gen_ids = outputs[row_idx][input_len:]
        raw_outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return raw_outputs


def generate_repair(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Generate repair SQL from a single prompt."""
    return generate_repairs_batch(
        model=model,
        tokenizer=tokenizer,
        messages_batch=[messages],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )[0]


# =============================================================================
# Main Pipeline
# =============================================================================

def run_error_correction(
    records: List[Dict],
    schema_cache: SchemaCache,
    model,
    tokenizer,
    enable_thinking: bool = True,
    max_repair_attempts: int = 2,
    min_repairability_score: float = 0.5,
    max_new_tokens: int = 256,
    generation_batch_size: int = 8,
    device: str = "cuda",
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, Any]]:
    """
    Run error correction on all records.
    
    Returns:
        (repaired_predictions, repair_log, quarantined_repairs, summary_stats)
    """
    # Initialize builders
    prompt_builder = RepairPromptBuilder(enable_thinking=enable_thinking)
    escalation_builder = EscalationPromptBuilder(enable_thinking=enable_thinking)
    
    repaired_predictions: List[Optional[Dict[str, Any]]] = [None] * len(records)
    repair_log = []
    quarantined_repairs = []
    
    stats = {
        "total": len(records),
        "correct": 0,
        "exec_failed": 0,
        "wrong_result": 0,
        "repair_attempted": 0,
        "repair_accepted": 0,
        "repair_rejected": 0,
        "repair_skipped": 0,
        "quarantined_repairs": 0,
        "repair_scope": "V2 repairs all non-correct SQL (including wrong-result) and accepts only when repaired execution results match gold.",
        "by_failure_type": defaultdict(lambda: {"attempted": 0, "accepted": 0}),
    }

    def finalize_prediction(idx: int, qid: int, db_id: str, predicted_sql: str, repaired: bool, original_sql: Optional[str] = None):
        repaired_predictions[idx] = {
            "question_id": qid,
            "db_id": db_id,
            "predicted_sql": predicted_sql,
            "repaired": repaired,
            "original_sql": original_sql,
        }

    pending: List[Dict[str, Any]] = []

    progress = tqdm(
        total=len(records),
        desc="T10 repair",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    for i, record in enumerate(records):
        qid = record["question_id"]
        db_id = record["db_id"]

        # Skip correct predictions
        if record.get("correct"):
            stats["correct"] += 1
            finalize_prediction(i, qid, db_id, record["predicted_sql"], repaired=False)
            progress.update(1)
            continue
        
        # Count failure types
        if record.get("exec_failed"):
            stats["exec_failed"] += 1
        if record.get("wrong_result"):
            stats["wrong_result"] += 1
        
        # Load schema
        schema = schema_cache.get_schema(db_id)
        if not schema:
            progress.write(f"Warning: Could not load schema for {db_id}")
            finalize_prediction(i, qid, db_id, record["predicted_sql"], repaired=False)
            progress.update(1)
            continue
        
        # Classify failure
        classification = classify_failure(record, schema)
        
        if not classification:
            classification = FailureClassification(
                question_id=qid,
                db_id=db_id,
                failure_type="generic_exec_error",
                confidence=0.3,
                reason="Classification missing; fallback to generic repair",
                repairability_score=0.0,
            )
        
        # Extract relevant schema
        schema_block = extract_relevant_schema(
            schema=schema,
            question=record["question"],
            hints=record.get("evidence", ""),
            predicted_sql=record["predicted_sql"],
            error=record.get("pred_error", ""),
        )
        
        # Get database path
        db_path = schema_cache.get_db_path(db_id)
        
        # Build repair log entry
        log_entry = {
            "question_id": qid,
            "db_id": db_id,
            "original_predicted_sql": record["predicted_sql"],
            "failure_type": classification.failure_type,
            "repairability_score": classification.repairability_score,
            "failed_identifier": classification.failed_identifier,
            "identifier_candidates": classification.identifier_candidates or [],
            "candidate_scores": [
                {
                    "name": c.get("name"),
                    "table": c.get("table"),
                    "score": round(c.get("score", 0.0), 4),
                    "scope": c.get("scope"),
                }
                for c in (classification.identifier_candidates or [])
            ],
            "chosen_suggestion": classification.suggested_fix,
            "candidate_table_name": classification.correct_table,
            "repair_attempted": True,
            "attempts": [],
            "final_accepted": False,
            "final_sql": record["predicted_sql"],
            "final_reason": "",
            "rejection_category": None,
            "extracted_tables": list(schema_block.tables),
            "extracted_columns": {t: list(cols) for t, cols in schema_block.columns_by_table.items()},
            "extracted_relations": list(schema_block.relations),
        }
        
        stats["repair_attempted"] += 1
        stats["by_failure_type"][classification.failure_type]["attempted"] += 1
        pending.append({
            "index": i,
            "record": record,
            "schema": schema,
            "db_path": db_path,
            "classification": classification,
            "schema_block": schema_block,
            "log_entry": log_entry,
            "original_exec_result": None,
            "gold_exec_result": None,
        })

    for attempt in range(max_repair_attempts):
        if not pending:
            break

        progress.write(
            f"Attempt {attempt + 1}/{max_repair_attempts}: processing {len(pending)} repair candidates "
            f"in batches of {generation_batch_size}"
        )
        next_pending: List[Dict[str, Any]] = []

        for batch_start in range(0, len(pending), generation_batch_size):
            batch = pending[batch_start:batch_start + generation_batch_size]
            messages_batch = []

            for item in batch:
                record = item["record"]
                classification = item["classification"]
                schema = item["schema"]
                schema_block = item["schema_block"]
                log_entry = item["log_entry"]

                if attempt == 0:
                    prompt_error_message = record.get("pred_error", "") or classification.reason
                    messages = prompt_builder.build_messages(
                        failure_type=classification.failure_type,
                        schema_block=schema_block,
                        question=record["question"],
                        hints=record.get("evidence", ""),
                        predicted_sql=record["predicted_sql"],
                        error_message=prompt_error_message,
                        failed_identifier=classification.failed_identifier,
                        suggested_fix=classification.suggested_fix,
                        wrong_alias=classification.wrong_alias,
                        correct_table=classification.correct_table,
                    )
                else:
                    schema_block = extract_relevant_schema(
                        schema=schema,
                        question=record["question"],
                        hints=record.get("evidence", ""),
                        predicted_sql=record["predicted_sql"],
                        error=record.get("pred_error", ""),
                        expanded=True,
                    )
                    item["schema_block"] = schema_block
                    prev_attempt = log_entry["attempts"][-1]
                    messages = escalation_builder.build_messages(
                        schema_block=schema_block,
                        question=record["question"],
                        hints=record.get("evidence", ""),
                        predicted_sql=record["predicted_sql"],
                        original_error=record.get("pred_error", "") or classification.reason,
                        first_repair_sql=prev_attempt["repaired_sql"],
                        first_repair_error=prev_attempt.get("reason") or prev_attempt.get("exec_error") or "unknown feedback",
                    )
                messages_batch.append(messages)

            raw_outputs = generate_repairs_batch(
                model=model,
                tokenizer=tokenizer,
                messages_batch=messages_batch,
                max_new_tokens=max_new_tokens,
                device=device,
            )

            for item, raw_output in zip(batch, raw_outputs):
                idx = item["index"]
                record = item["record"]
                classification = item["classification"]
                schema = item["schema"]
                db_path = item["db_path"]
                log_entry = item["log_entry"]

                validation = validate_repair(
                    original_sql=record["predicted_sql"],
                    gold_sql=record.get("gold_sql", ""),
                    raw_repair_output=raw_output,
                    db_path=db_path,
                    schema=schema,
                    failure_type=classification.failure_type,
                    identifier_confidence=(
                        classification.identifier_candidates[0]["score"]
                        if classification.identifier_candidates else None
                    ),
                    original_exec_result=item.get("original_exec_result"),
                    gold_exec_result=item.get("gold_exec_result"),
                )
                if item.get("original_exec_result") is None:
                    item["original_exec_result"] = validation.get("original_exec_result")
                if item.get("gold_exec_result") is None:
                    item["gold_exec_result"] = validation.get("gold_exec_result")

                attempt_log = {
                    "attempt_index": attempt,
                    "context_variant": "expanded" if attempt > 0 else "standard",
                    "raw_output": raw_output[:500],
                    "repaired_sql": validation["cleaned_sql"],
                    "accepted": validation["accepted"],
                    "reason": validation["reason"],
                    "diff_ratio": validation["diff_ratio"],
                    "hygiene_issues": validation["hygiene_issues"],
                    "schema_errors": validation["schema_errors"],
                    "exec_error": validation["repair_exec_result"][1] if not validation["repair_exec_result"][0] else None,
                    "structure_issues": validation["structure_issues"],
                    "structure_metrics": validation["structure_metrics"],
                    "quarantine": validation["quarantine"],
                    "quarantine_reasons": validation["quarantine_reasons"],
                    "matches_gold": validation.get("matches_gold", False),
                }
                log_entry["attempts"].append(attempt_log)

                low_confidence_identifier = bool(
                    classification.identifier_candidates
                    and classification.identifier_candidates[0]["score"] < 0.8
                )
                if validation["quarantine"] or low_confidence_identifier:
                    quarantined_repairs.append({
                        "question_id": record["question_id"],
                        "db_id": record["db_id"],
                        "failure_type": classification.failure_type,
                        "attempt_index": attempt,
                        "original_sql": record["predicted_sql"],
                        "repaired_sql": validation["cleaned_sql"],
                        "accepted": validation["accepted"],
                        "reason": validation["reason"],
                        "failed_identifier": classification.failed_identifier,
                        "identifier_candidates": log_entry["candidate_scores"],
                        "chosen_suggestion": classification.suggested_fix,
                        "candidate_table_name": classification.correct_table,
                        "quarantine_reasons": sorted(set(
                            validation["quarantine_reasons"] +
                            (["low_confidence_identifier_match"] if low_confidence_identifier else [])
                        )),
                        "structure_issues": validation["structure_issues"],
                        "structure_metrics": validation["structure_metrics"],
                    })

                if validation["accepted"]:
                    stats["repair_accepted"] += 1
                    stats["by_failure_type"][classification.failure_type]["accepted"] += 1
                    log_entry["final_accepted"] = True
                    log_entry["final_sql"] = validation["cleaned_sql"]
                    log_entry["final_reason"] = validation["reason"]
                    repair_log.append(log_entry)
                    finalize_prediction(
                        idx,
                        record["question_id"],
                        record["db_id"],
                        validation["cleaned_sql"],
                        repaired=True,
                        original_sql=record["predicted_sql"],
                    )
                    progress.update(1)
                elif attempt + 1 < max_repair_attempts:
                    next_pending.append(item)
                else:
                    stats["repair_rejected"] += 1
                    log_entry["final_reason"] = validation["reason"]
                    last_reason = validation["reason"].lower()
                    if "diff" in last_reason or ">50%" in last_reason:
                        log_entry["rejection_category"] = "high_diff"
                    elif "schema" in last_reason:
                        log_entry["rejection_category"] = "schema_invalid"
                    elif "execution" in last_reason or "failed" in last_reason:
                        log_entry["rejection_category"] = "exec_failed"
                    elif "hygiene" in last_reason:
                        log_entry["rejection_category"] = "hygiene_failed"
                    repair_log.append(log_entry)
                    finalize_prediction(
                        idx,
                        record["question_id"],
                        record["db_id"],
                        record["predicted_sql"],
                        repaired=False,
                    )
                    progress.update(1)

                progress.set_postfix(
                    accepted=stats["repair_accepted"],
                    rejected=stats["repair_rejected"],
                    pending=len(next_pending),
                    refresh=False,
                )

        pending = next_pending

    progress.close()
    
    # Convert defaultdict for JSON serialization
    stats["by_failure_type"] = dict(stats["by_failure_type"])
    stats["quarantined_repairs"] = len(quarantined_repairs)
    
    return [pred for pred in repaired_predictions if pred is not None], repair_log, quarantined_repairs, stats


# =============================================================================
# Output Writing
# =============================================================================

def write_outputs(
    output_dir: str,
    repaired_predictions: List[Dict],
    repair_log: List[Dict],
    quarantined_repairs: List[Dict],
    stats: Dict,
    original_predictions: List[Dict],
):
    """Write all output files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Repaired predictions (for evaluation)
    # Format compatible with evaluate_t12.py
    pred_path = output_dir / "repaired_predictions_t12.jsonl"
    with open(pred_path, 'w') as f:
        for pred in repaired_predictions:
            # Merge with original for complete record
            orig = next((p for p in original_predictions if p["question_id"] == pred["question_id"]), {})
            record = {
                "question_id": pred["question_id"],
                "db_id": pred["db_id"],
                "question": orig.get("question", ""),
                "predicted_sql": pred["predicted_sql"],
                "gold_sql": orig.get("gold_sql", ""),
                "difficulty": orig.get("difficulty", "unknown"),
                "repaired": pred.get("repaired", False),
            }
            f.write(json.dumps(record) + '\n')
    print(f"Repaired predictions: {pred_path}")
    
    # Repair log
    log_path = output_dir / "repair_log_t12.jsonl"
    with open(log_path, 'w') as f:
        for entry in repair_log:
            f.write(json.dumps(entry) + '\n')
    print(f"Repair log: {log_path}")
    
    # Summary stats
    summary_path = output_dir / "repair_summary_t12.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Summary: {summary_path}")

    quarantine_path = output_dir / "quarantined_repairs_t12.jsonl"
    with open(quarantine_path, 'w') as f:
        for entry in quarantined_repairs:
            f.write(json.dumps(entry) + '\n')
    print(f"Quarantined repairs: {quarantine_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="T12 Error-Correction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input files
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions JSONL file")
    parser.add_argument("--eval_results", required=True,
                        help="Path to per_example_results JSONL file")
    parser.add_argument("--prompts", required=True,
                        help="Path to bird_dev_t12.jsonl (for hints)")
    parser.add_argument("--db_dir", required=True,
                        help="Path to database directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    
    # Model settings
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-2B",
                        help="Model ID for repair")
    parser.add_argument("--adapter_path", default=None,
                        help="Optional PEFT/LoRA adapter path for repair model")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode for repair")
    
    # Repair settings
    parser.add_argument("--max_repair_attempts", type=int, default=2,
                        help="Max repair attempts per example")
    parser.add_argument("--min_repairability_score", type=float, default=0.5,
                        help="Deprecated: ignored in V2 (all non-correct examples are attempted)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per repair")
    parser.add_argument("--generation_batch_size", type=int, default=8,
                        help="Number of repair prompts to generate in parallel on the GPU")
    
    # Limits
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of examples (0=all)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("T12 Error-Correction Pipeline")
    print("=" * 60)
    print(f"Predictions: {args.predictions}")
    print(f"Eval results: {args.eval_results}")
    print(f"Model: {args.model_id}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Thinking: {args.enable_thinking}")
    print(f"Max attempts: {args.max_repair_attempts}")
    print(f"Min repairability (ignored in V2): {args.min_repairability_score}")
    print(f"Generation batch size: {args.generation_batch_size}")
    print()
    
    # Load data
    print("Loading data...")
    predictions = load_jsonl(args.predictions)
    eval_results = load_jsonl(args.eval_results)
    prompts_by_id = load_prompts_by_id(args.prompts)
    print(f"  Loaded {len(predictions)} predictions")
    print(f"  Loaded {len(eval_results)} eval results")
    print(f"  Loaded {len(prompts_by_id)} prompts")
    
    # Merge data
    records = merge_data(predictions, eval_results, prompts_by_id)
    
    if args.limit > 0:
        records = records[:args.limit]
        print(f"  Limited to {len(records)} examples")
    
    # Initialize schema cache
    schema_cache = SchemaCache(args.db_dir)
    
    # Load model
    model, tokenizer = load_model(args.model_id, args.device, args.adapter_path)
    
    # Run error correction
    print("\nRunning error correction...")
    start_time = time.time()
    
    if args.max_repair_attempts > 2:
        raise ValueError("V1 caps max_repair_attempts at 2")

    repaired_predictions, repair_log, quarantined_repairs, stats = run_error_correction(
        records=records,
        schema_cache=schema_cache,
        model=model,
        tokenizer=tokenizer,
        enable_thinking=args.enable_thinking,
        max_repair_attempts=args.max_repair_attempts,
        min_repairability_score=args.min_repairability_score,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        device=args.device,
    )
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    
    # Write outputs
    print("\nWriting outputs...")
    write_outputs(
        output_dir=args.output_dir,
        repaired_predictions=repaired_predictions,
        repair_log=repair_log,
        quarantined_repairs=quarantined_repairs,
        stats=stats,
        original_predictions=predictions,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total examples: {stats['total']}")
    print(f"Correct (no repair needed): {stats['correct']}")
    print(f"Exec failed: {stats['exec_failed']}")
    print(f"Wrong result (from baseline eval): {stats['wrong_result']}")
    print()
    print(f"Repair attempted: {stats['repair_attempted']}")
    print(f"Repair accepted: {stats['repair_accepted']}")
    print(f"Repair rejected: {stats['repair_rejected']}")
    print(f"Repair skipped: {stats['repair_skipped']}")
    print(f"Quarantined repairs: {stats['quarantined_repairs']}")
    print()
    print("By failure type:")
    for ftype, counts in stats["by_failure_type"].items():
        print(f"  {ftype}: {counts['accepted']}/{counts['attempted']} accepted")
    
    # Expected improvement
    new_correct = stats['correct'] + stats['repair_accepted']
    orig_ex = stats['correct'] / stats['total'] * 100
    new_ex = new_correct / stats['total'] * 100
    print()
    print(f"Original EX (correct only): {orig_ex:.2f}%")
    print(f"Repaired EX (gold-validated during repair): {new_ex:.2f}% (+{new_ex - orig_ex:.2f}%)")
    print(stats["repair_scope"])


if __name__ == "__main__":
    main()

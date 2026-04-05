#!/usr/bin/env python3
"""
T10 Error-Correction Pipeline

Main script to run the error-correction pipeline on T10 predictions.

Features:
- Classifies failures by type
- Extracts relevant schema blocks
- Generates repair prompts
- Runs repair model inference with thinking enabled
- Validates and accepts/rejects repairs
- Produces repaired predictions and detailed logs

Usage:
    python run_error_correction.py \
        --predictions runs/t10_baseline_3090/qwen3-1.7b/without-sampling/predictions/predictions_t10.jsonl \
        --eval_results runs/t10_baseline_3090/qwen3-1.7b/without-sampling/eval/per_example_results.jsonl \
        --prompts data/training/t10/bird_dev_t10.jsonl \
        --db_dir data/bird_eval_datasets/dev_databases \
        --output_dir data/training/t10/error-correction \
        --model_id Qwen/Qwen3-1.7B \
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

def load_model(model_id: str, device: str = "cuda"):
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
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer


# =============================================================================
# Repair Generation
# =============================================================================

def generate_repair(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Generate repair SQL from messages."""
    import torch
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        add_special_tokens=False,
    ).to(device)
    
    # Generate
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
    
    # Decode
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    return raw_output


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
    
    repaired_predictions = []
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
        "repair_scope": "V1 repairs only non-executing SQL; wrong-result cases are classified but not auto-repaired.",
        "by_failure_type": defaultdict(lambda: {"attempted": 0, "accepted": 0}),
    }
    
    for i, record in enumerate(records):
        qid = record["question_id"]
        db_id = record["db_id"]
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(records)}] Processed {i + 1} examples, "
                  f"{stats['repair_accepted']} repairs accepted")
        
        # Skip correct predictions
        if record.get("correct"):
            stats["correct"] += 1
            repaired_predictions.append({
                "question_id": qid,
                "db_id": db_id,
                "predicted_sql": record["predicted_sql"],
                "repaired": False,
            })
            continue
        
        # Count failure types
        if record.get("exec_failed"):
            stats["exec_failed"] += 1
        if record.get("wrong_result"):
            stats["wrong_result"] += 1
        
        # Load schema
        schema = schema_cache.get_schema(db_id)
        if not schema:
            print(f"Warning: Could not load schema for {db_id}")
            repaired_predictions.append({
                "question_id": qid,
                "db_id": db_id,
                "predicted_sql": record["predicted_sql"],
                "repaired": False,
            })
            continue
        
        # Classify failure
        classification = classify_failure(record, schema)
        
        # Skip non-repairable (wrong_result or low score)
        if not classification or classification.repairability_score < min_repairability_score:
            stats["repair_skipped"] += 1
            repaired_predictions.append({
                "question_id": qid,
                "db_id": db_id,
                "predicted_sql": record["predicted_sql"],
                "repaired": False,
            })
            continue
        
        # Also skip wrong_result explicitly (defense in depth)
        if classification.failure_type == "wrong_result_non_exec_failure":
            stats["repair_skipped"] += 1
            repaired_predictions.append({
                "question_id": qid,
                "db_id": db_id,
                "predicted_sql": record["predicted_sql"],
                "repaired": False,
            })
            continue
        
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
        
        # Attempt repairs
        final_sql = record["predicted_sql"]
        repair_accepted = False
        
        for attempt in range(max_repair_attempts):
            if attempt == 0:
                # First attempt - use type-specific prompt
                messages = prompt_builder.build_messages(
                    failure_type=classification.failure_type,
                    schema_block=schema_block,
                    question=record["question"],
                    hints=record.get("evidence", ""),
                    predicted_sql=record["predicted_sql"],
                    error_message=record.get("pred_error", ""),
                    failed_identifier=classification.failed_identifier,
                    suggested_fix=classification.suggested_fix,
                    wrong_alias=classification.wrong_alias,
                    correct_table=classification.correct_table,
                )
            else:
                # Escalation attempt - re-extract schema with expanded context
                schema_block = extract_relevant_schema(
                    schema=schema,
                    question=record["question"],
                    hints=record.get("evidence", ""),
                    predicted_sql=record["predicted_sql"],
                    error=record.get("pred_error", ""),
                    expanded=True,
                )
                prev_attempt = log_entry["attempts"][-1]
                messages = escalation_builder.build_messages(
                    schema_block=schema_block,
                    question=record["question"],
                    hints=record.get("evidence", ""),
                    predicted_sql=record["predicted_sql"],
                    original_error=record.get("pred_error", ""),
                    first_repair_sql=prev_attempt["repaired_sql"],
                    first_repair_error=prev_attempt.get("exec_error", "unknown error"),
                )
            
            # Generate repair
            raw_output = generate_repair(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            
            # Validate repair
            validation = validate_repair(
                original_sql=record["predicted_sql"],
                raw_repair_output=raw_output,
                db_path=db_path,
                schema=schema,
                failure_type=classification.failure_type,
                identifier_confidence=(
                    classification.identifier_candidates[0]["score"]
                    if classification.identifier_candidates else None
                ),
            )
            
            # Log attempt
            attempt_log = {
                "attempt_index": attempt,
                "context_variant": "expanded" if attempt > 0 else "standard",
                "raw_output": raw_output[:500],  # Truncate for log
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
            }
            log_entry["attempts"].append(attempt_log)

            low_confidence_identifier = bool(
                classification.identifier_candidates
                and classification.identifier_candidates[0]["score"] < 0.8
            )
            if validation["quarantine"] or low_confidence_identifier:
                quarantined_repairs.append({
                    "question_id": qid,
                    "db_id": db_id,
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
            
            # Check if accepted
            if validation["accepted"]:
                repair_accepted = True
                final_sql = validation["cleaned_sql"]
                break
        
        # Update stats and log
        if repair_accepted:
            stats["repair_accepted"] += 1
            stats["by_failure_type"][classification.failure_type]["accepted"] += 1
            log_entry["final_accepted"] = True
            log_entry["final_sql"] = final_sql
            log_entry["final_reason"] = log_entry["attempts"][-1]["reason"]
        else:
            stats["repair_rejected"] += 1
            log_entry["final_reason"] = log_entry["attempts"][-1]["reason"] if log_entry["attempts"] else "no attempts"
            # Categorize rejection reason
            if log_entry["attempts"]:
                last_reason = log_entry["attempts"][-1]["reason"].lower()
                if "diff" in last_reason or ">50%" in last_reason:
                    log_entry["rejection_category"] = "high_diff"
                elif "schema" in last_reason:
                    log_entry["rejection_category"] = "schema_invalid"
                elif "execution" in last_reason or "failed" in last_reason:
                    log_entry["rejection_category"] = "exec_failed"
                elif "hygiene" in last_reason:
                    log_entry["rejection_category"] = "hygiene_failed"
        
        repair_log.append(log_entry)
        
        repaired_predictions.append({
            "question_id": qid,
            "db_id": db_id,
            "predicted_sql": final_sql,
            "repaired": repair_accepted,
            "original_sql": record["predicted_sql"] if repair_accepted else None,
        })
    
    # Convert defaultdict for JSON serialization
    stats["by_failure_type"] = dict(stats["by_failure_type"])
    stats["quarantined_repairs"] = len(quarantined_repairs)
    
    return repaired_predictions, repair_log, quarantined_repairs, stats


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
    # Format compatible with evaluate_t10.py
    pred_path = output_dir / "repaired_predictions_t10.jsonl"
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
    log_path = output_dir / "repair_log_t10.jsonl"
    with open(log_path, 'w') as f:
        for entry in repair_log:
            f.write(json.dumps(entry) + '\n')
    print(f"Repair log: {log_path}")
    
    # Summary stats
    summary_path = output_dir / "repair_summary_t10.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Summary: {summary_path}")

    quarantine_path = output_dir / "quarantined_repairs_t10.jsonl"
    with open(quarantine_path, 'w') as f:
        for entry in quarantined_repairs:
            f.write(json.dumps(entry) + '\n')
    print(f"Quarantined repairs: {quarantine_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="T10 Error-Correction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input files
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions JSONL file")
    parser.add_argument("--eval_results", required=True,
                        help="Path to per_example_results JSONL file")
    parser.add_argument("--prompts", required=True,
                        help="Path to bird_dev_t10.jsonl (for hints)")
    parser.add_argument("--db_dir", required=True,
                        help="Path to database directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    
    # Model settings
    parser.add_argument("--model_id", default="Qwen/Qwen3-1.7B",
                        help="Model ID for repair")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode for repair")
    
    # Repair settings
    parser.add_argument("--max_repair_attempts", type=int, default=2,
                        help="Max repair attempts per example")
    parser.add_argument("--min_repairability_score", type=float, default=0.5,
                        help="Minimum repairability score to attempt repair")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per repair")
    
    # Limits
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of examples (0=all)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("T10 Error-Correction Pipeline")
    print("=" * 60)
    print(f"Predictions: {args.predictions}")
    print(f"Eval results: {args.eval_results}")
    print(f"Model: {args.model_id}")
    print(f"Thinking: {args.enable_thinking}")
    print(f"Max attempts: {args.max_repair_attempts}")
    print(f"Min repairability: {args.min_repairability_score}")
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
    model, tokenizer = load_model(args.model_id, args.device)
    
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
    print(f"Wrong result (not repaired): {stats['wrong_result']}")
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
    print(f"Potential new EX: {new_ex:.2f}% (+{new_ex - orig_ex:.2f}%)")
    print("(Note: Actual EX requires re-evaluation against gold)")
    print(stats["repair_scope"])


if __name__ == "__main__":
    main()

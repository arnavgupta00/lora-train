#!/usr/bin/env python3
"""
Error-Correction Evaluation for BIRD Benchmark

Implements an error-correction loop:
1. Generate SQL with SFT model (no_think mode, fast)
2. Execute SQL
3. If error, use thinking mode to generate correction
4. Retry up to max_retries times

Usage:
    python evaluation/eval_error_correction.py \
        --model_id "Qwen/Qwen3-1.7B" \
        --adapter_dir "./outputs/sft_adapter" \
        --bird_dev_json "./bird_eval/dev.json" \
        --db_dir "./bird_eval/dev_databases" \
        --output_dir "./eval_error_correction" \
        --max_retries 3
"""

import argparse
import json
import os
import re
import sqlite3
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Auto-detect HuggingFace cache location
if "HF_HOME" not in os.environ:
    for cache_path in ["/workspace/hf", "/runpod-volume/hf", os.path.expanduser("~/.cache/huggingface")]:
        if os.path.isdir(cache_path):
            os.environ["HF_HOME"] = cache_path
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_path, "datasets")
            break

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if "HF_HOME" in os.environ:
    logger.info(f"Using HuggingFace cache: {os.environ['HF_HOME']}")

import sys
sys.stdout = sys.stderr


def get_ddl_schema_from_db(db_path: str) -> str:
    """Extract CREATE TABLE statements from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND sql IS NOT NULL
            ORDER BY name
        """)
        create_statements = []
        for row in cursor.fetchall():
            if row[0]:
                sql = ' '.join(row[0].strip().split())
                create_statements.append(sql + ";")
        conn.close()
        return "\n".join(create_statements)
    except Exception as e:
        logger.warning(f"Could not extract DDL schema: {e}")
        return ""


def find_database(db_dir: str, db_id: str) -> Optional[str]:
    """Find SQLite database file."""
    db_file = Path(db_dir) / db_id / f"{db_id}.sqlite"
    if db_file.exists():
        return str(db_file)
    for pattern in [f"*/{db_id}.sqlite", f"*/{db_id}.db"]:
        matches = list(Path(db_dir).glob(pattern))
        if matches:
            return str(matches[0])
    return None


def normalize_sql(sql: str) -> str:
    """Clean up generated SQL."""
    if not sql:
        return ""
    sql = sql.strip()
    
    # Handle Qwen3 thinking tags: extract SQL after </think>
    if "</think>" in sql:
        sql = sql.split("</think>")[-1].strip()
    elif "<think>" in sql:
        # Thinking started but never ended - likely truncated
        match = re.search(r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s', sql, re.IGNORECASE)
        if match:
            sql = sql[match.start():]
        else:
            sql = ""
    
    # Handle markdown code blocks
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.startswith("```"))
    sql = sql.strip()
    
    # Take only first statement
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    return sql


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[bool, Any]:
    """Execute SQL and return (success, results_or_error)."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def results_match(results1: Any, results2: Any) -> bool:
    """Check if two SQL result sets match (order-independent)."""
    if results1 is None or results2 is None:
        return False
    try:
        set1 = set(tuple(row) for row in results1)
        set2 = set(tuple(row) for row in results2)
        return set1 == set2
    except Exception:
        return False


def categorize_error(error_msg: str) -> str:
    """Categorize SQL error by type."""
    error_lower = error_msg.lower()
    if "no such column" in error_lower:
        return "column_not_found"
    elif "no such table" in error_lower:
        return "table_not_found"
    elif "syntax error" in error_lower:
        return "syntax_error"
    elif "ambiguous column" in error_lower:
        return "ambiguous_column"
    elif "unrecognized token" in error_lower:
        return "unrecognized_token"
    elif "misuse of aggregate" in error_lower:
        return "aggregate_misuse"
    elif "near" in error_lower:
        return "syntax_near"
    else:
        return "other"


class ErrorCorrectionEvaluator:
    """BIRD evaluator with error-correction loop using thinking mode."""
    
    SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Only output the SQL query, nothing else."""
    
    CORRECTION_PROMPT = """/think
The following SQL query failed with an error. Analyze the error and fix the query.

## Database Schema
{schema}

## Question
{question}

## Evidence/Hint
{evidence}

## Failed SQL
```sql
{failed_sql}
```

## Error Message
{error_message}

## Analysis Instructions
1. Read the error message carefully - it tells you exactly what's wrong
2. Find the problematic column/table name in the failed SQL
3. Search the schema above for the CORRECT column name (exact spelling, case, spaces)
4. If a column name has spaces or special characters, wrap it in backticks: `Column Name`
5. Verify all table aliases (T1, T2) are used correctly
6. Check JOIN conditions reference correct columns from each table

## Corrected SQL
Generate ONLY the corrected SQL query (no explanation):
"""
    
    def __init__(
        self,
        model,
        tokenizer,
        db_dir: str,
        batch_size: int = 16,
        initial_max_tokens: int = 256,
        correction_max_tokens: int = 1024,
        max_retries: int = 3,
        num_workers: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_dir = db_dir
        self.batch_size = batch_size
        self.initial_max_tokens = initial_max_tokens
        self.correction_max_tokens = correction_max_tokens
        self.max_retries = max_retries
        self.num_workers = num_workers
        
        # Pre-cache schemas
        self.schema_cache: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            "initial_correct": 0,
            "initial_errors": 0,
            "initial_wrong": 0,
            "fixed_at_attempt": defaultdict(int),
            "unfixable": 0,
            "error_categories": defaultdict(int),
            "correction_attempts": 0,
        }
    
    def _get_schema(self, db_id: str) -> str:
        """Get cached schema for database."""
        if db_id not in self.schema_cache:
            db_path = find_database(self.db_dir, db_id)
            if db_path:
                self.schema_cache[db_id] = get_ddl_schema_from_db(db_path)
            else:
                self.schema_cache[db_id] = ""
        return self.schema_cache[db_id]
    
    def _build_initial_prompt(self, question: str, schema: str, evidence: str = "") -> str:
        """Build prompt for initial SQL generation (no_think mode)."""
        user_content = "/no_think\n"
        user_content += f"Schema:\n{schema}\n\n"
        if evidence:
            user_content += f"Hint: {evidence}\n\n"
        user_content += f"Question: {question}"
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def _build_correction_prompt(
        self,
        question: str,
        schema: str,
        evidence: str,
        failed_sql: str,
        error_message: str,
    ) -> str:
        """Build prompt for error correction (with thinking mode)."""
        user_content = self.CORRECTION_PROMPT.format(
            schema=schema,
            question=question,
            evidence=evidence if evidence else "None",
            failed_sql=failed_sql,
            error_message=error_message,
        )
        
        messages = [
            {"role": "system", "content": "You are an expert SQL debugger. Fix SQL errors based on schema and error messages."},
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def generate_initial_batch(self, examples: List[Dict]) -> List[str]:
        """Generate initial SQL predictions (no_think mode, batched)."""
        prompts = []
        for ex in examples:
            schema = self._get_schema(ex.get("db_id", ""))
            prompt = self._build_initial_prompt(
                question=ex.get("question", ""),
                schema=schema,
                evidence=ex.get("evidence", ""),
            )
            prompts.append(prompt)
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.initial_max_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
            )
        
        predictions = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output[input_len:]
            pred_sql = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            predictions.append(normalize_sql(pred_sql))
        
        return predictions
    
    def generate_correction(self, example: Dict, failed_sql: str, error_message: str) -> str:
        """Generate corrected SQL using thinking mode."""
        schema = self._get_schema(example.get("db_id", ""))
        prompt = self._build_correction_prompt(
            question=example.get("question", ""),
            schema=schema,
            evidence=example.get("evidence", ""),
            failed_sql=failed_sql,
            error_message=error_message,
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.correction_max_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
            )
        
        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][input_len:]
        corrected_sql = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        self.stats["correction_attempts"] += 1
        return normalize_sql(corrected_sql)
    
    def generate_corrections_batch(
        self,
        error_examples: List[Tuple[Dict, str, str]],  # (example, failed_sql, error_msg)
    ) -> List[str]:
        """Generate corrections for a batch of errors (thinking mode)."""
        prompts = []
        for example, failed_sql, error_message in error_examples:
            schema = self._get_schema(example.get("db_id", ""))
            prompt = self._build_correction_prompt(
                question=example.get("question", ""),
                schema=schema,
                evidence=example.get("evidence", ""),
                failed_sql=failed_sql,
                error_message=error_message,
            )
            prompts.append(prompt)
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.correction_max_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
            )
        
        corrections = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output[input_len:]
            corrected_sql = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            corrections.append(normalize_sql(corrected_sql))
        
        self.stats["correction_attempts"] += len(corrections)
        return corrections
    
    def evaluate_single(
        self,
        pred_sql: str,
        gold_sql: str,
        db_id: str,
    ) -> Dict[str, Any]:
        """Evaluate a single SQL prediction."""
        db_path = find_database(self.db_dir, db_id)
        
        result = {
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "db_id": db_id,
            "exact_match": pred_sql.lower().strip() == gold_sql.lower().strip(),
            "exec_match": False,
            "pred_executes": False,
            "error": None,
            "error_category": None,
        }
        
        if not db_path:
            result["error"] = "Database not found"
            result["error_category"] = "db_not_found"
            return result
        
        # Execute gold
        gold_ok, gold_result = execute_sql(db_path, gold_sql)
        if not gold_ok:
            result["error"] = f"Gold SQL error: {gold_result}"
            result["error_category"] = "gold_error"
            return result
        
        # Execute prediction
        pred_ok, pred_result = execute_sql(db_path, pred_sql)
        result["pred_executes"] = pred_ok
        
        if not pred_ok:
            result["error"] = str(pred_result)
            result["error_category"] = categorize_error(str(pred_result))
            return result
        
        # Compare results
        result["exec_match"] = results_match(gold_result, pred_result)
        return result
    
    def run_evaluation(
        self,
        examples: List[Dict],
        output_dir: str,
        correction_batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Run full evaluation with error-correction loop."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Evaluating {len(examples)} examples with error-correction (max_retries={self.max_retries})")
        
        # Pre-cache schemas
        logger.info("Pre-caching database schemas...")
        db_ids = set(ex.get("db_id", "") for ex in examples)
        for db_id in db_ids:
            if db_id:
                self._get_schema(db_id)
        logger.info(f"Cached {len(self.schema_cache)} schemas")
        
        # ====== PHASE 1: Initial Generation ======
        logger.info(f"[PHASE 1] Initial SQL generation (batch_size={self.batch_size}, no_think mode)...")
        initial_predictions = []
        start_time = time.time()
        
        for batch_start in range(0, len(examples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(examples))
            batch = examples[batch_start:batch_end]
            
            batch_preds = self.generate_initial_batch(batch)
            initial_predictions.extend(batch_preds)
            
            elapsed = time.time() - start_time
            rate = len(initial_predictions) / elapsed
            remaining = (len(examples) - len(initial_predictions)) / rate if rate > 0 else 0
            
            logger.info(
                f"  [{len(initial_predictions)}/{len(examples)}] "
                f"{100*len(initial_predictions)/len(examples):.1f}% - "
                f"{rate:.1f} ex/s - ETA: {remaining/60:.1f}min"
            )
        
        initial_gen_time = time.time() - start_time
        logger.info(f"Initial generation complete: {initial_gen_time/60:.1f}min")
        
        # ====== PHASE 2: Initial Evaluation ======
        logger.info(f"[PHASE 2] Evaluating initial predictions...")
        eval_start = time.time()
        
        results = []
        errors_to_correct = []  # List of (index, example, failed_sql, error)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, (ex, pred) in enumerate(zip(examples, initial_predictions)):
                gold_sql = normalize_sql(ex.get("SQL", ex.get("sql", "")))
                future = executor.submit(
                    self.evaluate_single,
                    pred, gold_sql, ex.get("db_id", "")
                )
                futures.append((i, ex, pred, future))
            
            for i, ex, pred, future in futures:
                result = future.result()
                result["question"] = ex.get("question", "")
                result["index"] = i
                result["initial_sql"] = pred
                result["correction_attempts"] = []
                result["final_sql"] = pred
                result["corrected"] = False
                result["corrected_at_attempt"] = None
                
                if result["exec_match"]:
                    self.stats["initial_correct"] += 1
                elif not result["pred_executes"]:
                    self.stats["initial_errors"] += 1
                    if result["error_category"]:
                        self.stats["error_categories"][result["error_category"]] += 1
                    errors_to_correct.append((i, ex, pred, result["error"]))
                else:
                    self.stats["initial_wrong"] += 1
                
                results.append(result)
        
        initial_eval_time = time.time() - eval_start
        logger.info(f"Initial evaluation: {initial_eval_time:.1f}s")
        logger.info(f"  Correct: {self.stats['initial_correct']} ({100*self.stats['initial_correct']/len(examples):.1f}%)")
        logger.info(f"  Errors (to fix): {len(errors_to_correct)} ({100*len(errors_to_correct)/len(examples):.1f}%)")
        logger.info(f"  Wrong results: {self.stats['initial_wrong']} ({100*self.stats['initial_wrong']/len(examples):.1f}%)")
        
        # ====== PHASE 3: Error Correction Loop ======
        if errors_to_correct:
            logger.info(f"[PHASE 3] Error correction loop (max_retries={self.max_retries})...")
            correction_start = time.time()
            
            # Track which errors are still pending
            pending_corrections = list(errors_to_correct)
            
            for attempt in range(1, self.max_retries + 1):
                if not pending_corrections:
                    break
                
                logger.info(f"  Correction attempt {attempt}/{self.max_retries}: {len(pending_corrections)} errors...")
                
                # Generate corrections in batches
                all_corrections = []
                for batch_start in range(0, len(pending_corrections), correction_batch_size):
                    batch_end = min(batch_start + correction_batch_size, len(pending_corrections))
                    batch = pending_corrections[batch_start:batch_end]
                    
                    # Prepare batch: (example, failed_sql, error_msg)
                    batch_input = [(ex, sql, err) for _, ex, sql, err in batch]
                    corrections = self.generate_corrections_batch(batch_input)
                    all_corrections.extend(corrections)
                    
                    logger.info(f"    Generated corrections [{batch_start+1}-{batch_end}]/{len(pending_corrections)}")
                
                # Evaluate corrections
                still_pending = []
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for j, ((idx, ex, _, _), corrected_sql) in enumerate(zip(pending_corrections, all_corrections)):
                        gold_sql = normalize_sql(ex.get("SQL", ex.get("sql", "")))
                        future = executor.submit(
                            self.evaluate_single,
                            corrected_sql, gold_sql, ex.get("db_id", "")
                        )
                        futures.append((idx, ex, corrected_sql, future))
                    
                    for idx, ex, corrected_sql, future in futures:
                        eval_result = future.result()
                        
                        # Update the result entry
                        results[idx]["correction_attempts"].append({
                            "attempt": attempt,
                            "sql": corrected_sql,
                            "executes": eval_result["pred_executes"],
                            "error": eval_result["error"],
                            "exec_match": eval_result["exec_match"],
                        })
                        
                        if eval_result["exec_match"]:
                            # Fixed!
                            results[idx]["final_sql"] = corrected_sql
                            results[idx]["corrected"] = True
                            results[idx]["corrected_at_attempt"] = attempt
                            results[idx]["exec_match"] = True
                            results[idx]["pred_executes"] = True
                            results[idx]["error"] = None
                            self.stats["fixed_at_attempt"][attempt] += 1
                        elif eval_result["pred_executes"]:
                            # Executes but wrong results - stop trying
                            results[idx]["final_sql"] = corrected_sql
                            results[idx]["pred_executes"] = True
                            results[idx]["error"] = None
                        else:
                            # Still error - add to next retry
                            still_pending.append((idx, ex, corrected_sql, eval_result["error"]))
                
                fixed_this_round = len(pending_corrections) - len(still_pending)
                logger.info(f"    Fixed: {fixed_this_round}, Still errors: {len(still_pending)}")
                pending_corrections = still_pending
            
            # Mark unfixable
            self.stats["unfixable"] = len(pending_corrections)
            
            correction_time = time.time() - correction_start
            logger.info(f"Correction loop complete: {correction_time/60:.1f}min")
        
        # ====== PHASE 4: Compute Final Metrics ======
        logger.info("[PHASE 4] Computing final metrics...")
        
        exact_match = sum(1 for r in results if r["exact_match"])
        exec_match = sum(1 for r in results if r["exec_match"])
        exec_errors = sum(1 for r in results if not r["pred_executes"])
        corrected_count = sum(1 for r in results if r["corrected"])
        
        total = len(results)
        total_time = time.time() - start_time
        
        metrics = {
            "total_examples": total,
            "exact_match": exact_match,
            "exact_match_pct": round(100 * exact_match / total, 2) if total > 0 else 0,
            
            # Initial metrics (before correction)
            "initial_execution_match": self.stats["initial_correct"],
            "initial_execution_accuracy": round(100 * self.stats["initial_correct"] / total, 2) if total > 0 else 0,
            "initial_execution_errors": self.stats["initial_errors"],
            "initial_wrong_results": self.stats["initial_wrong"],
            
            # Final metrics (after correction)
            "final_execution_match": exec_match,
            "final_execution_accuracy": round(100 * exec_match / total, 2) if total > 0 else 0,
            "final_execution_errors": exec_errors,
            "final_execution_error_pct": round(100 * exec_errors / total, 2) if total > 0 else 0,
            
            # Improvement
            "accuracy_improvement": round(100 * (exec_match - self.stats["initial_correct"]) / total, 2) if total > 0 else 0,
            
            # Correction stats
            "correction_stats": {
                "errors_attempted": self.stats["initial_errors"],
                "fixed_total": corrected_count,
                "fixed_at_attempt_1": self.stats["fixed_at_attempt"][1],
                "fixed_at_attempt_2": self.stats["fixed_at_attempt"][2],
                "fixed_at_attempt_3": self.stats["fixed_at_attempt"][3],
                "unfixable": self.stats["unfixable"],
                "total_correction_attempts": self.stats["correction_attempts"],
            },
            
            # Error categories
            "error_categories": dict(self.stats["error_categories"]),
            
            # Timing
            "initial_generation_time_min": round(initial_gen_time / 60, 2),
            "correction_time_min": round((total_time - initial_gen_time - initial_eval_time) / 60, 2),
            "total_time_min": round(total_time / 60, 2),
        }
        
        # Per-database breakdown
        db_stats = defaultdict(lambda: {"total": 0, "initial_match": 0, "final_match": 0, "corrected": 0})
        for r in results:
            db_id = r["db_id"]
            db_stats[db_id]["total"] += 1
            if r["exec_match"]:
                db_stats[db_id]["final_match"] += 1
            if r["corrected"]:
                db_stats[db_id]["corrected"] += 1
        
        # Calculate initial matches from results that aren't corrected
        for r in results:
            if r["exec_match"] and not r["corrected"]:
                db_stats[r["db_id"]]["initial_match"] += 1
        
        db_breakdown = []
        for db_id, stats in sorted(db_stats.items(), key=lambda x: -x[1]["total"]):
            db_breakdown.append({
                "db_id": db_id,
                "total": stats["total"],
                "initial_match": stats["initial_match"],
                "final_match": stats["final_match"],
                "corrected": stats["corrected"],
                "initial_accuracy": round(100 * stats["initial_match"] / stats["total"], 1) if stats["total"] > 0 else 0,
                "final_accuracy": round(100 * stats["final_match"] / stats["total"], 1) if stats["total"] > 0 else 0,
            })
        
        # Sample corrections
        sample_corrections = [r for r in results if r["corrected"]][:10]
        sample_unfixed = [r for r in results if not r["pred_executes"] and not r["corrected"]][:10]
        
        # Build report
        report = {
            **metrics,
            "by_database": db_breakdown,
            "sample_corrections": sample_corrections,
            "sample_unfixed": sample_unfixed,
        }
        
        # Save results
        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump([
                {
                    "db_id": r["db_id"],
                    "question": r["question"],
                    "initial_sql": r["initial_sql"],
                    "final_sql": r["final_sql"],
                    "gold_sql": r["gold_sql"],
                    "corrected": r["corrected"],
                    "corrected_at_attempt": r["corrected_at_attempt"],
                    "exec_match": r["exec_match"],
                }
                for r in results
            ], f, indent=2)
        
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Full results with correction attempts
        full_results_path = os.path.join(output_dir, "full_results.json")
        with open(full_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(metrics, db_breakdown)
        
        logger.info(f"Results saved to: {output_dir}")
        return report
    
    def _print_summary(self, metrics: Dict, db_breakdown: List[Dict]):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("BIRD BENCHMARK - ERROR CORRECTION EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nTotal examples:           {metrics['total_examples']}")
        print()
        
        print("BEFORE CORRECTION:")
        print(f"  Execution accuracy:     {metrics['initial_execution_match']} ({metrics['initial_execution_accuracy']}%)")
        print(f"  Execution errors:       {metrics['initial_execution_errors']}")
        print(f"  Wrong results:          {metrics['initial_wrong_results']}")
        print()
        
        print("AFTER CORRECTION:")
        print(f"  Execution accuracy:     {metrics['final_execution_match']} ({metrics['final_execution_accuracy']}%)")
        print(f"  Execution errors:       {metrics['final_execution_errors']} ({metrics['final_execution_error_pct']}%)")
        print()
        
        print(f"IMPROVEMENT:              +{metrics['accuracy_improvement']}%")
        print()
        
        cs = metrics['correction_stats']
        print("CORRECTION BREAKDOWN:")
        print(f"  Errors attempted:       {cs['errors_attempted']}")
        print(f"  Fixed at attempt 1:     {cs['fixed_at_attempt_1']}")
        print(f"  Fixed at attempt 2:     {cs['fixed_at_attempt_2']}")
        print(f"  Fixed at attempt 3:     {cs['fixed_at_attempt_3']}")
        print(f"  Total fixed:            {cs['fixed_total']}")
        print(f"  Unfixable:              {cs['unfixable']}")
        print()
        
        print("ERROR CATEGORIES:")
        for cat, count in sorted(metrics['error_categories'].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
        print()
        
        print("=" * 70)
        print("TOP 5 DATABASES (AFTER CORRECTION)")
        print("=" * 70)
        for db in sorted(db_breakdown, key=lambda x: -x["final_accuracy"])[:5]:
            improvement = db['final_accuracy'] - db['initial_accuracy']
            print(f"  {db['db_id']}: {db['final_accuracy']:.1f}% ({db['final_match']}/{db['total']}) [+{improvement:.1f}%]")
        
        print()
        print("TIMING:")
        print(f"  Initial generation:     {metrics['initial_generation_time_min']} min")
        print(f"  Correction loop:        {metrics['correction_time_min']} min")
        print(f"  Total time:             {metrics['total_time_min']} min")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="BIRD Benchmark - Error Correction Evaluation")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True,
                        help="Base model ID (e.g., Qwen/Qwen3-1.7B)")
    parser.add_argument("--adapter_dir", type=str, default="",
                        help="Path to LoRA adapter directory (optional)")
    
    # Data arguments
    parser.add_argument("--bird_dev_json", type=str, required=True,
                        help="Path to BIRD dev.json file")
    parser.add_argument("--db_dir", type=str, required=True,
                        help="Directory containing BIRD databases")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    
    # Generation arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for initial generation")
    parser.add_argument("--initial_max_tokens", type=int, default=256,
                        help="Max tokens for initial generation (no_think)")
    parser.add_argument("--correction_max_tokens", type=int, default=1024,
                        help="Max tokens for correction (with thinking)")
    parser.add_argument("--correction_batch_size", type=int, default=8,
                        help="Batch size for correction generation")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum correction attempts per error")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for SQL execution")
    
    # Misc arguments
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of examples (0=all)")
    
    args = parser.parse_args()
    
    # Load BIRD data
    logger.info(f"Loading BIRD data: {args.bird_dev_json}")
    with open(args.bird_dev_json, "r") as f:
        bird_data = json.load(f)
    
    if args.limit > 0:
        bird_data = bird_data[:args.limit]
        logger.info(f"Limited to {len(bird_data)} examples")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    logger.info(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    logger.info("Model weights loaded successfully")
    
    # Load adapter if provided
    if args.adapter_dir and os.path.isdir(args.adapter_dir):
        logger.info(f"Loading LoRA adapter: {args.adapter_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        logger.info("Adapter loaded successfully")
    
    model = model.to("cuda")
    model.eval()
    
    # Create evaluator
    evaluator = ErrorCorrectionEvaluator(
        model=model,
        tokenizer=tokenizer,
        db_dir=args.db_dir,
        batch_size=args.batch_size,
        initial_max_tokens=args.initial_max_tokens,
        correction_max_tokens=args.correction_max_tokens,
        max_retries=args.max_retries,
        num_workers=args.num_workers,
    )
    
    # Run evaluation
    report = evaluator.run_evaluation(
        examples=bird_data,
        output_dir=args.output_dir,
        correction_batch_size=args.correction_batch_size,
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: {report['final_execution_accuracy']}% accuracy")
    print(f"Improvement: +{report['accuracy_improvement']}% from error correction")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

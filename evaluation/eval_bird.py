#!/usr/bin/env python3
"""
BIRD Benchmark Evaluation Script

Unified evaluation script supporting:
- Baseline model (no adapter)
- SFT model (with LoRA adapter)
- GRPO model (with LoRA adapter)
- Batch generation for efficiency
- Parallel SQL execution

Usage:
    # Baseline evaluation
    python evaluation/eval_bird.py \
        --model_id "Qwen/Qwen3-1.7B" \
        --bird_dev_json "/path/to/bird/dev.json" \
        --db_dir "/path/to/bird/databases" \
        --output_dir "./results/baseline"
    
    # Adapter evaluation
    python evaluation/eval_bird.py \
        --model_id "Qwen/Qwen3-1.7B" \
        --adapter_dir "./outputs/sft/" \
        --bird_dev_json "/path/to/bird/dev.json" \
        --db_dir "/path/to/bird/databases" \
        --output_dir "./results/sft"
"""

import argparse
import json
import os
import sqlite3
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.startswith("```"))
    sql = sql.strip()
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


class BIRDEvaluator:
    """BIRD benchmark evaluator with batch generation."""
    
    SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Only output the SQL query, nothing else."""
    
    def __init__(
        self,
        model,
        tokenizer,
        db_dir: str,
        batch_size: int = 16,
        max_new_tokens: int = 256,
        num_workers: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_dir = db_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_workers = num_workers
        
        # Pre-cache schemas
        self.schema_cache: Dict[str, str] = {}
    
    def _get_schema(self, db_id: str) -> str:
        """Get cached schema for database."""
        if db_id not in self.schema_cache:
            db_path = find_database(self.db_dir, db_id)
            if db_path:
                self.schema_cache[db_id] = get_ddl_schema_from_db(db_path)
            else:
                self.schema_cache[db_id] = ""
        return self.schema_cache[db_id]
    
    def _build_prompt(self, question: str, schema: str, evidence: str = "") -> str:
        """Build prompt for SQL generation."""
        user_content = f"Schema:\n{schema}\n\n"
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
    
    def generate_batch(self, examples: List[Dict]) -> List[str]:
        """Generate SQL for a batch of examples."""
        prompts = []
        for ex in examples:
            schema = self._get_schema(ex.get("db_id", ""))
            prompt = self._build_prompt(
                question=ex.get("question", ""),
                schema=schema,
                evidence=ex.get("evidence", ""),
            )
            prompts.append(prompt)
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
            )
        
        # Decode
        predictions = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output[input_len:]
            pred_sql = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            predictions.append(normalize_sql(pred_sql))
        
        return predictions
    
    def evaluate_prediction(
        self,
        pred_sql: str,
        gold_sql: str,
        db_id: str,
    ) -> Dict[str, Any]:
        """Evaluate a single prediction."""
        db_path = find_database(self.db_dir, db_id)
        
        result = {
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "db_id": db_id,
            "exact_match": pred_sql.lower().strip() == gold_sql.lower().strip(),
            "exec_match": False,
            "pred_executes": False,
            "error": None,
        }
        
        if not db_path:
            result["error"] = "Database not found"
            return result
        
        # Execute gold
        gold_ok, gold_result = execute_sql(db_path, gold_sql)
        if not gold_ok:
            result["error"] = f"Gold SQL error: {gold_result}"
            return result
        
        # Execute prediction
        pred_ok, pred_result = execute_sql(db_path, pred_sql)
        result["pred_executes"] = pred_ok
        
        if not pred_ok:
            result["error"] = str(pred_result)
            return result
        
        # Compare results
        result["exec_match"] = results_match(gold_result, pred_result)
        return result
    
    def run_evaluation(
        self,
        examples: List[Dict],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Run full evaluation on BIRD examples."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Evaluating {len(examples)} examples...")
        
        # Pre-cache schemas
        logger.info("Pre-caching database schemas...")
        db_ids = set(ex.get("db_id", "") for ex in examples)
        for db_id in db_ids:
            if db_id:
                self._get_schema(db_id)
        logger.info(f"Cached {len(self.schema_cache)} schemas")
        
        # Generate predictions in batches
        logger.info(f"Generating SQL (batch_size={self.batch_size})...")
        predictions = []
        start_time = time.time()
        
        for batch_start in range(0, len(examples), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(examples))
            batch = examples[batch_start:batch_end]
            
            batch_preds = self.generate_batch(batch)
            predictions.extend(batch_preds)
            
            # Progress
            elapsed = time.time() - start_time
            rate = len(predictions) / elapsed
            remaining = (len(examples) - len(predictions)) / rate if rate > 0 else 0
            
            logger.info(
                f"[{len(predictions)}/{len(examples)}] "
                f"{100*len(predictions)/len(examples):.1f}% - "
                f"{rate:.1f} ex/s - ETA: {remaining/60:.1f}min"
            )
        
        gen_time = time.time() - start_time
        logger.info(f"Generation complete: {gen_time/60:.1f} min ({len(examples)/gen_time:.1f} ex/s)")
        
        # Evaluate predictions in parallel
        logger.info(f"Evaluating predictions (workers={self.num_workers})...")
        eval_start = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i, (ex, pred) in enumerate(zip(examples, predictions)):
                gold_sql = normalize_sql(ex.get("SQL", ex.get("sql", "")))
                future = executor.submit(
                    self.evaluate_prediction,
                    pred, gold_sql, ex.get("db_id", "")
                )
                futures.append((i, ex, future))
            
            for i, ex, future in futures:
                result = future.result()
                result["question"] = ex.get("question", "")
                result["index"] = i
                results.append(result)
        
        eval_time = time.time() - eval_start
        logger.info(f"Evaluation complete: {eval_time:.1f}s")
        
        # Compute metrics
        exact_match = sum(1 for r in results if r["exact_match"])
        exec_match = sum(1 for r in results if r["exec_match"])
        exec_errors = sum(1 for r in results if not r["pred_executes"])
        
        total = len(results)
        metrics = {
            "total_examples": total,
            "exact_match": exact_match,
            "exact_match_pct": round(100 * exact_match / total, 2) if total > 0 else 0,
            "execution_match": exec_match,
            "execution_accuracy": round(100 * exec_match / total, 2) if total > 0 else 0,
            "execution_errors": exec_errors,
            "execution_error_pct": round(100 * exec_errors / total, 2) if total > 0 else 0,
            "generation_time_min": round(gen_time / 60, 2),
            "evaluation_time_sec": round(eval_time, 2),
        }
        
        # Per-database breakdown
        db_stats = defaultdict(lambda: {"total": 0, "exec_match": 0, "errors": 0})
        for r in results:
            db_id = r["db_id"]
            db_stats[db_id]["total"] += 1
            if r["exec_match"]:
                db_stats[db_id]["exec_match"] += 1
            if not r["pred_executes"]:
                db_stats[db_id]["errors"] += 1
        
        db_breakdown = []
        for db_id, stats in sorted(db_stats.items(), key=lambda x: -x[1]["total"]):
            db_breakdown.append({
                "db_id": db_id,
                "total": stats["total"],
                "exec_match": stats["exec_match"],
                "errors": stats["errors"],
                "accuracy": round(100 * stats["exec_match"] / stats["total"], 1) if stats["total"] > 0 else 0,
            })
        
        # Sample errors and successes
        sample_errors = [r for r in results if not r["pred_executes"]][:10]
        sample_successes = [r for r in results if r["exec_match"]][:5]
        
        # Build report
        report = {
            **metrics,
            "by_database": db_breakdown,
            "sample_errors": sample_errors,
            "sample_successes": sample_successes,
        }
        
        # Save results
        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump([
                {
                    "db_id": ex.get("db_id", ""),
                    "question": ex.get("question", ""),
                    "predicted_sql": pred,
                    "gold_sql": normalize_sql(ex.get("SQL", ex.get("sql", ""))),
                }
                for ex, pred in zip(examples, predictions)
            ], f, indent=2)
        
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(metrics, db_breakdown)
        
        logger.info(f"Results saved to: {output_dir}")
        return report
    
    def _print_summary(self, metrics: Dict, db_breakdown: List[Dict]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("BIRD BENCHMARK EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total examples:     {metrics['total_examples']}")
        print(f"Exact match:        {metrics['exact_match']} ({metrics['exact_match_pct']}%)")
        print(f"Execution match:    {metrics['execution_match']} ({metrics['execution_accuracy']}%)")
        print(f"Execution errors:   {metrics['execution_errors']} ({metrics['execution_error_pct']}%)")
        print()
        
        print("=" * 60)
        print("TOP 5 DATABASES BY ACCURACY")
        print("=" * 60)
        for db in sorted(db_breakdown, key=lambda x: -x["accuracy"])[:5]:
            print(f"  {db['db_id']}: {db['accuracy']:.1f}% ({db['exec_match']}/{db['total']})")
        
        print()
        print("BOTTOM 3 DATABASES BY ACCURACY")
        for db in sorted(db_breakdown, key=lambda x: x["accuracy"])[:3]:
            print(f"  {db['db_id']}: {db['accuracy']:.1f}% ({db['exec_match']}/{db['total']})")
        
        print()
        print("=" * 60)
        print(f"Generation time:    {metrics['generation_time_min']} min")
        print(f"Evaluation time:    {metrics['evaluation_time_sec']} sec")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BIRD Benchmark Evaluation")
    
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
                        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
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
    
    # Load adapter if provided
    if args.adapter_dir and os.path.isdir(args.adapter_dir):
        logger.info(f"Loading LoRA adapter: {args.adapter_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        logger.info("Adapter loaded successfully")
    
    model = model.to("cuda")
    model.eval()
    
    # Create evaluator
    evaluator = BIRDEvaluator(
        model=model,
        tokenizer=tokenizer,
        db_dir=args.db_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.num_workers,
    )
    
    # Run evaluation
    report = evaluator.run_evaluation(bird_data, args.output_dir)
    
    logger.info(f"Evaluation complete! Accuracy: {report['execution_accuracy']}%")


if __name__ == "__main__":
    main()

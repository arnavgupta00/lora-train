#!/usr/bin/env python3
"""
Self-Consistency Evaluation for Text-to-SQL

Implements self-consistency voting:
1. Generate N SQL candidates per question (with temperature sampling)
2. Execute each candidate against the database
3. Group candidates by execution results
4. Vote: select the SQL from the largest result group

This technique typically improves accuracy by 3-5% over greedy decoding.

Usage:
    python evaluation/eval_self_consistency.py \
        --model_id "Qwen/Qwen3-1.7B" \
        --adapter_dir "./outputs/sft/" \
        --bird_dev_json "/path/to/bird/dev.json" \
        --db_dir "/path/to/bird/databases" \
        --output_dir "./results/self_consistency" \
        --n_samples 10
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
from concurrent.futures import ThreadPoolExecutor

# Auto-detect HuggingFace cache location (same as training scripts)
# This prevents re-downloading models that are already cached
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

# Log HF cache location for debugging
if "HF_HOME" in os.environ:
    logger.info(f"Using HuggingFace cache: {os.environ['HF_HOME']}")


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
        # Try to find any SELECT/INSERT/UPDATE/DELETE after thinking
        import re
        match = re.search(r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s', sql, re.IGNORECASE)
        if match:
            sql = sql[match.start():]
        else:
            sql = ""  # No SQL found
    
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


def results_to_key(results: Any) -> str:
    """Convert SQL results to a hashable key for grouping."""
    if results is None:
        return "NULL_RESULT"
    try:
        # Sort rows and convert to tuple for consistent hashing
        sorted_rows = sorted(tuple(row) for row in results)
        return str(sorted_rows)
    except Exception:
        return str(results)


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


class SelfConsistencyEvaluator:
    """Self-consistency evaluation with voting."""
    
    SYSTEM_PROMPT = """You are an expert SQL assistant. Generate SQLite queries from natural language questions.
Given a database schema and a question, generate the correct SQL query.
Only output the SQL query, nothing else."""
    
    def __init__(
        self,
        model,
        tokenizer,
        db_dir: str,
        n_samples: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        num_workers: int = 4,
        thinking_mode: str = "no_think",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_dir = db_dir
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.num_workers = num_workers
        self.thinking_mode = thinking_mode
        
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
        if self.thinking_mode == "no_think":
            user_content = "/no_think\n" + user_content
        elif self.thinking_mode == "think":
            user_content = "/think\n" + user_content
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def generate_candidates(self, prompt: str) -> List[str]:
        """Generate N SQL candidates with temperature sampling."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(self.model.device)
        
        candidates = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_beams=1,
                )
                
                gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                candidates.append(normalize_sql(gen_text))
        
        return candidates
    
    def vote_on_candidates(
        self,
        candidates: List[str],
        db_path: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Vote on candidates based on execution results.
        
        Returns:
            - Selected SQL (from largest result group)
            - Voting metadata
        """
        if not db_path or not Path(db_path).exists():
            # Can't execute - return first candidate
            return candidates[0], {
                "method": "fallback",
                "reason": "no_database",
                "num_candidates": len(candidates),
            }
        
        # Execute each candidate and group by results
        result_groups: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        execution_errors = 0
        
        for sql in candidates:
            ok, result = execute_sql(db_path, sql)
            if ok:
                key = results_to_key(result)
                result_groups[key].append((sql, result))
            else:
                execution_errors += 1
        
        # If all failed, return first candidate
        if not result_groups:
            return candidates[0], {
                "method": "fallback",
                "reason": "all_failed",
                "num_candidates": len(candidates),
                "execution_errors": execution_errors,
            }
        
        # Find the largest group
        best_key = max(result_groups.keys(), key=lambda k: len(result_groups[k]))
        best_group = result_groups[best_key]
        
        # Return the first SQL in the largest group
        selected_sql = best_group[0][0]
        
        # Compute voting metadata
        metadata = {
            "method": "voting",
            "num_candidates": len(candidates),
            "num_groups": len(result_groups),
            "winning_group_size": len(best_group),
            "execution_errors": execution_errors,
            "confidence": len(best_group) / len(candidates),
            "group_sizes": sorted([len(g) for g in result_groups.values()], reverse=True),
        }
        
        return selected_sql, metadata
    
    def evaluate_example(
        self,
        example: Dict,
    ) -> Dict[str, Any]:
        """Evaluate a single example with self-consistency."""
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        evidence = example.get("evidence", "")
        gold_sql = normalize_sql(example.get("SQL", example.get("sql", "")))
        
        # Build prompt
        schema = self._get_schema(db_id)
        prompt = self._build_prompt(question, schema, evidence)
        
        # Generate candidates
        candidates = self.generate_candidates(prompt)
        
        # Find database
        db_path = find_database(self.db_dir, db_id)
        
        # Vote
        selected_sql, vote_meta = self.vote_on_candidates(candidates, db_path)
        
        # Evaluate selected SQL
        result = {
            "db_id": db_id,
            "question": question,
            "selected_sql": selected_sql,
            "gold_sql": gold_sql,
            "candidates": candidates,
            "vote_metadata": vote_meta,
            "exact_match": selected_sql.lower().strip() == gold_sql.lower().strip(),
            "exec_match": False,
            "error": None,
        }
        
        if not db_path:
            result["error"] = "Database not found"
            return result
        
        # Execute gold SQL
        gold_ok, gold_result = execute_sql(db_path, gold_sql)
        if not gold_ok:
            result["error"] = f"Gold SQL error: {gold_result}"
            return result
        
        # Execute selected SQL
        pred_ok, pred_result = execute_sql(db_path, selected_sql)
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
        """Run full self-consistency evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Running self-consistency evaluation on {len(examples)} examples")
        logger.info(f"N samples: {self.n_samples}, Temperature: {self.temperature}")
        
        # Pre-cache schemas
        logger.info("Pre-caching database schemas...")
        db_ids = set(ex.get("db_id", "") for ex in examples)
        for db_id in db_ids:
            if db_id:
                self._get_schema(db_id)
        logger.info(f"Cached {len(self.schema_cache)} schemas")
        
        # Evaluate each example
        results = []
        start_time = time.time()
        
        for i, example in enumerate(examples):
            result = self.evaluate_example(example)
            result["index"] = i
            results.append(result)
            
            # Progress logging
            if (i + 1) % 10 == 0 or (i + 1) == len(examples):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(examples) - i - 1) / rate if rate > 0 else 0
                
                exec_match_so_far = sum(1 for r in results if r["exec_match"])
                acc_so_far = 100 * exec_match_so_far / len(results)
                
                logger.info(
                    f"[{i+1}/{len(examples)}] "
                    f"Acc: {acc_so_far:.1f}% - "
                    f"{rate:.2f} ex/s - "
                    f"ETA: {remaining/60:.1f}min"
                )
        
        total_time = time.time() - start_time
        
        # Compute metrics
        exact_match = sum(1 for r in results if r["exact_match"])
        exec_match = sum(1 for r in results if r["exec_match"])
        
        # Voting analysis
        voting_used = [r for r in results if r["vote_metadata"].get("method") == "voting"]
        avg_confidence = (
            sum(r["vote_metadata"].get("confidence", 0) for r in voting_used) / len(voting_used)
            if voting_used else 0
        )
        avg_groups = (
            sum(r["vote_metadata"].get("num_groups", 0) for r in voting_used) / len(voting_used)
            if voting_used else 0
        )
        
        total = len(results)
        metrics = {
            "total_examples": total,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "exact_match": exact_match,
            "exact_match_pct": round(100 * exact_match / total, 2) if total > 0 else 0,
            "execution_match": exec_match,
            "execution_accuracy": round(100 * exec_match / total, 2) if total > 0 else 0,
            "voting_stats": {
                "voting_used": len(voting_used),
                "fallback_used": len(results) - len(voting_used),
                "avg_confidence": round(avg_confidence, 3),
                "avg_num_groups": round(avg_groups, 2),
            },
            "total_time_min": round(total_time / 60, 2),
            "examples_per_sec": round(total / total_time, 2),
        }
        
        # Per-database breakdown
        db_stats = defaultdict(lambda: {"total": 0, "exec_match": 0})
        for r in results:
            db_id = r["db_id"]
            db_stats[db_id]["total"] += 1
            if r["exec_match"]:
                db_stats[db_id]["exec_match"] += 1
        
        db_breakdown = []
        for db_id, stats in sorted(db_stats.items(), key=lambda x: -x[1]["total"]):
            db_breakdown.append({
                "db_id": db_id,
                "total": stats["total"],
                "exec_match": stats["exec_match"],
                "accuracy": round(100 * stats["exec_match"] / stats["total"], 1) if stats["total"] > 0 else 0,
            })
        
        # Build report
        report = {
            **metrics,
            "by_database": db_breakdown,
        }
        
        # Save results
        predictions_path = os.path.join(output_dir, "predictions_sc.json")
        with open(predictions_path, "w") as f:
            # Save without full candidate lists to reduce file size
            slim_results = []
            for r in results:
                slim = {k: v for k, v in r.items() if k != "candidates"}
                slim["num_unique_candidates"] = len(set(r.get("candidates", [])))
                slim_results.append(slim)
            json.dump(slim_results, f, indent=2)
        
        report_path = os.path.join(output_dir, "evaluation_report_sc.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(metrics, db_breakdown)
        
        logger.info(f"Results saved to: {output_dir}")
        return report
    
    def _print_summary(self, metrics: Dict, db_breakdown: List[Dict]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("SELF-CONSISTENCY EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total examples:     {metrics['total_examples']}")
        print(f"N samples:          {metrics['n_samples']}")
        print(f"Temperature:        {metrics['temperature']}")
        print()
        print(f"Exact match:        {metrics['exact_match']} ({metrics['exact_match_pct']}%)")
        print(f"Execution match:    {metrics['execution_match']} ({metrics['execution_accuracy']}%)")
        print()
        print("VOTING STATISTICS")
        vs = metrics["voting_stats"]
        print(f"  Voting used:      {vs['voting_used']}")
        print(f"  Fallback used:    {vs['fallback_used']}")
        print(f"  Avg confidence:   {vs['avg_confidence']:.3f}")
        print(f"  Avg num groups:   {vs['avg_num_groups']:.2f}")
        print()
        
        print("=" * 60)
        print("TOP 5 DATABASES BY ACCURACY")
        print("=" * 60)
        for db in sorted(db_breakdown, key=lambda x: -x["accuracy"])[:5]:
            print(f"  {db['db_id']}: {db['accuracy']:.1f}% ({db['exec_match']}/{db['total']})")
        
        print()
        print("=" * 60)
        print(f"Total time:         {metrics['total_time_min']} min")
        print(f"Speed:              {metrics['examples_per_sec']} ex/s")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Self-Consistency Evaluation for Text-to-SQL")
    
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
    
    # Self-consistency arguments
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of SQL candidates to generate per question")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for SQL execution")
    parser.add_argument("--thinking_mode", type=str, default="no_think",
                        choices=["auto", "no_think", "think"],
                        help="Thinking control for Qwen3 models")
    
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
    
    # Load model with progress tracking
    logger.info(f"Loading model: {args.model_id}")
    logger.info("Downloading model weights (this may take a few minutes)...")
    
    import os
    from transformers.utils import logging as hf_logging
    
    # Set verbose logging for transformers to capture download info
    hf_logging.set_verbosity_debug()
    
    # Enable HF progress bars - they show in terminal, our logging shows in files
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    
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
    evaluator = SelfConsistencyEvaluator(
        model=model,
        tokenizer=tokenizer,
        db_dir=args.db_dir,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_workers=args.num_workers,
        thinking_mode=args.thinking_mode,
    )
    
    # Run evaluation
    report = evaluator.run_evaluation(bird_data, args.output_dir)
    
    logger.info(f"Self-consistency evaluation complete! Accuracy: {report['execution_accuracy']}%")


if __name__ == "__main__":
    main()

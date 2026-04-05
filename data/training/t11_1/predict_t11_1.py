#!/usr/bin/env python3
"""
T11.1 Prediction Script

Generate SQL predictions using prebuilt T11.1 prompts.

Usage:
    python data/training/t11_1/predict_t11_1.py \
        --model_id "Qwen/Qwen3-1.7B" \
        --adapter_dir "./runs/my_adapter" \
        --prompts_file data/training/t11_1/bird_dev_t11_1.jsonl \
        --output_dir ./results/t11_1
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# Auto-detect HuggingFace cache location
if "HF_HOME" not in os.environ:
    for cache_path in ["/workspace/hf", "/runpod-volume/hf", os.path.expanduser("~/.cache/huggingface")]:
        if os.path.isdir(cache_path):
            os.environ["HF_HOME"] = cache_path
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_path, "datasets")
            break


def normalize_sql(sql: str) -> str:
    """
    Clean up raw model output into a single executable SQL statement.
    
    Handles:
    - Qwen3 thinking tags (<think>...</think>)
    - Markdown code fences (```sql ... ```)
    - Multiple statements (keeps only the first)
    - Whitespace normalization
    """
    if not sql:
        return ""
    
    s = sql.strip()
    
    # Strip Qwen3 thinking tags
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    
    # Strip markdown code fences
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    
    # Keep only the first statement
    s = s.split(";")[0].strip()
    
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    return s


def clean_generated_text(text: str) -> str:
    """Remove chat-role prefixes that sometimes appear at the start of completions."""
    s = text.strip()
    s = re.sub(r"^(assistant|system|user)\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
    return s


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def load_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    """Load prebuilt prompts from JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def get_chat_messages(prompt_record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return chat messages from either T10 prompt records or message-style records."""
    if "t10_prompt" in prompt_record:
        t10_prompt = prompt_record["t10_prompt"]
        return [
            {"role": "system", "content": t10_prompt["system"]},
            {"role": "user", "content": t10_prompt["user"]},
        ]

    if "messages" in prompt_record:
        messages = prompt_record["messages"]
        # Use system+user turns as the prompt and ignore assistant gold output.
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg.get("role") in {"system", "user"}
        ]

    raise KeyError("Prompt record must contain either 't10_prompt' or 'messages'")


def build_generation_config(args) -> Dict[str, Any]:
    """Build generation configuration dict."""
    return {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
    }


def generate_batch(
    model,
    tokenizer,
    prompts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """
    Generate SQL predictions for a batch of prompts.
    
    Returns list of dicts with raw_output and normalized_sql.
    """
    import torch
    
    results = []
    
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch = prompts[batch_start:batch_end]
        
        batch_messages = [get_chat_messages(p) for p in batch]
        
        # Apply chat template
        batch_texts = []
        for messages in batch_messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"] if generation_config["do_sample"] else None,
                top_p=generation_config["top_p"] if generation_config["do_sample"] else None,
                top_k=generation_config["top_k"] if generation_config["do_sample"] else None,
                num_beams=generation_config["num_beams"],
                repetition_penalty=generation_config["repetition_penalty"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        for i, output in enumerate(outputs):
            # With left padding, generated sequences still include the full padded input width.
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output[input_len:]
            raw_output = clean_generated_text(
                tokenizer.decode(gen_ids, skip_special_tokens=True)
            )
            normalized = normalize_sql(raw_output)
            
            results.append({
                "raw_output": raw_output,
                "normalized_sql": normalized,
            })
        
        # Progress
        print(f"  [{batch_end}/{len(prompts)}] Generated {batch_end} predictions")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="T11.1 Prediction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        required=True,
        help="Base model ID (e.g., Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="Path to LoRA adapter directory (optional)",
    )
    
    # Data arguments
    parser.add_argument(
        "--prompts_file",
        default="data/training/t11_1/bird_dev_t11_1.jsonl",
        help="Path to prebuilt prompts file",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for predictions",
    )
    # Generation arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy)")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search beams")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    
    # Misc
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0=all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("T11.1 Prediction Script")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_dir or 'None'}")
    print(f"Prompts: {args.prompts_file}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts
    print("Loading prebuilt prompts...")
    prompts = load_prompts(args.prompts_file)
    print(f"  Loaded {len(prompts)} prompts")
    
    if args.limit > 0:
        prompts = prompts[:args.limit]
        print(f"  Limited to {len(prompts)} examples")
    
    # Build and save generation config
    generation_config = build_generation_config(args)
    config_path = os.path.join(args.output_dir, "generation_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir or None,
            "prompts_file": args.prompts_file,
            "git_commit": get_git_commit_hash(),
            "generation": generation_config,
            "batch_size": args.batch_size,
            "device": args.device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)
    print(f"Generation config saved to: {config_path}")
    print()
    
    # Load model
    print("Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    
    # Load adapter if provided
    if args.adapter_dir and os.path.isdir(args.adapter_dir):
        print(f"Loading LoRA adapter from: {args.adapter_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print("  Adapter loaded successfully")
    
    model = model.to(args.device)
    model.eval()
    print("Model loaded successfully")
    print()
    
    # Generate predictions
    print("Generating predictions...")
    start_time = time.time()
    
    gen_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=generation_config,
        batch_size=args.batch_size,
    )
    
    gen_time = time.time() - start_time
    print(f"\nGeneration complete: {gen_time/60:.1f} min ({len(prompts)/gen_time:.1f} ex/s)")
    
    # Build output records
    predictions = []
    raw_outputs = []
    
    for prompt, result in zip(prompts, gen_results):
        # Prediction record
        pred = {
            "question_id": prompt.get("question_id"),
            "db_id": prompt.get("db_id"),
            "question": prompt.get("question"),
            "predicted_sql": result["normalized_sql"],
            "gold_sql": prompt.get("gold_sql"),
            "difficulty": prompt.get("difficulty", "unknown"),
        }
        
        if "compaction_metadata" in prompt:
            pred["compaction_metadata"] = prompt["compaction_metadata"]
        
        predictions.append(pred)
        
        # Raw output record
        raw = {
            "question_id": prompt.get("question_id"),
            "db_id": prompt.get("db_id"),
            "raw_output": result["raw_output"],
            "normalized_sql": result["normalized_sql"],
        }
        raw_outputs.append(raw)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, "predictions_t11_1.jsonl")
    with open(predictions_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Predictions saved to: {predictions_path}")
    
    # Save raw outputs
    raw_path = os.path.join(args.output_dir, "raw_outputs_t11_1.jsonl")
    with open(raw_path, 'w') as f:
        for raw in raw_outputs:
            f.write(json.dumps(raw) + '\n')
    print(f"Raw outputs saved to: {raw_path}")
    
    # Also save as JSON array for compatibility
    predictions_json_path = os.path.join(args.output_dir, "predictions_t11_1.json")
    with open(predictions_json_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions (JSON) saved to: {predictions_json_path}")
    
    # Print summary
    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(predictions)}")
    print(f"Generation time: {gen_time/60:.1f} min")
    print(f"Rate: {len(prompts)/gen_time:.1f} examples/sec")
    print()
    print("Output files:")
    print(f"  - {predictions_path}")
    print(f"  - {predictions_json_path}")
    print(f"  - {raw_path}")
    print(f"  - {config_path}")


if __name__ == "__main__":
    main()

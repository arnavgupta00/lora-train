#!/usr/bin/env python3
"""
T12 Prediction Script

Generate SQL predictions on BIRD dev using prebuilt T12 prompts.

Features:
- Loads prebuilt prompts from bird_dev_t12.jsonl
- Runs batch inference with model + optional LoRA adapter
- Saves raw model outputs before normalization
- Logs exact generation configuration
- Outputs structured predictions

Usage:
    python predict_t12.py \
        --model_id "Qwen/Qwen3.5-2B" \
        --adapter_dir "./runs/t12_sft_001" \
        --prompts_file data/training/t12/bird_dev_t12.jsonl \
        --output_dir "./runs/t12_sft_001/predictions"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from t12_utils import (
    T12_SYSTEM_PROMPT,
    get_t12_system_prompt_hash,
    normalize_sql,
)

# Auto-detect HuggingFace cache location
if "HF_HOME" not in os.environ:
    for cache_path in ["/workspace/hf", "/runpod-volume/hf", os.path.expanduser("~/.cache/huggingface")]:
        if os.path.isdir(cache_path):
            os.environ["HF_HOME"] = cache_path
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_path, "transformers")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_path, "datasets")
            break


def load_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    """Load prebuilt prompts from JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


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
        
        # Build chat messages
        batch_messages = []
        for p in batch:
            t12_prompt = p["t12_prompt"]
            messages = [
                {"role": "system", "content": t12_prompt["system"]},
                {"role": "user", "content": t12_prompt["user"]},
            ]
            batch_messages.append(messages)
        
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
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = output[input_len:]
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
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
        description="T12 Prediction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model_id",
        required=True,
        help="Base model ID (e.g., Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="Path to LoRA adapter directory (optional)",
    )
    
    # Data arguments
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="Path to prebuilt prompts file (bird_dev_t12.jsonl)",
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
    print("T12 Prediction Script")
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
            "t12_system_prompt_hash": get_t12_system_prompt_hash(),
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
    
    # Load adapter if provided (supports local path or HF adapter repo ID)
    if args.adapter_dir:
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
            "question_id": prompt["question_id"],
            "db_id": prompt["db_id"],
            "question": prompt["question"],
            "predicted_sql": result["normalized_sql"],
            "gold_sql": prompt["gold_sql"],
            "difficulty": prompt.get("difficulty", "unknown"),
        }
        predictions.append(pred)
        
        # Raw output record
        raw = {
            "question_id": prompt["question_id"],
            "db_id": prompt["db_id"],
            "raw_output": result["raw_output"],
            "normalized_sql": result["normalized_sql"],
        }
        raw_outputs.append(raw)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, "predictions_t12.jsonl")
    with open(predictions_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Predictions saved to: {predictions_path}")
    
    # Save raw outputs
    raw_path = os.path.join(args.output_dir, "raw_outputs_t12.jsonl")
    with open(raw_path, 'w') as f:
        for raw in raw_outputs:
            f.write(json.dumps(raw) + '\n')
    print(f"Raw outputs saved to: {raw_path}")
    
    # Also save as JSON array for compatibility with evaluate.py
    predictions_json_path = os.path.join(args.output_dir, "predictions_t12.json")
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

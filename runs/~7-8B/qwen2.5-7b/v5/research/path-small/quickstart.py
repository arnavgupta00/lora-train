#!/usr/bin/env python3
"""
Quick Start: Download and Test SLM-SQL Model

This script downloads the SLM-SQL-1.5B model and runs a test inference.
Run this first to verify everything works before building the full pipeline.

Requirements:
    pip install transformers torch accelerate huggingface_hub

Hardware:
    - Minimum: 8GB VRAM (with float16)
    - Recommended: 16GB VRAM
    - CPU-only: Works but slower (~30 sec/query)
"""

import argparse
import sys


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  Device: Apple MPS (Metal)")
        else:
            print(f"  Device: CPU only (will be slow)")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import accelerate
        print(f"✓ Accelerate {accelerate.__version__}")
    except ImportError:
        missing.append("accelerate")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def download_model(model_name: str = "cycloneboy/SLM-SQL-1.5B"):
    """Download the model from HuggingFace."""
    from huggingface_hub import snapshot_download
    
    print(f"\nDownloading {model_name}...")
    path = snapshot_download(model_name)
    print(f"✓ Downloaded to: {path}")
    return path


def load_model(model_name: str = "cycloneboy/SLM-SQL-1.5B"):
    """Load the model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading {model_name}...")
    
    # Determine device
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    
    print(f"✓ Model loaded on {device_map}")
    return model, tokenizer


def format_prompt(schema: str, question: str) -> str:
    """Format prompt in SLM-SQL style."""
    
    # Based on SLM-SQL's prompting approach
    prompt = f"""<|im_start|>system
You are a helpful assistant that generates SQL queries based on database schemas and natural language questions.<|im_end|>
<|im_start|>user
### Database Schema:
{schema}

### Question:
{question}

Generate a SQL query to answer the question.<|im_end|>
<|im_start|>assistant
"""
    return prompt


def generate_sql(model, tokenizer, schema: str, question: str, 
                 max_new_tokens: int = 512, temperature: float = 0.1) -> str:
    """Generate SQL for a question."""
    import torch
    
    prompt = format_prompt(schema, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL from response
    sql = response.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in sql:
        sql = sql.split("<|im_end|>")[0].strip()
    
    return sql


def run_test():
    """Run a quick test with sample data."""
    
    # Sample schema and question
    test_schema = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL,
    hire_date DATE
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT,
    budget REAL
);
"""
    
    test_questions = [
        "What is the total salary of all employees?",
        "List all employees in the Engineering department",
        "What is the average salary by department?",
    ]
    
    print("\n" + "="*60)
    print("QUICK TEST: SLM-SQL-1.5B")
    print("="*60)
    
    model, tokenizer = load_model()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"Q: {question}")
        
        import time
        start = time.time()
        sql = generate_sql(model, tokenizer, test_schema, question)
        elapsed = time.time() - start
        
        print(f"SQL: {sql}")
        print(f"Time: {elapsed:.2f}s")
    
    print("\n" + "="*60)
    print("✓ Quick test completed!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Quick Start for SLM-SQL")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    parser.add_argument("--download", action="store_true", help="Download model only")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--model", default="cycloneboy/SLM-SQL-1.5B", help="Model name")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SLM-SQL Quick Start")
    print("="*60)
    
    # Always check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check:
        return
    
    if args.download:
        download_model(args.model)
        return
    
    if args.test or (not args.check and not args.download):
        run_test()


if __name__ == "__main__":
    main()

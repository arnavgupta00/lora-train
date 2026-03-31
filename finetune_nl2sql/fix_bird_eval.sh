#!/bin/bash
# Quick fix for BIRD eval - use existing bird_dev_chatml.jsonl

WORKSPACE="/workspace/lora-train"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR="${WORKSPACE}/outputs/qwen2.5-7b-t7-bird-20260331_193013"
BIRD_DIR="${WORKSPACE}/bird_eval"
OUTPUT_DIR="${WORKSPACE}/outputs/bird_evaluation"

echo "Using existing bird_dev_chatml.jsonl from dataset..."

# Generate predictions using our existing data
python3 -u << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import json
import os
import re
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def normalize_sql(sql: str) -> str:
    s = sql.strip()
    if "```" in s:
        m = re.search(r"```(?:sql)?\s*(.*?)```", s, re.S | re.I)
        if m:
            s = m.group(1).strip()
    s = s.split(";")[0].strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_prompt(tokenizer, messages: list) -> str:
    # Remove assistant message if present
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.inference_mode()
def generate_sql(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
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
    
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return normalize_sql(text)

# Configuration
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "/workspace/lora-train/outputs/qwen2.5-7b-t7-bird-20260331_193013"
BIRD_DEV_FILE = "/workspace/lora-train/dataset/bird_dev_chatml.jsonl"
OUTPUT_FILE = "/workspace/lora-train/outputs/bird_evaluation/predictions.json"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

print(f"Loading LoRA adapters from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.to("cuda")
model.eval()

print(f"Loading BIRD dev data from: {BIRD_DEV_FILE}")
bird_data = []
with open(BIRD_DEV_FILE, 'r') as f:
    for line in f:
        bird_data.append(json.loads(line))

print(f"\n🚀 Generating predictions for {len(bird_data)} examples...")

predictions = []
for i, example in enumerate(bird_data):
    messages = example.get("messages", [])
    
    # Extract db_id, question, gold_sql from messages
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    
    if not user_msg or not assistant_msg:
        continue
    
    user_content = str(user_msg.get("content", ""))
    gold_sql = str(assistant_msg.get("content", ""))
    
    # Extract db_id from user content (usually in the schema part)
    db_id_match = re.search(r"Database:\s*(\w+)", user_content)
    db_id = db_id_match.group(1) if db_id_match else "unknown"
    
    # Extract question
    question_match = re.search(r"Question:\s*(.+?)(?:\n|$)", user_content)
    question = question_match.group(1).strip() if question_match else ""
    
    prompt = build_prompt(tokenizer, messages)
    pred_sql = generate_sql(tokenizer, model, prompt)
    
    predictions.append({
        "db_id": db_id,
        "question": question,
        "predicted_sql": pred_sql,
        "gold_sql": normalize_sql(gold_sql),
    })
    
    if (i + 1) % 50 == 0 or i == len(bird_data) - 1:
        print(f"  Progress: {i+1}/{len(bird_data)} ({100*(i+1)/len(bird_data):.1f}%)")

# Save predictions
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(predictions, f, indent=2)
print(f"\n✓ Saved predictions to {OUTPUT_FILE}")
print(f"✓ Generated {len(predictions)} predictions")
PYTHON_SCRIPT

echo ""
echo "✓ Predictions generated!"
echo "Next: Run evaluation against BIRD databases"

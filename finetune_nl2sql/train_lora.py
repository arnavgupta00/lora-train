import argparse
import json
import math
import os
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import is_bitsandbytes_available


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _messages_without_assistant(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    if not messages:
        raise ValueError("messages is empty")
    if messages[-1].get("role") != "assistant":
        raise ValueError("expected last message to be assistant")
    return messages[:-1], messages[-1]


def _build_labels(
    tokenizer,
    messages: List[Dict[str, str]],
    max_seq_len: int,
) -> Dict[str, Any]:
    prompt_msgs, assistant_msg = _messages_without_assistant(messages)

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + [assistant_msg],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    start = len(prompt_enc["input_ids"])
    if start >= len(input_ids):
        # Edge case: truncation clipped away the assistant content.
        # Make this row a no-op (all masked) so it won't contribute to loss.
        labels = [-100] * len(input_ids)
    else:
        labels = [-100] * start + input_ids[start:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _pack_features(
    features: List[Dict[str, Any]],
    max_seq_len: int,
    eos_token_id: Optional[int],
) -> List[Dict[str, Any]]:
    if eos_token_id is None:
        return features

    packed: List[Dict[str, Any]] = []
    cur_input: List[int] = []
    cur_attn: List[int] = []
    cur_labels: List[int] = []

    def flush():
        if not cur_input:
            return
        packed.append(
            {
                "input_ids": cur_input.copy(),
                "attention_mask": cur_attn.copy(),
                "labels": cur_labels.copy(),
            }
        )
        cur_input.clear()
        cur_attn.clear()
        cur_labels.clear()

    for ex in features:
        ex_input = ex["input_ids"]
        ex_attn = ex["attention_mask"]
        ex_labels = ex["labels"]

        # If the example doesn't fit in the current block, flush first.
        if cur_input and (len(cur_input) + 1 + len(ex_input) > max_seq_len):
            flush()

        # If the single example is too long, keep as-is (already truncated).
        if not cur_input:
            cur_input.extend(ex_input[:max_seq_len])
            cur_attn.extend(ex_attn[:max_seq_len])
            cur_labels.extend(ex_labels[:max_seq_len])
            continue

        # Add a masked EOS separator between examples to avoid accidental cross-row learning.
        if len(cur_input) < max_seq_len:
            cur_input.append(eos_token_id)
            cur_attn.append(1)
            cur_labels.append(-100)

        remaining = max_seq_len - len(cur_input)
        if remaining <= 0:
            flush()
            cur_input.extend(ex_input[:max_seq_len])
            cur_attn.extend(ex_attn[:max_seq_len])
            cur_labels.extend(ex_labels[:max_seq_len])
        else:
            cur_input.extend(ex_input[:remaining])
            cur_attn.extend(ex_attn[:remaining])
            cur_labels.extend(ex_labels[:remaining])

    flush()
    return packed


class JsonlChatDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path: str, max_seq_len: int, pack: bool):
        rows = _read_jsonl(jsonl_path)
        feats: List[Dict[str, Any]] = []
        for row in rows:
            feats.append(_build_labels(tokenizer, row["messages"], max_seq_len=max_seq_len))
        if pack:
            feats = _pack_features(feats, max_seq_len=max_seq_len, eos_token_id=tokenizer.eos_token_id)
        self._features = feats

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._features[idx]


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad_list(values: List[int], pad_value: int) -> List[int]:
            return values + [pad_value] * (max_len - len(values))

        input_ids = torch.tensor([pad_list(f["input_ids"], self.pad_token_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor([pad_list(f["attention_mask"], 0) for f in features], dtype=torch.long)
        labels = torch.tensor([pad_list(f["labels"], -100) for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--dev_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 matmul (speed on Ampere+)")

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names",
    )
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the base model in 8-bit (bitsandbytes) to fit smaller VRAM GPUs. This is LoRA (not QLoRA).",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_8bit:
        if not is_bitsandbytes_available():
            raise SystemExit(
                "bitsandbytes is not available but --load_in_8bit was set. "
                "Install it (pip install bitsandbytes) or run on a larger VRAM GPU."
            )
        # 8-bit base load is still "LoRA" (not QLoRA); it just reduces memory for the frozen base weights.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=True,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.load_in_8bit:
        # Prepares LayerNorm etc for stable k-bit training.
        try:
            from peft import prepare_model_for_kbit_training
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"prepare_model_for_kbit_training import failed: {e}")
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, lora_config)

    train_ds = JsonlChatDataset(tokenizer, args.train_jsonl, max_seq_len=args.max_seq_len, pack=args.pack)
    dev_ds = JsonlChatDataset(tokenizer, args.dev_jsonl, max_seq_len=args.max_seq_len, pack=False)

    total_train_steps = math.ceil(len(train_ds) / (args.per_device_train_batch_size * 1.0))
    total_train_steps = math.ceil(total_train_steps / max(args.gradient_accumulation_steps, 1))
    total_train_steps = int(total_train_steps * args.num_train_epochs)

    # HF has renamed some TrainingArguments fields across versions (notably evaluation_strategy -> eval_strategy).
    ta_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        # Use bf16 when supported, otherwise fp16.
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "run_name": os.path.basename(args.output_dir),
    }

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        # Fall back to no periodic eval if the arg name is unknown.
        pass

    training_args = TrainingArguments(**ta_kwargs)

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save PEFT adapter + tokenizer.
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Save minimal run metadata (no secrets).
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_id": args.model_id,
                "train_jsonl": args.train_jsonl,
                "dev_jsonl": args.dev_jsonl,
                "max_seq_len": args.max_seq_len,
                "pack": bool(args.pack),
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": [m.strip() for m in args.target_modules.split(",") if m.strip()],
                },
                "training": {
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "num_train_epochs": args.num_train_epochs,
                    "warmup_ratio": args.warmup_ratio,
                    "eval_steps": args.eval_steps,
                    "save_steps": args.save_steps,
                    "estimated_train_steps": total_train_steps,
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

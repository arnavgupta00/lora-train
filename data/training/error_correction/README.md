# Error-Correction Training Pipeline

Training entrypoints for the benchmark-clean SQL repair dataset.

## Quick Start

```bash
# Train with the provided Qwen3.5-2B config
./data/training/error_correction/train_error_correction.sh \
    --config training/configs/error_correction_qwen3_5_2b_3090.yaml \
    2>&1 | tee runs/error_correction_qwen3_5_2b_3090/train.log

# Or train with explicit CLI arguments
./data/training/error_correction/train_error_correction.sh \
    --model_id "Qwen/Qwen3.5-2B" \
    --output_dir "./runs/error_correction_sft_001"
```

## Dataset Files

- `train_error_repair_v1_clean.jsonl`
- `dev_error_repair_v1_clean.jsonl`

These are the benchmark-clean repair examples. The launcher validates prompt
structure before calling `/Users/arnav/programming/lm/training/train_lora.py`.

## Prompt Contract

System prompt:

```text
You are an expert SQL repair assistant. Given schema, question, hints, broken SQL, and optional database error, output the corrected SQL query only.
```

User prompt structure:

```text
Schema:
...

Hints:
...

Question:
...

Broken SQL:
...

Error:
...        # optional

Failure Type:
...        # optional
```

Assistant output:

- corrected SQL only
- no markdown
- no code fences
- no reasoning or prose

## Included Launcher + Config

- `/Users/arnav/programming/lm/data/training/error_correction/train_error_correction.sh`
- `/Users/arnav/programming/lm/training/configs/error_correction_qwen3_5_2b_3090.yaml`

The default config is tuned for a faster single-GPU run on a 24 GB class card:

- `Qwen/Qwen3.5-2B`
- `max_seq_len: 2048`
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 16`
- `pack: true`
- `bf16 + gradient_checkpointing`
- `logging_steps: 50`
- standard Trainer progress bar + saved `run_meta.json`

This profile is intended to keep training in the `3-4 hour` range on the current
repair dataset. If you hit memory pressure, drop `per_device_train_batch_size`
back to `1` and restore `gradient_accumulation_steps: 32`.

# Training with YAML Config - Quick Guide

## Prerequisites

```bash
# Install PyYAML if not already installed
pip install PyYAML>=6.0.0
```

## Using the Pre-configured t10_baseline_3090 Config

The `training/configs/t10_baseline_3090.yaml` config is optimized for RTX 3090 (24GB VRAM).

### Configuration Highlights

- **Model**: Qwen/Qwen3-1.7B
- **Effective Batch Size**: 32 (2 per device × 16 gradient accumulation)
- **Sequence Length**: 4096 tokens
- **LoRA**: r=32, alpha=64 (higher rank for better capacity)
- **Learning Rate**: 1.5e-4 with cosine schedule
- **Precision**: BF16 with gradient checkpointing
- **Estimated Training Time**: ~4-6 hours for 2 epochs on RTX 3090

### Run Training

```bash
cd /Users/arnav/programming/lm

# Run with config file
./data/training/t10/train_t10.sh \
    --config training/configs/t10_baseline_3090.yaml
```

### Monitor Training

The script will output:
1. **Configuration summary** - Shows all hyperparameters before training starts
2. **Training logs** - Every 10 steps, showing:
   - `loss` - Current training loss
   - `grad_norm` - Gradient norm (watch for explosions >10.0)
   - `learning_rate` - Current LR (decreases with cosine schedule)
   - `epoch` - Current epoch progress

Example output:
```
==================================================
TRAINING CONFIGURATION
==================================================
Run name:              t10_baseline_3090
Model:                 Qwen/Qwen3-1.7B
Method:                lora_sft
Output dir:            runs/t10_baseline_3090

Training examples:     14034
Dev examples:          665
Max sequence length:   4096

Batch size:            2
Gradient accum:        16
Effective batch size:  32

Learning rate:         0.00015
LR scheduler:          cosine
Warmup ratio:          0.05
Weight decay:          0.01
Max grad norm:         1.0

Epochs:                2
Estimated steps:       878

LoRA r:                32
LoRA alpha:            64
LoRA dropout:          0.05

Precision:             bf16
Gradient checkpointing: True
==================================================

{'loss': 1.234, 'grad_norm': 2.156, 'learning_rate': 1.425e-4, 'epoch': 0.02, 'step': 10}
{'loss': 0.987, 'grad_norm': 1.543, 'learning_rate': 1.448e-4, 'epoch': 0.05, 'step': 20}
...
```

### Override Config Values via CLI

You can override specific config values:

```bash
./data/training/t10/train_t10.sh \
    --config training/configs/t10_baseline_3090.yaml \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --output_dir runs/t10_baseline_custom
```

CLI arguments take precedence over config file values.

### Output Files

After training, you'll find in `runs/t10_baseline_3090/`:
- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - Trained LoRA weights
- `tokenizer*` - Tokenizer files
- `run_meta.json` - Full training metadata
- `trainer_state.json` - Training state (loss curves, etc.)
- `checkpoint-*/` - Intermediate checkpoints (every 200 steps)

### Next Steps

After training completes:

1. **Generate predictions**:
```bash
python data/training/t10/predict_t10.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --adapter_dir "./runs/t10_baseline_3090" \
    --prompts_file data/training/t10/bird_dev_t10.jsonl \
    --output_dir "./runs/t10_baseline_3090/predictions"
```

2. **Evaluate**:
```bash
python data/training/t10/evaluate_t10.py \
    --predictions_file ./runs/t10_baseline_3090/predictions/predictions_t10.jsonl \
    --output_dir ./runs/t10_baseline_3090/eval
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

```yaml
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 32  # Keep effective batch size at 32

# Or reduce sequence length
max_seq_length: 2048

# Or enable 4-bit loading (QLoRA)
load_in_4bit: true
bf16: false  # 4-bit uses fp16 internally
```

### Gradient Explosion

If you see `grad_norm` > 10.0 consistently:

```yaml
# Reduce learning rate
learning_rate: 1e-4  # Down from 1.5e-4

# Or reduce max_grad_norm
max_grad_norm: 0.5  # Down from 1.0
```

### Slow Training

If training is too slow:

```yaml
# Enable packing (fits more samples per batch)
pack: true

# Reduce sequence length if acceptable
max_seq_length: 2048
```

## Creating Your Own Config

Copy the template:
```bash
cp training/configs/t10_baseline_3090.yaml training/configs/my_config.yaml
```

Then edit `my_config.yaml` with your settings.

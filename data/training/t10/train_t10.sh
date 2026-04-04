#!/bin/bash
# =============================================================================
# T10 Training Launcher
# =============================================================================
# Convenience wrapper for training on T10 dataset using LoRA fine-tuning.
# Uses training/train_lora.py with T10-specific defaults.
#
# Usage:
#   # With YAML config
#   ./train_t10.sh --config training/configs/t10_baseline_3090.yaml
#
#   # With CLI arguments
#   ./train_t10.sh --model_id "Qwen/Qwen3-1.7B" --output_dir "./runs/t10_sft_001"
#
# Common options:
#   --config            Path to YAML config file
#   --model_id          Base model ID (required if no config)
#   --output_dir        Output directory for checkpoints (required if no config)
#   --num_train_epochs  Number of epochs (default: 2)
#   --learning_rate     Learning rate (default: 1e-4)
#   --lora_r            LoRA rank (default: 8)
#   --max_seq_len       Max sequence length (default: 1024)
#   --per_device_train_batch_size  Batch size (default: 1)
#   --gradient_accumulation_steps  Gradient accumulation (default: 8)
# =============================================================================

set -e

# Get script directory (where T10 data lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Check if using config file
USING_CONFIG=false
for arg in "$@"; do
    if [[ "$arg" == --config=* ]] || [[ "$arg" == "--config" ]]; then
        USING_CONFIG=true
        break
    fi
done

# T10 data paths (can be overridden by config or CLI)
TRAIN_FILE="$SCRIPT_DIR/train_t10.jsonl"
DEV_FILE="$SCRIPT_DIR/dev_t10.jsonl"

# Validate T10 data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$DEV_FILE" ]; then
    echo "ERROR: Dev file not found: $DEV_FILE"
    exit 1
fi

# Count examples
TRAIN_COUNT=$(wc -l < "$TRAIN_FILE" | tr -d ' ')
DEV_COUNT=$(wc -l < "$DEV_FILE" | tr -d ' ')

echo "=========================================="
echo "T10 Training Launcher"
echo "=========================================="
if [ "$USING_CONFIG" = true ]; then
    echo "Mode: Using YAML config file"
else
    echo "Mode: Using CLI arguments"
fi
echo "Training file: $TRAIN_FILE ($TRAIN_COUNT examples)"
echo "Dev file: $DEV_FILE ($DEV_COUNT examples)"
echo ""

# Validate T10 prompts before training
echo "Validating T10 prompt format..."
python3 -c "
import json
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from t10_utils import validate_t10_messages, T10_SYSTEM_PROMPT

errors = []
with open('$TRAIN_FILE', 'r') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            messages = ex.get('messages', [])
            is_valid, errs = validate_t10_messages(messages, strict=False)
            if not is_valid:
                errors.append(f'Line {i+1}: {errs}')
                if len(errors) >= 5:
                    break
        except Exception as e:
            errors.append(f'Line {i+1}: JSON parse error: {e}')
            if len(errors) >= 5:
                break

if errors:
    print('T10 Validation FAILED:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('T10 Validation PASSED - all prompts follow T10 contract')
"

if [ $? -ne 0 ]; then
    echo "ERROR: T10 validation failed. Fix training data before proceeding."
    exit 1
fi

echo ""
echo "Starting training..."
echo "=========================================="

# Build command with T10 data paths if not in config
if [ "$USING_CONFIG" = false ]; then
    # Add T10 data paths to arguments
    python3 "$REPO_ROOT/training/train_lora.py" \
        --train_jsonl "$TRAIN_FILE" \
        --dev_jsonl "$DEV_FILE" \
        "$@"
else
    # Config file will provide paths
    python3 "$REPO_ROOT/training/train_lora.py" "$@"
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

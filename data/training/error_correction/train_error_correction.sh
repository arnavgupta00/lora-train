#!/bin/bash
# =============================================================================
# Error-Correction Training Launcher
# =============================================================================
# Convenience wrapper for training on the benchmark-clean SQL repair dataset.
# Uses training/train_lora.py with error-correction-specific defaults.
#
# Usage:
#   # With YAML config
#   ./train_error_correction.sh --config training/configs/error_correction_qwen3_5_2b_3090.yaml
#
#   # With CLI arguments
#   ./train_error_correction.sh --model_id "Qwen/Qwen3.5-2B" --output_dir "./runs/error_correction_sft_001"
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

USING_CONFIG=false
for arg in "$@"; do
    if [[ "$arg" == --config=* ]] || [[ "$arg" == "--config" ]]; then
        USING_CONFIG=true
        break
    fi
done

TRAIN_FILE="$SCRIPT_DIR/train_error_repair_v1_clean.jsonl"
DEV_FILE="$SCRIPT_DIR/dev_error_repair_v1_clean.jsonl"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$DEV_FILE" ]; then
    echo "ERROR: Dev file not found: $DEV_FILE"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_FILE" | tr -d ' ')
DEV_COUNT=$(wc -l < "$DEV_FILE" | tr -d ' ')

echo "=========================================="
echo "Error-Correction Training Launcher"
echo "=========================================="
if [ "$USING_CONFIG" = true ]; then
    echo "Mode: Using YAML config file"
    if ! python3 -c "import yaml" >/dev/null 2>&1; then
        echo "ERROR: PyYAML is required for --config mode. Install it with: pip install pyyaml"
        exit 1
    fi
else
    echo "Mode: Using CLI arguments"
fi
echo "Training file: $TRAIN_FILE ($TRAIN_COUNT examples)"
echo "Dev file: $DEV_FILE ($DEV_COUNT examples)"
echo ""

echo "Validating error-correction prompt format..."
python3 -c "
import json
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from error_correction_utils import validate_error_correction_messages

errors = []
with open('$TRAIN_FILE', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            messages = ex.get('messages', [])
            is_valid, errs = validate_error_correction_messages(messages, strict=False)
            if not is_valid:
                errors.append(f'Line {i+1}: {errs}')
                if len(errors) >= 5:
                    break
        except Exception as e:
            errors.append(f'Line {i+1}: JSON parse error: {e}')
            if len(errors) >= 5:
                break

if errors:
    print('Error-correction validation FAILED:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('Error-correction validation PASSED - all prompts follow the repair contract')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Error-correction validation failed. Fix training data before proceeding."
    exit 1
fi

echo ""
echo "Starting training..."
echo "=========================================="

if [ "$USING_CONFIG" = false ]; then
    python3 "$REPO_ROOT/training/train_lora.py" \
        --train_jsonl "$TRAIN_FILE" \
        --dev_jsonl "$DEV_FILE" \
        "$@"
else
    python3 "$REPO_ROOT/training/train_lora.py" "$@"
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

#!/usr/bin/env bash
###############################################################################
# Script 3a: Basic BIRD Evaluation (Greedy Decoding)
#
# Evaluates baseline and trained models with greedy decoding (temperature=0).
# Can run multiple models in parallel on the same GPU.
#
# Usage:
#   # Evaluate all available models (baseline, SFT, GRPO)
#   nohup bash evaluation/run_eval_basic.sh > eval.log 2>&1 &
#   tail -f eval.log
#
#   # Evaluate specific model
#   MODEL_PATH=./outputs/qwen3-1.7b-sft-t9 bash evaluation/run_eval_basic.sh
#
#   # Baseline only
#   EVAL_BASELINE_ONLY=1 bash evaluation/run_eval_basic.sh
#
# Environment Variables:
#   MODEL_ID           - Base model (default: Qwen/Qwen3-1.7B)
#   MODEL_PATH         - Path to LoRA adapter (evaluates one model)
#   EVAL_BASELINE_ONLY - Only evaluate baseline (default: 0)
#   BATCH_SIZE         - Generation batch size (default: 8)
#   MAX_NEW_TOKENS     - Max tokens to generate (default: 256)
#   BIRD_DEV_JSON      - Path to BIRD dev.json
#   DB_DIR             - Path to BIRD databases
#
# Output:
#   Results saved to ./results/basic_eval_<timestamp>/
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  BIRD Evaluation - Basic (Greedy)"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_BASELINE_ONLY="${EVAL_BASELINE_ONLY:-0}"

# =============================================================================
# Directory Setup
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
echo "Working directory: $ROOT_DIR"

# HuggingFace cache setup
if [[ -d /runpod-volume ]]; then
    export HF_HOME="/runpod-volume/hf"
    export TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
    export HF_DATASETS_CACHE="/runpod-volume/hf/datasets"
    OUT_BASE="/runpod-volume/outputs"
elif [[ -d /workspace ]]; then
    export HF_HOME="/workspace/hf"
    export TRANSFORMERS_CACHE="/workspace/hf/transformers"
    export HF_DATASETS_CACHE="/workspace/hf/datasets"
    OUT_BASE="/workspace/outputs"
else
    export HF_HOME="$HOME/.cache/huggingface"
    OUT_BASE="$ROOT_DIR/outputs"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

# Results directory
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/basic_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"
echo "Results directory: $RESULTS_DIR"

# =============================================================================
# Find BIRD Data
# =============================================================================

BIRD_DEV_JSON="${BIRD_DEV_JSON:-}"
DB_DIR="${DB_DIR:-}"

# Find BIRD dev.json
if [[ -z "$BIRD_DEV_JSON" ]]; then
    for candidate in \
        "/workspace/bird_eval/dev.json" \
        "/workspace/lora-train/bird_eval/dev_20240627/dev.json" \
        "/runpod-volume/bird_eval/dev.json" \
        "$ROOT_DIR/bird_eval/dev.json" \
        "$ROOT_DIR/data/bird/dev.json"; do
        if [[ -f "$candidate" ]]; then
            BIRD_DEV_JSON="$candidate"
            break
        fi
    done
fi

# Find database directory
if [[ -z "$DB_DIR" ]]; then
    for candidate in \
        "/workspace/bird_eval/dev_databases" \
        "/workspace/lora-train/bird_eval/dev_20240627/dev_databases" \
        "/runpod-volume/bird_eval/dev_databases" \
        "$ROOT_DIR/bird_eval/dev_databases" \
        "$ROOT_DIR/data/bird/databases"; do
        if [[ -d "$candidate" ]]; then
            DB_DIR="$candidate"
            break
        fi
    done
fi

# Validate paths
if [[ -z "$BIRD_DEV_JSON" || ! -f "$BIRD_DEV_JSON" ]]; then
    echo "ERROR: BIRD dev.json not found!"
    echo "Set BIRD_DEV_JSON environment variable"
    exit 1
fi

if [[ -z "$DB_DIR" || ! -d "$DB_DIR" ]]; then
    echo "ERROR: BIRD databases not found!"
    echo "Set DB_DIR environment variable"
    exit 1
fi

echo "BIRD dev.json: $BIRD_DEV_JSON"
echo "Databases:     $DB_DIR"
echo ""

# =============================================================================
# GPU Detection
# =============================================================================

echo ">>> GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi
echo ""

# =============================================================================
# Find Models to Evaluate
# =============================================================================

declare -a MODELS_TO_EVAL=()

# Specific model specified
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODELS_TO_EVAL+=("$MODEL_PATH")
    echo "Evaluating specific model: $MODEL_PATH"
else
    # Add baseline
    MODELS_TO_EVAL+=("baseline")
    
    if [[ "$EVAL_BASELINE_ONLY" != "1" ]]; then
        # Find SFT models
        for adapter in "$OUT_BASE"/qwen3-1.7b-sft-*; do
            if [[ -d "$adapter" && -f "$adapter/adapter_config.json" ]]; then
                MODELS_TO_EVAL+=("$adapter")
            fi
        done
        
        # Also check from exported env vars
        if [[ -n "${SFT_OUTPUT_DIR:-}" && -d "$SFT_OUTPUT_DIR" ]]; then
            # Avoid duplicates
            already_added=0
            for m in "${MODELS_TO_EVAL[@]}"; do
                if [[ "$m" == "$SFT_OUTPUT_DIR" ]]; then
                    already_added=1
                    break
                fi
            done
            if [[ $already_added -eq 0 ]]; then
                MODELS_TO_EVAL+=("$SFT_OUTPUT_DIR")
            fi
        fi
        
        # Find GRPO models
        for adapter in "$OUT_BASE"/qwen3-1.7b-grpo-*; do
            if [[ -d "$adapter" && -f "$adapter/adapter_config.json" ]]; then
                MODELS_TO_EVAL+=("$adapter")
            fi
        done
        
        if [[ -n "${GRPO_OUTPUT_DIR:-}" && -d "$GRPO_OUTPUT_DIR" ]]; then
            already_added=0
            for m in "${MODELS_TO_EVAL[@]}"; do
                if [[ "$m" == "$GRPO_OUTPUT_DIR" ]]; then
                    already_added=1
                    break
                fi
            done
            if [[ $already_added -eq 0 ]]; then
                MODELS_TO_EVAL+=("$GRPO_OUTPUT_DIR")
            fi
        fi
    fi
fi

echo ""
echo ">>> Models to evaluate:"
for model in "${MODELS_TO_EVAL[@]}"; do
    echo "  - $model"
done
echo ""

# =============================================================================
# Evaluation Function
# =============================================================================

evaluate_model() {
    local adapter_path="$1"
    local model_name
    
    if [[ "$adapter_path" == "baseline" ]]; then
        model_name="baseline"
        adapter_arg=""
    else
        model_name=$(basename "$adapter_path")
        adapter_arg="--adapter_path $adapter_path"
    fi
    
    local output_file="$RESULTS_DIR/${model_name}_results.json"
    local log_file="$RESULTS_DIR/${model_name}_eval.log"
    
    echo ""
    echo "=============================================="
    echo ">>> Evaluating: $model_name"
    echo "=============================================="
    echo "Adapter: ${adapter_path}"
    echo "Output:  $output_file"
    echo "Started: $(date)"
    echo ""
    
    python3 -u evaluation/eval_bird.py \
        --model_id "$MODEL_ID" \
        $adapter_arg \
        --bird_dev_json "$BIRD_DEV_JSON" \
        --db_dir "$DB_DIR" \
        --output_file "$output_file" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature 0.0 \
        --use_greedy 2>&1 | tee "$log_file"
    
    echo ""
    echo ">>> $model_name evaluation complete: $(date)"
    
    # Extract accuracy from results
    if [[ -f "$output_file" ]]; then
        accuracy=$(python3 -c "import json; d=json.load(open('$output_file')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
        echo ">>> $model_name Accuracy: $accuracy"
    fi
}

# =============================================================================
# Run Evaluations
# =============================================================================

echo ""
echo "=============================================="
echo ">>> Starting Evaluations"
echo "=============================================="

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "BIRD Evaluation Summary" > "$SUMMARY_FILE"
echo "========================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Base Model: $MODEL_ID" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for model in "${MODELS_TO_EVAL[@]}"; do
    evaluate_model "$model"
    
    # Add to summary
    if [[ "$model" == "baseline" ]]; then
        name="Baseline"
    else
        name=$(basename "$model")
    fi
    
    result_file="$RESULTS_DIR/${name}_results.json"
    if [[ "$model" == "baseline" ]]; then
        result_file="$RESULTS_DIR/baseline_results.json"
    fi
    
    if [[ -f "$result_file" ]]; then
        accuracy=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
        echo "$name: $accuracy" >> "$SUMMARY_FILE"
    fi
done

# =============================================================================
# Final Summary
# =============================================================================

echo ""
echo "=============================================="
echo ">>> Evaluation Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary:"
cat "$SUMMARY_FILE"
echo ""
echo "Files:"
ls -la "$RESULTS_DIR"

#!/usr/bin/env bash
###############################################################################
# Script 3b: Self-Consistency Evaluation for BIRD
#
# Uses Self-Consistency (SC) voting: generates N candidates per question,
# executes each against the database, and selects the SQL from the most
# frequent result group.
#
# Algorithm:
#   1. Generate N=10 SQL candidates with temperature=0.7
#   2. Execute each SQL, group by result hash
#   3. Vote: select SQL from the largest result group
#   4. Expected gain: +3-5% over greedy decoding
#
# Usage:
#   # Self-consistency evaluation (N=10 samples)
#   nohup bash evaluation/run_eval_self_consistency.sh > eval_sc.log 2>&1 &
#   tail -f eval_sc.log
#
#   # Faster with fewer samples
#   N_SAMPLES=5 bash evaluation/run_eval_self_consistency.sh
#
# Environment Variables:
#   MODEL_ID       - Base model (default: Qwen/Qwen3-1.7B)
#   MODEL_PATH     - Path to LoRA adapter (required)
#   N_SAMPLES      - Number of candidates (default: 10)
#   TEMPERATURE    - Sampling temperature (default: 0.7)
#   TOP_P          - Nucleus sampling (default: 0.95)
#   SQL_WORKERS    - Parallel SQL execution workers (default: 4)
#
# Output:
#   Results saved to ./results/sc_eval_<timestamp>/
###############################################################################

set -euo pipefail

echo "=============================================="
echo "  BIRD Evaluation - Self-Consistency (SC)"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
N_SAMPLES="${N_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SQL_WORKERS="${SQL_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Lower for SC due to memory from multiple generations

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
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/sc_eval_$(date +%Y%m%d_%H%M%S)}"
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
    # Find SFT models (most recent)
    latest_sft=$(ls -dt "$OUT_BASE"/qwen3-1.7b-sft-* 2>/dev/null | head -n1 || echo "")
    if [[ -n "$latest_sft" && -d "$latest_sft" ]]; then
        MODELS_TO_EVAL+=("$latest_sft")
    fi
    
    # Also check from exported env vars
    if [[ -n "${SFT_OUTPUT_DIR:-}" && -d "$SFT_OUTPUT_DIR" ]]; then
        already_added=0
        for m in "${MODELS_TO_EVAL[@]:-}"; do
            if [[ "$m" == "$SFT_OUTPUT_DIR" ]]; then
                already_added=1
                break
            fi
        done
        if [[ $already_added -eq 0 ]]; then
            MODELS_TO_EVAL+=("$SFT_OUTPUT_DIR")
        fi
    fi
    
    # Find GRPO models (most recent)
    latest_grpo=$(ls -dt "$OUT_BASE"/qwen3-1.7b-grpo-* 2>/dev/null | head -n1 || echo "")
    if [[ -n "$latest_grpo" && -d "$latest_grpo" ]]; then
        MODELS_TO_EVAL+=("$latest_grpo")
    fi
    
    if [[ -n "${GRPO_OUTPUT_DIR:-}" && -d "$GRPO_OUTPUT_DIR" ]]; then
        already_added=0
        for m in "${MODELS_TO_EVAL[@]:-}"; do
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

# Check if any models found
if [[ ${#MODELS_TO_EVAL[@]} -eq 0 ]]; then
    echo "ERROR: No models found for self-consistency evaluation!"
    echo ""
    echo "Please either:"
    echo "  1. Train a model first"
    echo "  2. Set MODEL_PATH to a specific adapter"
    echo ""
    echo "Example:"
    echo "  MODEL_PATH=./outputs/qwen3-1.7b-sft-t9 bash evaluation/run_eval_self_consistency.sh"
    exit 1
fi

echo ""
echo ">>> Models to evaluate with Self-Consistency (N=$N_SAMPLES):"
for model in "${MODELS_TO_EVAL[@]}"; do
    echo "  - $model"
done
echo ""

# =============================================================================
# Evaluation Function
# =============================================================================

evaluate_model_sc() {
    local adapter_path="$1"
    local model_name=$(basename "$adapter_path")
    local output_dir="$RESULTS_DIR/${model_name}_sc"
    local log_file="$RESULTS_DIR/${model_name}_sc_eval.log"
    
    echo ""
    echo "=============================================="
    echo ">>> Self-Consistency Evaluation: $model_name"
    echo "=============================================="
    echo "Adapter:     $adapter_path"
    echo "N Samples:   $N_SAMPLES"
    echo "Temperature: $TEMPERATURE"
    echo "Top-P:       $TOP_P"
    echo "Output:      $output_dir"
    echo "Started:     $(date)"
    echo ""
    
    python3 -u evaluation/eval_self_consistency.py \
        --model_id "$MODEL_ID" \
        --adapter_dir "$adapter_path" \
        --bird_dev_json "$BIRD_DEV_JSON" \
        --db_dir "$DB_DIR" \
        --output_dir "$output_dir" \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_workers "$SQL_WORKERS" 2>&1 | tee "$log_file"
    
    echo ""
    echo ">>> $model_name SC evaluation complete: $(date)"
    
    # Extract accuracy from results
    if [[ -f "$output_dir/results.json" ]]; then
        accuracy=$(python3 -c "import json; d=json.load(open('$output_dir/results.json')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
        echo ">>> $model_name SC Accuracy: $accuracy"
    fi
}

# =============================================================================
# Run Evaluations
# =============================================================================

echo ""
echo "=============================================="
echo ">>> Starting Self-Consistency Evaluations"
echo "=============================================="
echo "N Samples:   $N_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Top-P:       $TOP_P"
echo ""

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "BIRD Self-Consistency Evaluation Summary" > "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Base Model: $MODEL_ID" >> "$SUMMARY_FILE"
echo "N Samples: $N_SAMPLES" >> "$SUMMARY_FILE"
echo "Temperature: $TEMPERATURE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for model in "${MODELS_TO_EVAL[@]}"; do
    evaluate_model_sc "$model"
    
    # Add to summary
    name=$(basename "$model")
    result_file="$RESULTS_DIR/${name}_sc_results.json"
    
    if [[ -f "$result_file" ]]; then
        accuracy=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d.get(\"accuracy\", 0)*100:.2f}%')" 2>/dev/null || echo "N/A")
        echo "$name (SC N=$N_SAMPLES): $accuracy" >> "$SUMMARY_FILE"
    fi
done

# =============================================================================
# Final Summary
# =============================================================================

echo ""
echo "=============================================="
echo ">>> Self-Consistency Evaluation Complete!"
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
echo ""
echo "Compare SC vs Greedy:"
echo "  - SC typically adds +3-5% accuracy over greedy decoding"
echo "  - Higher N_SAMPLES may improve results but increases time"

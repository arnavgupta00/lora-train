#!/bin/bash
# Cloudflare R2 Sync Script (using rclone for faster incremental sync)
# Syncs any directory to/from R2
#
# Usage:
#   ./scripts/sync_to_r2.sh [MODE] [DIRECTORY]
#
# Modes:
#   push/upload    Upload/sync to R2 (default)
#   pull/download  Download/sync from R2
#   setup          Setup rclone config
#   list           List R2 contents
#   check          Check sync status
#
# Examples:
#   ./scripts/sync_to_r2.sh                      # Upload current project
#   ./scripts/sync_to_r2.sh push /workspace      # Upload remote workspace
#   ./scripts/sync_to_r2.sh push lora-train      # Upload just lora-train folder
#   ./scripts/sync_to_r2.sh pull /workspace      # Download to remote workspace

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RCLONE_REMOTE="r2"

# Parse arguments
MODE="${1:-push}"
WORKSPACE_DIR="${2:-$DEFAULT_DIR}"

# Convert to absolute path
if [[ ! "$WORKSPACE_DIR" = /* ]]; then
    WORKSPACE_DIR="$(cd "$WORKSPACE_DIR" 2>/dev/null && pwd || echo "$PWD/$WORKSPACE_DIR")"
fi

# Generate R2 path based on directory name
DIR_NAME="$(basename "$WORKSPACE_DIR")"
R2_PATH="$RCLONE_REMOTE:model-training/$DIR_NAME"

echo "============================================================"
echo "Cloudflare R2 Workspace Sync (via rclone)"
echo "============================================================"
echo "Local Directory: $WORKSPACE_DIR"
echo "R2 Path: $R2_PATH"
echo "Mode: $MODE"
echo "============================================================"
echo ""

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install rclone
        else
            curl https://rclone.org/install.sh | sudo bash
        fi
    else
        # Linux
        curl https://rclone.org/install.sh | sudo bash
    fi
    
    echo "✓ rclone installed"
    echo ""
fi

# Setup mode - configure rclone
if [[ "$MODE" == "setup" ]]; then
    echo "Setting up rclone for Cloudflare R2..."
    echo ""
    echo "Please enter your R2 credentials:"
    echo ""
    
    read -p "Access Key ID: " ACCESS_KEY_ID
    read -sp "Secret Access Key: " SECRET_ACCESS_KEY
    echo ""
    read -p "R2 Endpoint URL: " ENDPOINT
    
    # Create rclone config
    mkdir -p ~/.config/rclone
    
    cat > ~/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = $ACCESS_KEY_ID
secret_access_key = $SECRET_ACCESS_KEY
endpoint = $ENDPOINT
acl = private
no_check_bucket = true
EOF
    
    echo ""
    echo "✓ rclone configured!"
    echo ""
    echo "Test connection:"
    rclone lsd r2:
    
    echo ""
    echo "Now you can sync with:"
    echo "  ./scripts/sync_to_r2.sh                     # Upload current project"
    echo "  ./scripts/sync_to_r2.sh push /workspace     # Upload remote workspace"
    echo "  ./scripts/sync_to_r2.sh pull /workspace     # Download to remote"
    
    exit 0
fi

# Check if directory exists
if [[ ! -d "$WORKSPACE_DIR" ]]; then
    echo "Error: Directory does not exist: $WORKSPACE_DIR"
    exit 1
fi

# Check if rclone is configured
if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:$"; then
    echo "Error: rclone not configured for R2"
    echo ""
    echo "Run setup first:"
    echo "  ./scripts/sync_to_r2.sh setup"
    exit 1
fi

# Define exclude patterns
EXCLUDE_FILE="$WORKSPACE_DIR/.rsyncignore"
CREATED_EXCLUDE=false

if [[ ! -f "$EXCLUDE_FILE" ]]; then
    cat > "$EXCLUDE_FILE" <<'EOF'
# Version control
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.pytest_cache/
.mypy_cache/
*.egg-info/
.ipynb_checkpoints/

# Node
node_modules/

# System
.DS_Store
*.tmp
.cache/

# Logs (optional - comment out if you want to backup logs)
*.log

# Large model files (optional - comment out if you want to backup models)
# *.safetensors
# *.bin
# *.pt
# *.pth
# models/
# checkpoints/

# HuggingFace cache (usually huge, can be re-downloaded)
hf/
.cache/huggingface/

# Evaluation results (optional - comment out to backup)
# results/*/eval_*/
EOF
    CREATED_EXCLUDE=true
    echo "Created .rsyncignore with default excludes"
    echo "Edit $EXCLUDE_FILE to customize what gets excluded"
fi

if [[ "$MODE" == "push" || "$MODE" == "upload" ]]; then
    # ============================================================
    # PUSH/UPLOAD MODE
    # ============================================================
    echo "Syncing TO R2..."
    echo "From: $WORKSPACE_DIR"
    echo "To: $R2_PATH"
    echo ""
    
    # Dry run first
    echo "Dry run (showing what will be synced)..."
    rclone sync "$WORKSPACE_DIR" "$R2_PATH" \
        --exclude-from "$EXCLUDE_FILE" \
        --progress \
        --dry-run \
        --stats 1s
    
    echo ""
    read -p "Proceed with sync? [y/N]: " CONFIRM
    
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    echo ""
    echo "Syncing..."
    rclone sync "$WORKSPACE_DIR" "$R2_PATH" \
        --exclude-from "$EXCLUDE_FILE" \
        --progress \
        --stats 1s \
        --transfers 8 \
        --checkers 16
    
    echo ""
    echo "✓ Sync to R2 complete!"
    
elif [[ "$MODE" == "pull" || "$MODE" == "download" ]]; then
    # ============================================================
    # PULL/DOWNLOAD MODE
    # ============================================================
    echo "Syncing FROM R2..."
    echo "From: $R2_PATH"
    echo "To: $WORKSPACE_DIR"
    echo ""
    
    # Dry run first
    echo "Dry run (showing what will be synced)..."
    rclone sync "$R2_PATH" "$WORKSPACE_DIR" \
        --exclude-from "$EXCLUDE_FILE" \
        --progress \
        --dry-run \
        --stats 1s
    
    echo ""
    read -p "Proceed with sync? [y/N]: " CONFIRM
    
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    echo ""
    echo "Syncing..."
    rclone sync "$R2_PATH" "$WORKSPACE_DIR" \
        --exclude-from "$EXCLUDE_FILE" \
        --progress \
        --stats 1s \
        --transfers 8 \
        --checkers 16
    
    echo ""
    echo "✓ Sync from R2 complete!"
    
elif [[ "$MODE" == "list" ]]; then
    # ============================================================
    # LIST MODE
    # ============================================================
    echo "Listing R2 contents..."
    rclone ls "$R2_PATH" --max-depth 2
    
elif [[ "$MODE" == "check" ]]; then
    # ============================================================
    # CHECK MODE
    # ============================================================
    echo "Checking sync status..."
    rclone check "$WORKSPACE_DIR" "$R2_PATH" \
        --exclude-from "$EXCLUDE_FILE"
    
else
    echo "Unknown mode: $MODE"
    echo ""
    echo "Usage:"
    echo "  ./scripts/sync_to_r2.sh [MODE] [DIRECTORY]"
    echo ""
    echo "Modes:"
    echo "  push/upload    Upload/sync to R2 (default)"
    echo "  pull/download  Download/sync from R2"
    echo "  setup          Setup rclone config"
    echo "  list           List R2 contents"
    echo "  check          Check sync status"
    echo ""
    echo "Examples:"
    echo "  ./scripts/sync_to_r2.sh                      # Upload current project"
    echo "  ./scripts/sync_to_r2.sh push /workspace      # Upload remote workspace (all folders)"
    echo "  ./scripts/sync_to_r2.sh push lora-train      # Upload just lora-train subfolder"
    echo "  ./scripts/sync_to_r2.sh pull /workspace      # Download to remote workspace"
    exit 1
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

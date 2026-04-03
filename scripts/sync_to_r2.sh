#!/bin/bash
# Cloudflare R2 Sync Script (using rclone for faster incremental sync)
# Syncs workspace directory to/from R2
#
# Usage:
#   ./scripts/sync_to_r2.sh          # Upload/sync to R2
#   ./scripts/sync_to_r2.sh pull     # Download/sync from R2
#   ./scripts/sync_to_r2.sh setup    # Setup rclone config

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RCLONE_REMOTE="r2"
R2_PATH="$RCLONE_REMOTE:model-training/lm-workspace"

MODE="${1:-push}"

echo "============================================================"
echo "Cloudflare R2 Workspace Sync (via rclone)"
echo "============================================================"
echo "Workspace: $WORKSPACE_DIR"
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
    echo "  ./scripts/sync_to_r2.sh          # Upload"
    echo "  ./scripts/sync_to_r2.sh pull     # Download"
    
    exit 0
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
if [[ ! -f "$EXCLUDE_FILE" ]]; then
    cat > "$EXCLUDE_FILE" <<'EOF'
.git/
__pycache__/
*.pyc
.venv/
venv/
node_modules/
.pytest_cache/
.mypy_cache/
*.egg-info/
.DS_Store
*.log
*.tmp
.cache/
results/*/eval_*/
*.safetensors
*.bin
*.pt
*.pth
models/
checkpoints/
EOF
    echo "Created .rsyncignore with default excludes"
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
    echo "  ./scripts/sync_to_r2.sh          # Upload/sync to R2"
    echo "  ./scripts/sync_to_r2.sh pull     # Download/sync from R2"
    echo "  ./scripts/sync_to_r2.sh setup    # Setup rclone config"
    echo "  ./scripts/sync_to_r2.sh list     # List R2 contents"
    echo "  ./scripts/sync_to_r2.sh check    # Check sync status"
    exit 1
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

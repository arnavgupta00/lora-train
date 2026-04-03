#!/bin/bash
# Cloudflare R2 Workspace Backup Script
# Uploads the entire workspace to R2 for backup/resume on another machine
#
# Usage: ./scripts/backup_to_r2.sh [--download]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_NAME="lm-workspace-$(date +%Y%m%d_%H%M%S).tar.gz"

MODE="upload"
if [[ "$1" == "--download" ]]; then
    MODE="download"
fi

echo "============================================================"
echo "Cloudflare R2 Workspace Backup/Restore"
echo "============================================================"
echo "Workspace: $WORKSPACE_DIR"
echo "Mode: $MODE"
echo "============================================================"
echo ""

# Check if credentials are already in environment
if [[ -n "$R2_ACCESS_KEY_ID" && -n "$R2_SECRET_ACCESS_KEY" && -n "$R2_ENDPOINT" ]]; then
    echo "Using credentials from environment variables"
    ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    ENDPOINT="$R2_ENDPOINT"
    BUCKET="${R2_BUCKET:-model-training}"
else
    # Prompt for credentials
    echo "Enter Cloudflare R2 credentials:"
    echo ""
    
    read -p "Access Key ID: " ACCESS_KEY_ID
    read -sp "Secret Access Key: " SECRET_ACCESS_KEY
    echo ""
    read -p "R2 Endpoint URL: " ENDPOINT
    read -p "Bucket Name [model-training]: " BUCKET
    BUCKET="${BUCKET:-model-training}"
    
    echo ""
fi

# Validate inputs
if [[ -z "$ACCESS_KEY_ID" || -z "$SECRET_ACCESS_KEY" || -z "$ENDPOINT" ]]; then
    echo "Error: Missing required credentials"
    exit 1
fi

echo "Configuration:"
echo "  Endpoint: $ENDPOINT"
echo "  Bucket: $BUCKET"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI not found. Installing..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "On macOS, install with: brew install awscli"
    else
        # Linux
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip -q awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
fi

# Configure AWS CLI for R2
export AWS_ACCESS_KEY_ID="$ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"

if [[ "$MODE" == "upload" ]]; then
    # ============================================================
    # UPLOAD MODE
    # ============================================================
    echo "============================================================"
    echo "Creating workspace archive..."
    echo "============================================================"
    
    # Exclude large/temporary files
    EXCLUDE_DIRS=(
        ".git"
        "__pycache__"
        "*.pyc"
        ".venv"
        "venv"
        "node_modules"
        ".pytest_cache"
        ".mypy_cache"
        "*.egg-info"
        ".DS_Store"
    )
    
    # Build tar exclude arguments
    TAR_EXCLUDES=""
    for pattern in "${EXCLUDE_DIRS[@]}"; do
        TAR_EXCLUDES="$TAR_EXCLUDES --exclude=$pattern"
    done
    
    cd "$(dirname "$WORKSPACE_DIR")"
    WORKSPACE_NAME="$(basename "$WORKSPACE_DIR")"
    
    echo "Archiving: $WORKSPACE_NAME"
    echo "Output: $BACKUP_NAME"
    echo ""
    
    tar czf "$BACKUP_NAME" $TAR_EXCLUDES "$WORKSPACE_NAME"
    
    ARCHIVE_SIZE=$(du -h "$BACKUP_NAME" | cut -f1)
    echo "Archive created: $BACKUP_NAME ($ARCHIVE_SIZE)"
    echo ""
    
    # Upload to R2
    echo "============================================================"
    echo "Uploading to R2..."
    echo "============================================================"
    
    aws s3 cp "$BACKUP_NAME" \
        "s3://$BUCKET/$BACKUP_NAME" \
        --endpoint-url "$ENDPOINT" \
        --no-verify-ssl
    
    echo ""
    echo "✓ Upload complete!"
    echo ""
    echo "To download on another machine:"
    echo "  ./scripts/backup_to_r2.sh --download"
    echo ""
    
    # Save metadata
    echo "$BACKUP_NAME" > "$WORKSPACE_DIR/.last_backup"
    
    # Clean up
    rm "$BACKUP_NAME"
    
elif [[ "$MODE" == "download" ]]; then
    # ============================================================
    # DOWNLOAD MODE
    # ============================================================
    echo "============================================================"
    echo "Listing available backups..."
    echo "============================================================"
    
    aws s3 ls "s3://$BUCKET/" \
        --endpoint-url "$ENDPOINT" \
        --no-verify-ssl | grep "lm-workspace-" || true
    
    echo ""
    read -p "Enter backup filename to restore: " BACKUP_TO_RESTORE
    
    if [[ -z "$BACKUP_TO_RESTORE" ]]; then
        echo "Error: No filename provided"
        exit 1
    fi
    
    echo ""
    echo "Downloading: $BACKUP_TO_RESTORE"
    
    aws s3 cp \
        "s3://$BUCKET/$BACKUP_TO_RESTORE" \
        "./$BACKUP_TO_RESTORE" \
        --endpoint-url "$ENDPOINT" \
        --no-verify-ssl
    
    echo ""
    echo "✓ Download complete!"
    echo ""
    read -p "Extract archive? [y/N]: " EXTRACT
    
    if [[ "$EXTRACT" =~ ^[Yy]$ ]]; then
        echo "Extracting..."
        tar xzf "$BACKUP_TO_RESTORE"
        echo "✓ Extracted!"
        
        read -p "Delete archive? [y/N]: " DELETE
        if [[ "$DELETE" =~ ^[Yy]$ ]]; then
            rm "$BACKUP_TO_RESTORE"
            echo "✓ Archive deleted"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

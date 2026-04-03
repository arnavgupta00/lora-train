#!/bin/bash
# Cloudflare R2 Workspace Backup Script
# Creates timestamped tar.gz archives and uploads to R2
#
# Usage: 
#   ./scripts/backup_to_r2.sh [DIRECTORY]
#   ./scripts/backup_to_r2.sh --download BACKUP_NAME [TARGET_DIR]
#
# Examples:
#   ./scripts/backup_to_r2.sh                    # Backup current project
#   ./scripts/backup_to_r2.sh /workspace         # Backup remote workspace
#   ./scripts/backup_to_r2.sh lora-train         # Backup just lora-train
#   ./scripts/backup_to_r2.sh --download workspace-20260402_123456.tar.gz

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="upload"
if [[ "$1" == "--download" ]]; then
    MODE="download"
    BACKUP_NAME="${2:-}"
    TARGET_DIR="${3:-.}"
    
    if [[ -z "$BACKUP_NAME" ]]; then
        echo "Error: Must specify backup name to download"
        echo "Usage: $0 --download BACKUP_NAME [TARGET_DIR]"
        exit 1
    fi
else
    WORKSPACE_DIR="${1:-$DEFAULT_DIR}"
    
    # Convert to absolute path
    if [[ ! "$WORKSPACE_DIR" = /* ]]; then
        WORKSPACE_DIR="$(cd "$WORKSPACE_DIR" 2>/dev/null && pwd || echo "$PWD/$WORKSPACE_DIR")"
    fi
    
    # Check if directory exists
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        echo "Error: Directory does not exist: $WORKSPACE_DIR"
        exit 1
    fi
    
    DIR_NAME="$(basename "$WORKSPACE_DIR")"
    BACKUP_NAME="${DIR_NAME}-$(date +%Y%m%d_%H%M%S).tar.gz"
fi

echo "============================================================"
echo "Cloudflare R2 Backup/Restore"
echo "============================================================"
if [[ "$MODE" == "upload" ]]; then
    echo "Directory: $WORKSPACE_DIR"
    echo "Backup Name: $BACKUP_NAME"
else
    echo "Backup: $BACKUP_NAME"
    echo "Target: $TARGET_DIR"
fi
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
    EXCLUDE_PATTERNS=(
        ".git"
        "__pycache__"
        "*.pyc"
        "*.pyo"
        ".venv"
        "venv"
        "node_modules"
        ".pytest_cache"
        ".mypy_cache"
        "*.egg-info"
        ".DS_Store"
        "*.log"
        "hf"
        ".cache"
    )
    
    # Build tar exclude arguments
    TAR_EXCLUDES=""
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
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
    echo "Backup saved as: s3://$BUCKET/$BACKUP_NAME"
    echo ""
    echo "To restore on another machine:"
    echo "  ./scripts/backup_to_r2.sh --download $BACKUP_NAME"
    echo ""
    
    # Clean up local archive
    rm "$BACKUP_NAME"
    
elif [[ "$MODE" == "download" ]]; then
    # ============================================================
    # DOWNLOAD MODE
    # ============================================================
    echo "============================================================"
    echo "Downloading backup..."
    echo "============================================================"
    
    echo "Downloading: $BACKUP_NAME"
    echo "To: $TARGET_DIR"
    echo ""
    
    mkdir -p "$TARGET_DIR"
    cd "$TARGET_DIR"
    
    aws s3 cp \
        "s3://$BUCKET/$BACKUP_NAME" \
        "./$BACKUP_NAME" \
        --endpoint-url "$ENDPOINT" \
        --no-verify-ssl
    
    echo ""
    echo "✓ Download complete!"
    echo ""
    read -p "Extract archive now? [Y/n]: " EXTRACT
    
    if [[ ! "$EXTRACT" =~ ^[Nn]$ ]]; then
        echo "Extracting..."
        tar xzf "$BACKUP_NAME"
        echo "✓ Extracted!"
        echo ""
        
        read -p "Delete archive file? [Y/n]: " DELETE
        if [[ ! "$DELETE" =~ ^[Nn]$ ]]; then
            rm "$BACKUP_NAME"
            echo "✓ Archive deleted"
        fi
    fi
fi

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

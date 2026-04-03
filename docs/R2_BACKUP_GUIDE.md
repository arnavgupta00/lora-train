# R2 Backup & Sync Guide

## Two Options Available

### Option 1: Full Archive Backup (Simple)
Use `backup_to_r2.sh` for complete workspace backups as tar.gz archives.

**Best for:** Saving snapshots, moving between machines rarely

```bash
# Upload (creates timestamped archive)
./scripts/backup_to_r2.sh

# Download (lists available backups)
./scripts/backup_to_r2.sh --download
```

### Option 2: Incremental Sync (Recommended)
Use `sync_to_r2.sh` with rclone for fast incremental syncing.

**Best for:** Regular syncing, working across multiple machines

```bash
# First time setup
./scripts/sync_to_r2.sh setup
# Enter credentials when prompted

# Upload/sync workspace to R2
./scripts/sync_to_r2.sh

# Download/sync from R2 to workspace
./scripts/sync_to_r2.sh pull

# List what's in R2
./scripts/sync_to_r2.sh list

# Check sync status
./scripts/sync_to_r2.sh check
```

## Your R2 Credentials

Store these in a safe place:

```bash
# For environment variables (add to ~/.bashrc or ~/.zshrc)
export R2_ACCESS_KEY_ID="22a8c72f51d0605d12a5e55280616a08"
export R2_SECRET_ACCESS_KEY="2e1f230964651cf3c5c674cba733d42ee6089779ecb3601776ad44b4c6c850ae"
export R2_ENDPOINT="https://604aa516310f147817bea0cf2c82336b.r2.cloudflarestorage.com"
export R2_BUCKET="model-training"
```

## Quick Start

### Setup on New Machine

```bash
# Clone repo
git clone <your-repo-url> lm
cd lm

# Setup rclone
./scripts/sync_to_r2.sh setup
# Enter:
#   Access Key ID: 22a8c72f51d0605d12a5e55280616a08
#   Secret Access Key: 2e1f230964651cf3c5c674cba733d42ee6089779ecb3601776ad44b4c6c850ae
#   R2 Endpoint: https://604aa516310f147817bea0cf2c82336b.r2.cloudflarestorage.com

# Download workspace
./scripts/sync_to_r2.sh pull
```

### Regular Workflow

```bash
# Before starting work (get latest)
./scripts/sync_to_r2.sh pull

# Do your work...

# After work (save to cloud)
./scripts/sync_to_r2.sh
```

## What Gets Synced

### Included
- Source code (`.py`, `.sh`, `.md`)
- Configuration files
- Documentation
- Small results/logs

### Excluded (see `.rsyncignore`)
- `.git/` directory
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.venv/`)
- Node modules
- Large model files (`*.safetensors`, `*.bin`, `*.pt`)
- Temporary files (`*.log`, `*.tmp`)
- Large evaluation results

## Edit Exclusions

Edit `.rsyncignore` in workspace root to customize what gets synced:

```bash
# Add patterns (one per line)
echo "*.big_file" >> .rsyncignore
echo "temp_data/" >> .rsyncignore
```

## Troubleshooting

### "rclone not found"
Scripts will auto-install rclone. Or install manually:
```bash
# macOS
brew install rclone

# Linux
curl https://rclone.org/install.sh | sudo bash
```

### "Remote not configured"
Run setup:
```bash
./scripts/sync_to_r2.sh setup
```

### Check what will be synced (dry run)
The scripts show dry run before actual sync - review carefully!

### Manual rclone commands
```bash
# List remotes
rclone listremotes

# List bucket
rclone ls r2:model-training

# Manual sync
rclone sync . r2:model-training/lm-workspace --dry-run
```

## Archive vs Sync Comparison

| Feature | Archive (backup_to_r2.sh) | Sync (sync_to_r2.sh) |
|---------|---------------------------|----------------------|
| Speed | Slow (full copy) | Fast (incremental) |
| Storage | Multiple snapshots | Single latest copy |
| Best for | Backups, snapshots | Daily sync |
| Size | Larger (compressed tar.gz) | Smaller (only changes) |
| Restore | Download + extract | Direct sync |

## Cost Estimation

Cloudflare R2:
- Storage: $0.015/GB/month
- Class A operations (write): $4.50/million
- Class B operations (read): $0.36/million
- Egress: FREE (no bandwidth charges!)

Typical workspace: ~500MB
- Storage cost: ~$0.01/month
- Sync operations: ~$0.05/month
- **Total: ~$0.10/month**

## Security Notes

- Credentials are stored locally in `~/.config/rclone/rclone.conf`
- Do NOT commit credentials to git
- Add `.rclone.conf` to `.gitignore` if storing config in workspace
- Consider using environment variables instead of storing in config

## Advanced: Multiple Backups

```bash
# Backup with custom name
R2_PATH="r2:model-training/backup-$(date +%Y%m%d)" ./scripts/sync_to_r2.sh

# Sync specific directory only
rclone sync ./results r2:model-training/results-only
```

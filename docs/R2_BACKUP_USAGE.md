# R2 Backup & Sync - Usage Guide

## Quick Reference

### For Local Development (Mac)
```bash
# Upload entire project
./scripts/sync_to_r2.sh

# Or use archive backup
./scripts/backup_to_r2.sh
```

### For Remote GPU Machine
```bash
# Upload entire /workspace (includes hf, lora-train, outputs)
./scripts/sync_to_r2.sh push /workspace

# Or just lora-train folder
./scripts/sync_to_r2.sh push /workspace/lora-train

# Archive backup
./scripts/backup_to_r2.sh /workspace
```

---

## sync_to_r2.sh (Recommended)

**Fast incremental sync using rclone** - only uploads changed files.

### First Time Setup
```bash
./scripts/sync_to_r2.sh setup

# Enter credentials when prompted:
Access Key ID: 22a8c72f51d0605d12a5e55280616a08
Secret: 2e1f230964651cf3c5c674cba733d42ee6089779ecb3601776ad44b4c6c850ae
Endpoint: https://604aa516310f147817bea0cf2c82336b.r2.cloudflarestorage.com
```

### Upload Examples
```bash
# Upload current directory
./scripts/sync_to_r2.sh

# Upload entire remote workspace
./scripts/sync_to_r2.sh push /workspace

# Upload specific subfolder
./scripts/sync_to_r2.sh push /workspace/lora-train
./scripts/sync_to_r2.sh push /workspace/outputs
```

### Download Examples
```bash
# Download to current location
./scripts/sync_to_r2.sh pull

# Download to specific location
./scripts/sync_to_r2.sh pull /workspace
./scripts/sync_to_r2.sh pull /workspace/lora-train
```

### Other Commands
```bash
# List what's in R2
./scripts/sync_to_r2.sh list

# Check sync status (what's different)
./scripts/sync_to_r2.sh check
```

### What Gets Excluded?
Edit `.rsyncignore` in your workspace to customize. Defaults:
- `.git/` (version control)
- `__pycache__/`, `*.pyc` (Python cache)
- `hf/` (HuggingFace cache - huge!)
- `*.log` (logs)
- System files (`.DS_Store`, etc.)

**Model files are NOT excluded by default** - uncomment in `.rsyncignore` if you want to skip them.

---

## backup_to_r2.sh (Archive Backups)

**Creates timestamped tar.gz archives** - good for snapshots.

### Upload Examples
```bash
# Backup current directory
./scripts/backup_to_r2.sh

# Backup entire remote workspace
./scripts/backup_to_r2.sh /workspace

# Backup specific folder
./scripts/backup_to_r2.sh /workspace/lora-train
```

### Download & Restore
```bash
# Download and extract
./scripts/backup_to_r2.sh --download workspace-20260402_123456.tar.gz

# Download to specific location
./scripts/backup_to_r2.sh --download workspace-20260402_123456.tar.gz /workspace
```

### What Gets Excluded?
- `.git/`, `__pycache__/`, `*.pyc`
- `hf/`, `.cache/` (HuggingFace cache)
- `*.log` (logs)
- Virtual envs (`venv/`, `.venv/`)

---

## Common Workflows

### Daily Work on Remote GPU
```bash
# End of day - backup everything
cd /workspace
./scripts/sync_to_r2.sh push /workspace

# Next day on different machine - restore
./scripts/sync_to_r2.sh setup  # first time only
./scripts/sync_to_r2.sh pull /workspace
```

### Backup Just the Important Stuff
```bash
# Backup trained models
./scripts/sync_to_r2.sh push /workspace/outputs

# Backup code changes
./scripts/sync_to_r2.sh push /workspace/lora-train
```

### Full Snapshot Before Major Changes
```bash
# Create timestamped archive
./scripts/backup_to_r2.sh /workspace
# Creates: workspace-20260402_153045.tar.gz

# Later, restore exact snapshot
./scripts/backup_to_r2.sh --download workspace-20260402_153045.tar.gz
```

---

## R2 Storage Structure

```
s3://model-training/
├── workspace/          (from: /workspace)
│   ├── hf/            (excluded by default)
│   ├── lora-train/
│   └── outputs/
├── lora-train/        (from: /workspace/lora-train)
└── workspace-20260402_123456.tar.gz  (archive backups)
```

Each directory you backup gets its own folder in R2 based on the directory name.

---

## Costs & Limits

- **Storage**: $0.015/GB/month (~$0.10/month for typical 5-10GB workspace)
- **Egress**: FREE (Cloudflare doesn't charge for downloads!)
- **Operations**: Basically free for this use case

**Tip**: Exclude the `hf/` cache folder (done by default) - it can be 10-50GB and is re-downloadable.

---

## Troubleshooting

### "rclone not configured"
Run setup first:
```bash
./scripts/sync_to_r2.sh setup
```

### "Directory does not exist"
Make sure you're using the correct path:
```bash
# Check what exists
ls -la /workspace

# Use absolute path
./scripts/sync_to_r2.sh push /workspace
```

### Sync is slow
Check what's being synced:
```bash
# See what will be uploaded
./scripts/sync_to_r2.sh push /workspace
# It shows dry-run first - look for large files

# Edit exclusions
nano /workspace/.rsyncignore
```

### Want to exclude model checkpoints?
Add to `.rsyncignore`:
```
*.safetensors
*.bin
*.pt
*.pth
checkpoints/
```

---

## Security Notes

- Credentials are stored in `~/.config/rclone/rclone.conf` (chmod 600)
- Archives are encrypted in transit (HTTPS)
- Bucket is private (acl = private)
- Never commit credentials to git

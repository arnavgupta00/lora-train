#!/usr/bin/env bash
# Setup Spider benchmark for evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="${SCRIPT_DIR}/spider_data"

mkdir -p "$SPIDER_DIR"
cd "$SPIDER_DIR"

echo "=== Setting up Spider Benchmark ==="

# Download Spider via HuggingFace
python3 << 'PY'
import os
from datasets import load_dataset

print("Loading Spider dataset from HuggingFace...")
ds = load_dataset("spider", trust_remote_code=True)

import json

# Train set
with open("train.json", "w") as f:
    json.dump([dict(x) for x in ds["train"]], f, indent=2)
print(f"Saved {len(ds['train'])} train examples")

# Validation/Dev set (this is what you report)
with open("dev.json", "w") as f:
    json.dump([dict(x) for x in ds["validation"]], f, indent=2)
print(f"Saved {len(ds['validation'])} dev examples")

print("\nSpider dataset ready!")
PY

# Download the actual SQLite databases (needed for execution eval)
echo ""
echo "Downloading Spider databases..."
if [ ! -d "database" ]; then
    wget -q https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m -O spider.zip 2>/dev/null || \
    curl -L "https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m" -o spider.zip 2>/dev/null || \
    echo "Note: Could not auto-download databases. Download manually from: https://yale-lily.github.io/spider"
    
    if [ -f spider.zip ]; then
        unzip -q spider.zip
        echo "Databases extracted!"
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo "Next: python3 ${SCRIPT_DIR}/convert_spider_to_chatml.py"

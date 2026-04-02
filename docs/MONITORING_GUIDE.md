# Training Monitoring & Troubleshooting Guide

Quick reference for monitoring and debugging your Qwen3-1.7B training.

## Check if Training is Running

### Quick Check
```bash
# Simple process check
ps aux | grep [p]ython

# More specific
pgrep -f train_lora.py
pgrep -f train_grpo.py

# Check with details
ps -ef | grep -E "train_lora|train_grpo" | grep -v grep
```

### Live Monitoring
```bash
# Watch processes (updates every 2 seconds)
watch -n 2 'ps aux | grep [t]rain'

# Monitor GPU usage
nvidia-smi

# Real-time GPU monitoring
watch -n 2 nvidia-smi
```

## View Training Logs

### From your nohup session
```bash
# Follow log in real-time
tail -f pipeline.log

# Last 50 lines
tail -50 pipeline.log

# Last 100 lines
tail -100 pipeline.log
```

### Search logs for specific info
```bash
# View all loss values
tail -200 pipeline.log | grep loss

# Check for errors
tail -100 pipeline.log | grep -i "error\|failed\|cuda"

# View recent progress
tail -20 pipeline.log

# See epoch/step progress
tail -100 pipeline.log | grep -E "epoch|step|Epoch"

# Find training times
tail -50 pipeline.log | grep "time\|duration\|completed"
```

## GPU Monitoring

### Current GPU status
```bash
# Simple status
nvidia-smi

# Detailed memory info
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature --format=csv

# Watch GPU continuously
watch -n 1 nvidia-smi

# Process-level GPU usage (if nvidia-utils installed)
nvidia-smi pmon -c 1
```

### GPU Utilities
```bash
# Python GPU memory check
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB')"

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## Monitoring Setup (Multi-Terminal)

Recommended setup for effective monitoring:

### Terminal 1: Training Log
```bash
cd /workspace/lora-train
tail -f pipeline.log
```

### Terminal 2: GPU Status
```bash
watch -n 2 nvidia-smi
```

### Terminal 3: Process Check
```bash
watch -n 5 'ps aux | grep [t]rain'
```

### Terminal 4: Results (after training)
```bash
ls -la results/
tail -f results/pipeline_*/eval_results/*
```

## Expected Behavior During Training

### SFT Phase (2-3 hours)
```
✓ High GPU memory usage (20-24GB)
✓ Loss decreasing over epochs
✓ ~50-100 step intervals
✓ Checkpoint saves every 500 steps
```

Example log output:
```
Epoch 1/3: loss=2.234
Epoch 1/3: loss=1.856
Epoch 1/3: loss=1.523
...
Saved checkpoint to outputs/qwen3-1.7b-sft-*/checkpoint-500
```

### GRPO Phase (4-6 hours, if enabled)
```
✓ Lower GPU memory than SFT
✓ Reward values 0.0-1.0
✓ Slower step progress
✓ Generation + evaluation overhead
```

Example log output:
```
GRPO Training...
Generation: [████████░░] 85% - 68/80
Reward: avg=0.45, min=0.0, max=1.0
Step 10/100
```

### Eval Phase (1-2 hours, if enabled)
```
✓ Varied GPU memory usage
✓ SQL execution and evaluation
✓ Accuracy percentage output
```

Example log output:
```
Evaluating: baseline
Generated 500 SQL queries
Execution accuracy: 46.23%

Evaluating: sft_model
Generated 500 SQL queries
Execution accuracy: 58.45%
```

## If Training Seems Stuck

### Check if process is alive
```bash
# Find the process ID
PID=$(pgrep -f train_lora.py)
if [ -n "$PID" ]; then
    echo "Process running with PID: $PID"
    ps -p $PID -o etime=  # Show elapsed time
else
    echo "Process not running"
fi
```

### Check for errors
```bash
# Last 50 lines with timestamps
tail -50 pipeline.log

# Search for error patterns
grep -i "error\|exception\|traceback" pipeline.log

# Check for CUDA errors
grep -i "cuda\|memory\|out of" pipeline.log
```

### Common Issues & Solutions

**OOM (Out of Memory)**
```
Look for: "CUDA out of memory"
Solution: Kill process and restart with smaller batch size
kill $(pgrep -f train_lora.py)
BATCH_SIZE=2 SEQ_LEN=1024 bash training/configs/qwen3-1.7b-full-pipeline.sh
```

**Slow Training**
```
Check: GPU utilization should be 80-100%
watch -n 1 nvidia-smi
If low: Increase BATCH_SIZE or reduce SEQ_LEN
```

**No Progress**
```
Check: Is loss decreasing?
tail -50 pipeline.log | grep loss

If stuck: Kill and restart
kill $(pgrep -f train_lora.py)
```

## After Training Completes

### Check Results
```bash
# View summary
cat results/pipeline_*/SUMMARY.txt

# Check model outputs
ls -la outputs/qwen3-1.7b-sft-*
ls -la outputs/qwen3-1.7b-grpo-*

# View evaluation results
ls -la results/pipeline_*/eval_results/
cat results/pipeline_*/eval_results/*.json | python3 -m json.tool
```

### Verify Results
```bash
# Check if all phases completed
grep -E "Complete|Summary" results/pipeline_*/pipeline.log

# Check final accuracy
tail -20 results/pipeline_*/SUMMARY.txt

# Compare baseline vs trained
python3 -c "
import json
import glob

for f in glob.glob('results/pipeline_*/eval_results/*.json'):
    with open(f) as fp:
        data = json.load(fp)
        print(f'{f.split(\"/\")[-1]}: {data.get(\"accuracy\", 0)*100:.2f}%')
"
```

## Kill Training if Needed

```bash
# Find and kill by process name
pkill -f train_lora.py
pkill -f train_grpo.py

# Or by PID
kill <PID>

# Force kill if needed
kill -9 <PID>

# Verify it's dead
ps aux | grep [t]rain
```

## Full Monitoring Dashboard (One Command)

Create an alias for comprehensive monitoring:

```bash
# Add to ~/.bashrc or run directly
watch -n 1 'echo "=== PROCESSES ===" && ps aux | grep [t]rain && echo "" && echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,temperature,memory.used,memory.total --format=csv,noheader'
```

Or create a script:

```bash
#!/bin/bash
while true; do
    clear
    echo "=== TRAINING MONITOR ==="
    echo "Time: $(date)"
    echo ""
    echo "=== Process Status ==="
    ps aux | grep [t]rain | grep -v grep || echo "No training process"
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,temperature,memory.used,memory.total --format=csv
    echo ""
    echo "=== Recent Log ==="
    tail -5 pipeline.log 2>/dev/null
    echo ""
    sleep 2
done
```

Save as `monitor.sh` and run:
```bash
bash monitor.sh
```

## Useful grep patterns for logs

```bash
# Show all training metrics
grep -E "loss|accuracy|eval" pipeline.log

# Show only errors and warnings
grep -i "error\|warning\|failed" pipeline.log

# Show phase transitions
grep -E "Phase|Starting|Completed" pipeline.log

# Show timing information
grep -i "time|duration|seconds" pipeline.log

# Show memory usage
grep -i "memory|cuda|gpu" pipeline.log

# Combine multiple patterns
grep -E "(error|cuda|memory|failed)" pipeline.log | tail -20
```

# How to Recover t3 Dataset

The t3 dataset (23,577 examples) is currently **missing from local machine**.

## Location History
- **RunPod:** `/workspace/dataset/all-all-train.qwen.jsonl` (used for 14B training March 30)
- **Local (expected):** `data/t3_test1000_rebalanced/all-all-train.qwen.jsonl`
- **Status:** NOT PRESENT

## Option 1: Download from RunPod
If your RunPod workspace still exists:

```bash
# SSH into RunPod
ssh root@<your-runpod-ip>

# Check if dataset exists
ls -lh /workspace/dataset/all-all-train.qwen.jsonl

# Download to local machine
scp root@<runpod-ip>:/workspace/dataset/all-all-train.qwen.jsonl \
    ~/programming/lm/data/t3_test1000_rebalanced/
```

## Option 2: Regenerate Dataset
The t3 dataset needs to be regenerated from source. We don't currently have a script for this.

You'll need to:
1. Identify what sources were used to create t3
2. Create a generation script similar to `tools/create_t7_dataset.py`
3. Run the generation

## Option 3: Use t7 Instead
The **t7 dataset already exists** at `data/training/t7/` and has:
- 16,699 examples (vs t3's 23,577)
- BIRD training data + custom schemas
- Used for successful 7B training

**Recommendation:** Focus on t7 since:
- It's already here
- Achieved 44.26% on BIRD
- More focused on BIRD benchmark
- t3 was just an intermediate step

## Files Currently Available
✅ `data/training/t2/` - 562 examples (early version)
✅ `data/training/t7/` - 16,699 examples (current best)
❌ `data/t3_test1000_rebalanced/` - MISSING (23,577 examples)

## Action Required
**Tell me which option you want:**
1. Connect to RunPod and download t3
2. Regenerate t3 from scratch (need more info on sources)
3. Skip t3, use t7 for future work

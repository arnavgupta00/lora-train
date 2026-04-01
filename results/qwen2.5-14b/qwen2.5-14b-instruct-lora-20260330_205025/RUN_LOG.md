# Run: Qwen2.5-14B with t3 Dataset

**Model:** Qwen2.5-14B-Instruct + LoRA  
**Dataset:** t3 rebalanced (23,577 training examples)  
**Date:** March 30, 2026  
**Status:** ✅ Training completed, limited evaluation

---

## 🔗 Quick Links

**Files used in this run:**
- 📊 **Training Dataset:** [`dataset_used_t3/`](dataset_used_t3/) → `data/t3_test1000_rebalanced/` ⚠️ **MISSING**
- ⚙️ **Training Config:** [`training_config.sh`](training_config.sh) → `training/configs/qwen2.5-14b.sh`
- 📈 **Evaluation Results:** 
  - [`eval_report.base.json`](eval_report.base.json) - Base model: 12.2%
  - [`eval_report.lora.json`](eval_report.lora.json) - LoRA model: 24.4%
- 🧠 **Model Adapters:** `adapter_model.safetensors` (154MB, gitignored)

---

## 📊 Results Summary

| Model | Execution Match | Improvement |
|-------|-----------------|-------------|
| Base 14B | 12.2% | Baseline |
| LoRA 14B | 24.4% | **+12.2%** (2x) |

### Training Details
- **Loss:** Converged to 0.04 (excellent)
- **Time:** ~1.5 hours
- **Effective batch size:** 16 (BS=4, GA=4)
- **Learning rate:** 2e-4

---

## ⚠️ Important Notes

1. **Dataset Missing:** The t3 dataset (23,577 examples) is NOT in git
   - Was located at `/workspace/dataset/` on RunPod
   - See [`RECOVER_T3_DATASET.md`](../../RECOVER_T3_DATASET.md) for recovery options

2. **Limited Evaluation:** Only tested on custom schemas (562 examples)
   - NOT evaluated on BIRD benchmark
   - The v1-3 run (7B + t7) achieved 44.26% on BIRD

3. **Superseded by v1-3:**
   - v1-3 used t7 dataset (more focused on BIRD)
   - v1-3 achieved 44.26% on official BIRD benchmark
   - This run was an intermediate experiment

---

## 📝 Training Configuration

```json
{
  "model": "Qwen/Qwen2.5-14B-Instruct",
  "dataset": "t3 (23,577 examples) - MISSING",
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05
  },
  "training": {
    "batch_size": 4,
    "gradient_accumulation": 4,
    "effective_batch": 16,
    "learning_rate": 2e-4,
    "epochs": 1,
    "time": "~1.5 hours"
  }
}
```

---

## 🔍 What We Learned

1. **14B vs 7B:** 14B shows better base performance (12.2% vs lower)
2. **LoRA doubles performance:** 12.2% → 24.4% improvement
3. **Need BIRD evaluation:** Can't compare to benchmarks without it
4. **Custom eval insufficient:** 24.4% on custom schemas doesn't predict BIRD performance

---

## 🎯 Next Steps (Completed in v1-3)

The v1-3 run addressed these issues:
- ✅ Used t7 dataset (BIRD-focused, 16,699 examples)
- ✅ Ran full BIRD benchmark evaluation
- ✅ Achieved 44.26% execution accuracy
- ✅ Properly formatted schemas (DDL)

See [`../qwen2.5-7b/v1-3/RUN_LOG.md`](../qwen2.5-7b/v1-3/RUN_LOG.md) for current best results.

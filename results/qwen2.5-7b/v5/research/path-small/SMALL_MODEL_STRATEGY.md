# Small Model Path: The "Wow Factor" Approach

## The Narrative

**"A 0.5B model that beats GPT-4 on enterprise SQL generation"**

This is the wow factor. Not incremental improvement, but a paradigm shift:
- Runs on **edge devices** (phones, laptops, embedded)
- **100x cheaper** inference than GPT-4
- **Private & secure** - no data leaves the device
- **Real-time** - sub-second latency

---

## What's Already Been Done (State of the Art)

### SLM-SQL Results (July 2025)

| Model | Size | BIRD Dev | BIRD Test | Beats |
|-------|------|----------|-----------|-------|
| Qwen2.5-Coder-0.5B | **0.5B** | 56.87% | 61.82% | GPT-4 (54.89%) ✅ |
| Qwen2.5-Coder-1.5B | **1.5B** | 67.08% | 70.49% | Claude Opus (70.15%) ✅ |

**Key insight**: A 0.5B model (500 million parameters) beats GPT-4's 54.89%!

### How They Did It

1. **High-quality training data**: SynSQL-2.5M dataset
2. **Think-style prompting**: Model reasons before generating SQL
3. **GRPO (RL) training**: Execution-based rewards
4. **Self-consistency voting**: Generate 10, vote on results
5. **Merge revision model**: Fix errors with second model

### Struct-SQL Results (Jan 2026)

| Model | Size | BIRD Test | Method |
|-------|------|-----------|--------|
| Qwen3-4B | **4B** | 60.42% | Structured CoT distillation |

Achieves 60% with **only 1,000 training samples** and 29 minutes of training!

---

## Our Opportunity

### What We Can Build

> ⚠️ **Option A (Using pre-trained SLM-SQL directly) is NOT recommended**
> - Readers will see you "just used someone else's model"
> - Devalues your effort and credibility
> - No differentiation from existing work

**✅ Option B: Train Your Own Model (Recommended)**
- Base: Qwen2.5-Coder-0.5B, 1.5B, or 3B
- Training: Your T9 dataset + SFT + GRPO
- Inference: Self-consistency voting
- **YOUR training data, YOUR fine-tuning, YOUR results**
- See: `OPTION_B_TRAIN_YOUR_OWN.md`

**✅ Option C: Knowledge Distillation (Most Impressive)**
- Distill GPT-4's SQL reasoning into a small model
- Generate synthetic training data with structured reasoning
- Train student to mimic teacher's thinking process
- **Advanced ML technique, high academic/industry value**
- See: `OPTION_C_DISTILLATION.md`

### Why These Options Maintain Credibility

Both Option B and C are **100% YOUR work**:
- You curate/generate the training data
- You implement the training pipeline
- You build the inference system
- You achieve the benchmark results

The base model (Qwen2.5-Coder) is just a starting point - like using PyTorch doesn't mean someone else wrote your neural network.

---

## BIRD Leaderboard Rules

### What's Allowed

✅ **Multiple model calls** - Self-consistency uses 10-20 generations
✅ **Ensemble/voting** - Standard practice
✅ **Multiple specialized models** - Schema filter + Generator is common
✅ **Post-processing** - Execution-based validation
✅ **Any model size** - No restrictions

### What's Required

- Must report model size/parameters
- Must be reproducible
- Test set submissions verified by BIRD team

---

## Proposed Architecture: "TinySQL Pipeline"

### Design: 3 Small Models Working Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    TinySQL Pipeline (~2.5B total)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Question + Full Schema                                  │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐                                        │
│  │  Schema Filter      │  0.5B - Qwen2.5-Coder-0.5B             │
│  │  "Which tables?"    │  Selects relevant tables/columns       │
│  └─────────────────────┘                                        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐                                        │
│  │  SQL Generator      │  1.5B - Qwen2.5-Coder-1.5B             │
│  │  "Generate SQL"     │  Produces SQL candidates (n=10)        │
│  └─────────────────────┘                                        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐                                        │
│  │  Execution Check    │  SQLite (no model needed)              │
│  │  "Does it run?"     │  Validates syntax, returns results     │
│  └─────────────────────┘                                        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────┐                                        │
│  │  Result Voter       │  Simple algorithm                      │
│  │  "Most common?"     │  Selects majority result               │
│  └─────────────────────┘                                        │
│           │                                                     │
│  (Optional) If error:   │                                       │
│           ▼                                                     │
│  ┌─────────────────────┐                                        │
│  │  Error Corrector    │  0.5B - Fine-tuned for SQL repair     │
│  │  "Fix this SQL"     │  Corrects syntax/column errors         │
│  └─────────────────────┘                                        │
│           │                                                     │
│           ▼                                                     │
│  Output: Final SQL                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Total Parameters: ~2.5B (runs on any modern laptop GPU)
```

### Why This Architecture?

1. **Schema Filter (0.5B)**: Reduces context length, improves accuracy
2. **SQL Generator (1.5B)**: Main workhorse, generates candidates
3. **Voting**: Free accuracy boost (+3-5%)
4. **Error Corrector (0.5B)**: Handles edge cases

### Expected Performance

| Component | Contribution |
|-----------|-------------|
| Base 1.5B generator | ~50% |
| + Schema filtering | +5-8% |
| + Self-consistency (n=10) | +5-8% |
| + Error correction | +2-3% |
| **Total** | **62-69%** |

---

## Implementation Plan

### Phase 1: Baseline (1 week)

1. Download SLM-SQL models from HuggingFace
2. Run inference on BIRD dev set
3. Verify published results (~67% for 1.5B)

```bash
# Models to download
cycloneboy/SLM-SQL-0.5B
cycloneboy/SLM-SQL-1.5B
cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct
```

### Phase 2: Custom Training (2 weeks)

1. Fine-tune Qwen2.5-Coder-1.5B on T9 dataset
2. Train schema filter on BIRD schemas
3. Train error corrector on execution failures

### Phase 3: Pipeline Integration (1 week)

1. Build inference pipeline
2. Implement voting mechanism
3. Add error correction loop

### Phase 4: Optimization (1 week)

1. Quantize models (INT8/INT4)
2. Optimize inference speed
3. Package for deployment

---

## Hardware Requirements

### Training
- **GPU**: 1x RTX 4090 (24GB) or equivalent
- **Time**: ~2-4 hours per model
- **Cost**: ~$10-20 on cloud

### Inference
- **Minimum**: 8GB VRAM (INT4 quantized)
- **Recommended**: 16GB VRAM (FP16)
- **CPU-only**: Possible with llama.cpp (slower)

### Deployment Options
- **Edge**: Runs on M1/M2 MacBooks, gaming laptops
- **Cloud**: Single T4 GPU ($0.35/hour)
- **Mobile**: With quantization, runs on phones (future)


### Key Differentiators

1. **Size**: 1.5B vs GPT-4's ~1.7T (1000x smaller)
2. **Cost**: ~$0.0001/query vs ~$0.01/query (100x cheaper)
3. **Latency**: <1 second vs 2-5 seconds
4. **Privacy**: On-device vs cloud API
5. **Availability**: Works offline

---

## Risk Assessment

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Can't reproduce SLM-SQL results | Use their published models |
| Training fails | Start with their checkpoints, fine-tune |
| Pipeline too slow | Quantize models, batch processing |

### Competitive Risks

| Risk | Mitigation |
|------|------------|
| SLM-SQL already published | Add novel contributions (domain adaptation, deployment) |
| Larger models get better | Focus on edge/privacy narrative |

---

## Timeline

| Week | Goal | Deliverable |
|------|------|-------------|
| 1 | Baseline | Reproduce SLM-SQL results |
| 2-3 | Training | T9 dataset + fine-tuned models |
| 4 | Pipeline | Working 3-model pipeline |
| 5 | Polish | Quantization, benchmarks, blog post |

**Total: 5 weeks to "wow factor" demo**

---

## Next Steps

1. **Download SLM-SQL models** and verify results
2. **Create T9 dataset** with correct distribution
3. **Design schema filter** training data
4. **Build pipeline** infrastructure
5. **Benchmark** and iterate

---

## References

1. SLM-SQL Paper: [arXiv:2507.22478](https://arxiv.org/abs/2507.22478)
2. CSC-SQL Paper: [arXiv:2505.13271](https://arxiv.org/abs/2505.13271)
3. Struct-SQL Paper: [arXiv:2512.17053](https://arxiv.org/abs/2512.17053)
4. SLM-SQL Models: [huggingface.co/cycloneboy](https://huggingface.co/cycloneboy)
5. Struct-SQL Code: [github.com/craterlabs/Struct-SQL-Distillation](https://github.com/craterlabs/Struct-SQL-Distillation)

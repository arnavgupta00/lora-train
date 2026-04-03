# Implementation Roadmap: Small Model Path

## Executive Summary

**Goal**: Achieve 60%+ on BIRD with models that run on any laptop (total <3B parameters)

**Approach**: Use SLM-SQL's published models + self-consistency inference

**Timeline**: 2-3 weeks to working demo

---

## Phase 1: Baseline Verification (Days 1-3)

### Objective
Verify SLM-SQL's published results with their models

### Tasks

1. **Download Models**
   ```bash
   # Generator models
   huggingface-cli download cycloneboy/SLM-SQL-1.5B
   huggingface-cli download cycloneboy/SLM-SQL-0.5B
   
   # Merger models
   huggingface-cli download cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct
   huggingface-cli download cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct
   ```

2. **Download Datasets**
   ```bash
   huggingface-cli download cycloneboy/bird_train
   ```

3. **Run Baseline Inference**
   - Single model, greedy decoding
   - Expected: ~67% for 1.5B, ~57% for 0.5B

4. **Verify Against Published Numbers**
   - If within 1-2%, proceed
   - If not, debug prompt format

### Deliverables
- [ ] Models downloaded and loadable
- [ ] Baseline accuracy verified
- [ ] Inference script working

---

## Phase 2: Self-Consistency Pipeline (Days 4-7)

### Objective
Implement the corrective self-consistency approach

### Tasks

1. **Implement Multi-Sample Generation**
   - Generate n=10 SQL candidates per question
   - Temperature=0.7, top_p=0.95

2. **Implement Execution Validation**
   - Run each candidate against SQLite
   - Filter to valid (executable) queries

3. **Implement Voting Mechanism**
   - Hash execution results
   - Select most common result
   - Fallback to first valid if tie

4. **Test on BIRD Dev Subset**
   - Run on 100 examples first
   - Verify boost from self-consistency

### Expected Results
- +3-5% over single model baseline
- 0.5B: 57% → 60%
- 1.5B: 67% → 70%

### Deliverables
- [ ] Multi-sample generation working
- [ ] Execution validation working
- [ ] Voting mechanism working
- [ ] Accuracy boost verified

---

## Phase 3: Merge Revision (Days 8-11)

### Objective
Add merge model for error correction

### Tasks

1. **Implement Merge Model Integration**
   - When voting is ambiguous, use merger
   - Input: candidates + their results
   - Output: refined SQL

2. **Handle Edge Cases**
   - All candidates fail execution
   - Voter returns wrong answer
   - Merge model disagrees with voter

3. **Benchmark Full Pipeline**
   - Run on full BIRD dev set
   - Compare: single vs voting vs merge

### Expected Results
- Merge adds +1-2% over voting alone
- Handle more complex queries correctly

### Deliverables
- [ ] Merge model integrated
- [ ] Edge cases handled
- [ ] Full benchmark completed

---

## Phase 4: Optimization (Days 12-14)

### Objective
Optimize for deployment on laptops

### Tasks

1. **Quantization**
   - INT8 quantization (minimal accuracy loss)
   - INT4 quantization (more aggressive)
   - Benchmark both

2. **Model Loading Optimization**
   - Sequential loading (generator → merger)
   - Reduce peak memory usage

3. **Batch Processing**
   - Group similar-length queries
   - Optimize GPU utilization

4. **Speed Benchmarks**
   - Measure latency per query
   - Target: <5 seconds per query

### Deliverables
- [ ] Quantized models ready
- [ ] Memory optimized
- [ ] Speed benchmarks documented

---

## Phase 5: Packaging & Demo (Days 15-18)

### Objective
Create a compelling demo

### Tasks

1. **Create CLI Tool**
   ```bash
   tinysql --db ./my_database.db "What is the total sales?"
   ```

2. **Create Web Demo (Optional)**
   - Gradio or Streamlit interface
   - Upload database, ask questions

3. **Create Documentation**
   - Installation guide
   - Usage examples
   - Benchmark results

4. **Prepare "Wow Factor" Demo**
   - Run on a MacBook Air
   - Show speed and accuracy
   - Compare to GPT-4 API cost

### Deliverables
- [ ] CLI tool working
- [ ] Documentation complete
- [ ] Demo video/screenshots

---

## Resource Requirements

### Hardware (Minimum)
- GPU: RTX 3060 (12GB) or M1/M2 MacBook
- RAM: 16GB
- Storage: 20GB for models

### Hardware (Recommended)
- GPU: RTX 4090 (24GB)
- RAM: 32GB
- Storage: 50GB

### Time
- 2-3 weeks for implementation
- 1 week buffer for debugging

### Cost (Cloud Alternative)
- RunPod A100: ~$2/hour
- Total: ~$50-100 for development

---

## Success Metrics

### Must Have
- [ ] 60%+ accuracy on BIRD dev
- [ ] Runs on laptop with 8GB VRAM
- [ ] <10 seconds per query

### Nice to Have
- [ ] 65%+ accuracy
- [ ] Runs on MacBook M1/M2
- [ ] <5 seconds per query

### Wow Factor
- [ ] 67%+ accuracy (beats many 7B+ models)
- [ ] Runs on CPU-only (slower but possible)
- [ ] Real-time demo working

---

## Risk Mitigation

### Risk: Can't reproduce SLM-SQL results
**Mitigation**: 
- Use their exact prompt format
- Check tokenizer settings
- Contact authors if needed

### Risk: Self-consistency doesn't help
**Mitigation**:
- Verify sampling diversity
- Adjust temperature
- Try different n values

### Risk: Too slow on laptop
**Mitigation**:
- Use INT4 quantization
- Reduce n for self-consistency
- Use 0.5B model instead of 1.5B

### Risk: Merge model hurts accuracy
**Mitigation**:
- Make merge optional
- Only use when voting is ambiguous
- Tune threshold

---

## Comparison with Our Current Approach

| Metric | Current (7B LoRA) | Small Model Path |
|--------|-------------------|------------------|
| Parameters | 7B | 2-3B total |
| BIRD Dev Accuracy | 43.74% | 67%+ (expected) |
| Hardware Required | A100/H100 | Laptop GPU |
| Training Time | Hours | Pre-trained (0) |
| Inference Speed | ~5 sec | ~3-5 sec |
| Deployment | Server | Edge/Laptop |
| "Wow Factor" | Low | High |

**Clear winner: Small Model Path**

---

## Next Action Items

1. **Today**: Download SLM-SQL-1.5B model
2. **This week**: Verify baseline, implement self-consistency
3. **Next week**: Add merge model, optimize
4. **Week 3**: Demo and documentation

---

## Files in This Directory

- `SMALL_MODEL_STRATEGY.md` - High-level strategy
- `IMPLEMENTATION_ROADMAP.md` - This file (detailed plan)
- `tinysql_pipeline.py` - Code outline
- `benchmark_results.md` - (To be created) Actual results
- `demo/` - (To be created) Demo materials

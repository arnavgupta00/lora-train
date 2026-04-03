# BIRD Benchmark Deep Analysis

## What is BIRD?

**BIRD** (BIg Benchmark for Large-Scale Relational Database Grounded Text-to-SQL) is a benchmark introduced in NeurIPS 2023 that evaluates text-to-SQL models on real-world scenarios.

### Key Characteristics
- **12,751 text-to-SQL pairs** (train + dev)
- **95 databases** across **37 professional domains**
- **33.4 GB total database size**
- Focus on **database values**, not just schema
- Introduces challenges: dirty data, external knowledge, SQL efficiency

### Evaluation Metrics
1. **Execution Accuracy (EX)**: Primary metric - does predicted SQL return same results as gold SQL?
2. **Valid Efficiency Score (VES)**: Measures SQL execution time efficiency
3. **Exact Match**: Secondary - does predicted SQL exactly match gold SQL? (rarely achieved)

### Dev Set Split (1,534 examples)
- 11 databases
- Difficulty distribution: Simple (56.1%), Moderate (28.9%), Challenging (15.1%)

---

## Why T8 Score Decreased (44.26% → 43.74%)

### Root Cause Analysis

#### 1. **Distribution Shift from Upsampling**

| Pattern | BIRD % | T7 % | T8 % | T7→BIRD Gap | T8→BIRD Gap |
|---------|--------|------|------|-------------|-------------|
| **JOIN** | **74.3%** | 45.9% | 48.6% | **-28.4** | **-25.7** |
| ORDER BY | 24.3% | 54.8% | 47.1% | +30.5 | +22.8 |
| DISTINCT | 22.3% | 12.2% | 10.9% | -10.1 | -11.4 |
| LIMIT | 18.5% | 43.7% | 32.5% | +25.2 | +13.9 |
| SUBQUERY | 15.2% | 4.7% | 9.1% | -10.5 | -6.1 |
| CASE | 13.3% | 5.1% | 23.3% | -8.2 | **+10.0** ⚠️ |

**Key Finding**: T8 over-indexed on CASE statements (23.3% vs BIRD's 13.3%) while still under-representing JOINs (48.6% vs 74.3%).

#### 2. **LIMIT/ORDER Pattern Dilution**

T8 drastically reduced LIMIT patterns from 43.7% → 32.5%. This diluted the model's ability to handle:
- "Find the top N..." queries
- "List the highest/lowest..." queries
- Pagination patterns

#### 3. **Per-Database Impact**

| Database | T7→T8 Change | Root Cause |
|----------|-------------|------------|
| european_football_2 | **-6.2%** | Complex JOINs, less training coverage |
| debit_card_specializing | **-4.7%** | Pattern distribution mismatch |
| california_schools | **-2.2%** | Column hallucination persists |
| superhero | **+4.7%** | Benefited from CTE upsampling |
| student_club | **+2.5%** | Simple patterns matched better |

#### 4. **Exact Match Degradation**

T7: 190 exact matches → T8: 172 exact matches (**-18**)

The model is producing more "creative" SQL that may work but doesn't match gold SQL structure.

---

## BIRD Leaderboard Analysis (as of April 2026)

### Top Performers

| Rank | Method | Dev % | Test % | Model Size | Key Technique |
|------|--------|-------|--------|------------|---------------|
| 1 | AskData + GPT-4o | - | 81.95% | UNK | Multi-agent |
| 2 | Agentar-Scale-SQL | 74.90% | 81.67% | UNK | Agent scaling |
| 3 | CHASE-SQL + Gemini | 74.90% | 76.02% | UNK | Self-consistency + Self-correction |
| 5 | **CSC-SQL + Qwen 7B** | **69.19%** | **71.72%** | **7B** | GRPO + Self-consistency |
| 8 | SLM-SQL + Qwen 1.5B | 67.08% | 70.49% | **1.5B** | SFT + RL |
| 12 | CodeS-15B | 58.47% | 60.37% | 15B | Incremental pre-training |
| - | GPT-4 Baseline | 46.35% | 54.89% | UNK | Zero-shot |
| - | **Our Model (T8)** | **43.74%** | - | **7B** | SFT only |

### Gap Analysis

| Metric | Our Model | Target (CSC-SQL 7B) | Gap |
|--------|-----------|---------------------|-----|
| Dev Accuracy | 43.74% | 69.19% | **-25.45%** |
| Test Accuracy | ~45% (est.) | 71.72% | **~-27%** |

---

## What Top Methods Do Differently

### 1. **CSC-SQL (Qwen 7B achieving 71.72%)**

From [arXiv:2505.13271](https://arxiv.org/abs/2505.13271):

**Key Innovations:**
1. **Self-Consistency Sampling**: Generate multiple SQL candidates in parallel
2. **Merge Revision Model**: Feed top-2 candidates to a revision model for correction
3. **GRPO (Group Relative Policy Optimization)**: RL fine-tuning on both generation and revision
4. **Inference-Time Scaling**: Uses more compute at inference

**Why it works:**
- Execution-based rewards (not just text matching)
- Model learns to self-correct syntax errors
- Voting eliminates outlier errors

### 2. **CodeS (15B achieving 60%)**

From [arXiv:2402.16347](https://arxiv.org/abs/2402.16347):

**Key Innovations:**
1. **Incremental Pre-training**: Pre-trained on SQL-centric corpus before fine-tuning
2. **Schema Item Classifier**: Separate model filters relevant tables/columns
3. **Bi-directional Data Augmentation**: SQL→NL and NL→SQL augmentation
4. **BM25 for Examples**: Retrieves similar examples during inference

### 3. **CHASE-SQL (Gemini achieving 76%)**

From [arXiv:2410.01943](https://arxiv.org/abs/2410.01943):

**Key Innovations:**
1. **Divide-and-Conquer**: Decompose complex queries into sub-queries
2. **Execution Plan CoT**: Chain-of-thought based on database execution plans
3. **Instance-Aware Few-Shot**: Generate synthetic examples specific to test question
4. **Pairwise Selection Agent**: Fine-tuned model to rank candidates

---

## What Our Model Is Missing

### Critical Gaps

| Gap | Impact | Solution |
|-----|--------|----------|
| **No inference scaling** | -10-15% | Multi-sampling + voting |
| **No self-correction** | -5-8% | Add revision model |
| **No RL training** | -5-10% | GRPO with execution rewards |
| **Poor JOIN coverage** | -3-5% | Rebalance training data |
| **Column hallucination** | -2-3% | Schema-aware pre-training |

### Training Data Issues

1. **JOIN under-representation**: 74.3% in BIRD vs 48.6% in T8
2. **CASE over-representation**: 13.3% in BIRD vs 23.3% in T8
3. **Insufficient backtick columns**: Only 5% of training data has complex columns
4. **No domain-specific examples**: financial, california_schools databases need targeted data

---

## Recommendations for V5

### Quick Wins (Expected: +5-10%)

1. **Rebalance Training Data**
   - Increase JOIN examples to 70%+
   - Reduce CASE to ~15%
   - Keep LIMIT/ORDER at current T7 levels

2. **Add More Backtick Column Examples**
   - Need 500+ california_schools-style examples
   - Currently only 80 → increase to 2-3% of dataset

3. **Train for 3-4 epochs** (currently 2)
   - T7 was 1 epoch, T8 was 2
   - Optimal is usually 3-4 for this data size

### Medium-Term (Expected: +10-15%)

4. **Implement Self-Consistency**
   - Generate 5-10 SQL candidates
   - Execute each and vote on results
   - Proven to add 5-10% accuracy

5. **Add Execution-Based Validation**
   - Reject SQL that errors during inference
   - Retry with schema reminder

### Long-Term (Expected: +15-25%)

6. **GRPO Training**
   - Implement execution-based rewards
   - Train revision model for error correction

7. **Schema Pre-training**
   - Pre-train on schema understanding task
   - Then fine-tune on SQL generation

---

## Next Steps

1. Create T9 dataset with corrected pattern distribution
2. Implement multi-sampling at inference
3. Add execution validation loop
4. Research GRPO implementation

---

## References

1. BIRD Paper: [arXiv:2305.03111](https://arxiv.org/abs/2305.03111)
2. CSC-SQL: [arXiv:2505.13271](https://arxiv.org/abs/2505.13271)
3. SLM-SQL: [arXiv:2507.22478](https://arxiv.org/abs/2507.22478)
4. CodeS: [arXiv:2402.16347](https://arxiv.org/abs/2402.16347)
5. CHASE-SQL: [arXiv:2410.01943](https://arxiv.org/abs/2410.01943)
6. BIRD Leaderboard: [bird-bench.github.io](https://bird-bench.github.io/)

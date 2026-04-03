# How SOTA Methods Achieve 70%+ on BIRD

## Overview

The top BIRD leaderboard methods share common techniques that our model lacks.

---

## 1. CSC-SQL (Qwen 7B → 71.72%)

**Paper**: [arXiv:2505.13271](https://arxiv.org/abs/2505.13271)  
**Code**: [github.com/CycloneBoy/csc_sql](https://github.com/CycloneBoy/csc_sql)

### Key Technique: Corrective Self-Consistency

```
Step 1: Parallel Sampling
        Generate N SQL candidates (typically 10-20)
        
Step 2: Self-Consistency Voting
        Execute each SQL, group by result sets
        Select top-2 most frequent outputs
        
Step 3: Merge Revision
        Feed both SQLs to a revision model
        Output: corrected SQL that combines best of both
```

### Why It Works
- **Voting eliminates outliers**: Random errors get outvoted
- **Execution-based validation**: Only SQLs that run successfully are considered
- **Revision fixes edge cases**: Merge model learns to fix common mistakes

### Training Details
1. **SFT Phase**: Fine-tune on standard text-to-SQL data
2. **RL Phase**: GRPO (Group Relative Policy Optimization)
   - Reward = +1 if execution matches gold result
   - Reward = 0 otherwise
   - Train both generator AND revision model

### Ablation Results (from paper)
| Method | Dev Accuracy |
|--------|-------------|
| SFT only | 62.1% |
| + Self-Consistency | 65.8% (+3.7%) |
| + Merge Revision | 67.4% (+1.6%) |
| + GRPO | 69.2% (+1.8%) |

**Total gain from inference techniques: +7.1%**

---

## 2. CodeS-15B (→ 60.37%)

**Paper**: [arXiv:2402.16347](https://arxiv.org/abs/2402.16347)  
**Code**: [github.com/RUCKBReasoning/codes](https://github.com/RUCKBReasoning/codes)

### Key Technique: Incremental Pre-training

```
Phase 1: Start with StarCoder base model
         
Phase 2: Incremental pre-training on SQL corpus
         - GitHub SQL files
         - StackOverflow SQL Q&A
         - SQL tutorials and documentation
         
Phase 3: Fine-tune on text-to-SQL datasets
         - Spider + BIRD + custom data
```

### Why It Works
- Model develops SQL "intuition" during pre-training
- Better understanding of SQL syntax and patterns
- Improved schema comprehension

### Additional Techniques
1. **Schema Item Classifier**: Separate 3B model filters relevant columns
2. **BM25 Index**: Retrieves similar database values at inference
3. **Bi-directional Augmentation**: Generate SQL→NL and NL→SQL pairs

---

## 3. CHASE-SQL (Gemini → 76.02%)

**Paper**: [arXiv:2410.01943](https://arxiv.org/abs/2410.01943)

### Key Technique: Multi-Agent Generation

```
Agent 1: Divide-and-Conquer
         Break complex queries into sub-queries
         Each sub-query is easier to solve
         
Agent 2: Execution Plan CoT
         Think step-by-step like a database engine
         "First, filter table A where X"
         "Then, join with table B on Y"
         
Agent 3: Instance-Aware Few-Shot
         Generate synthetic examples similar to test question
         Use these as few-shot context
```

### Selection Agent
After generating candidates from all agents:
1. Execute each SQL
2. Use fine-tuned binary classifier to rank pairs
3. Select highest-ranked SQL

### Why It Works
- **Diversity**: Different agents generate different solutions
- **Specialization**: Each agent optimized for different query types
- **Robustness**: Selection agent chooses best from diverse candidates

---

## 4. SLM-SQL (Qwen 1.5B → 70.49%)

**Paper**: [arXiv:2507.22478](https://arxiv.org/abs/2507.22478)

### Key Insight: Small Models Can Compete

With 1.5B parameters, achieves near SOTA via:
1. **High-quality training data**: SynSQL-2.5M (2.5 million examples)
2. **Think-style prompting**: Train model to reason before SQL
3. **Merge revision**: Same as CSC-SQL
4. **GRPO training**: Execution-based rewards

### Data Details
- **SynSQL-Think-916K**: 916K examples with reasoning traces
- **SynSQL-Merge-Think-310K**: 310K merge revision examples

**Key finding**: Data quality + RL > model size

---

## Techniques We Can Implement

### Tier 1: Easy (Next Week)

| Technique | Expected Gain | Effort |
|-----------|---------------|--------|
| **Multi-sampling** | +3-5% | Low |
| **Execution validation** | +1-2% | Low |
| **Better data distribution** | +2-3% | Medium |

### Tier 2: Medium (Month)

| Technique | Expected Gain | Effort |
|-----------|---------------|--------|
| **Self-consistency voting** | +3-4% | Medium |
| **Merge revision model** | +1-2% | Medium |
| **Schema filtering** | +2-3% | Medium |

### Tier 3: Hard (Quarter)

| Technique | Expected Gain | Effort |
|-----------|---------------|--------|
| **GRPO training** | +3-5% | High |
| **Incremental pre-training** | +5-8% | High |
| **Multi-agent framework** | +5-7% | High |

---

## Implementation Priorities

### Priority 1: Multi-Sampling + Voting

```python
def generate_with_voting(prompt, n_samples=10):
    # Generate multiple candidates
    candidates = []
    for _ in range(n_samples):
        sql = model.generate(prompt, temperature=0.7)
        candidates.append(sql)
    
    # Execute each and group by results
    result_groups = defaultdict(list)
    for sql in candidates:
        try:
            result = execute_sql(sql, database)
            result_hash = hash(str(result))
            result_groups[result_hash].append(sql)
        except:
            continue
    
    # Return SQL from largest group
    best_group = max(result_groups.values(), key=len)
    return best_group[0]
```

**Expected gain**: +3-5% accuracy

### Priority 2: Fix Training Distribution

Create T9 dataset with:
- JOIN: 70% (up from 48%)
- CASE: 15% (down from 23%)
- LIMIT/ORDER: Keep T7 levels
- Backtick examples: 500+ (up from 80)

**Expected gain**: +2-3% accuracy

### Priority 3: Execution Retry Loop

```python
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        sql = model.generate(prompt)
        try:
            execute_sql(sql, database)
            return sql  # Success
        except Exception as e:
            # Add error to prompt
            prompt += f"\n\nPrevious attempt failed: {e}\nTry again:"
    
    return sql  # Return last attempt
```

**Expected gain**: +1-2% accuracy

---

## Realistic Target

With all Tier 1 + Tier 2 techniques:
- Current: 43.74%
- + Multi-sampling: 47%
- + Better distribution: 50%
- + Self-consistency: 53%
- + Merge revision: 55%

**Realistic target: 55-58%** (matching GPT-4 baseline)

To reach 70%+, need:
- GRPO training
- Larger compute budget for inference
- Possibly larger model or pre-training

# Expert Quality vs Teacher Model Size: Research Digest

## Hypothesis

A 70B teacher (10x student size) produces meaningfully better LoRA experts than
an 8B teacher (1x student size), and a mixed strategy (8B for simple domains,
70B for complex domains) matches uniform 70B quality at lower cost.

## What This Model Is

This experiment compares the quality of LoRA experts distilled from two teacher
models of different sizes into the same student (Qwen2.5-7B):

1. **Llama 3.1 8B** ($0.02/expert via Groq) -- barely larger than the 7B student
2. **Llama 3.3 70B** ($0.19/expert via Groq) -- proper 10x distillation ratio

Key question: does teacher capacity matter when the student adapter has fixed
capacity (rank-16 LoRA, ~22.5M params)? If the adapter is the bottleneck, not
the teacher, then cheaper teachers are sufficient for some or all domains.

## Lineage in the Arena

```
exp_distillation_pilot_50 (50 experts, 70B teacher)
    |
    v
exp_distillation_quality_vs_teacher (this experiment)
    |
    v
exp_scale_500_experts (future, informed by optimal teacher strategy)
```

## Key References

- Hinton et al. (2015) -- Knowledge Distillation: Distilling the Knowledge in a Neural Network
- Mukherjee & Awadallah (2020) -- Orca: Progressive Learning from Complex Explanation Traces (1st paper)
- Mukherjee et al. (2023) -- Orca 2: Teaching Small Language Models How to Reason
- West-of-N filtering (Pace et al., 2024) -- Best-of-N distillation for quality filtering
- Xu et al. (2024) -- "A Survey on Knowledge Distillation of LLMs" (comprehensive survey)
- Li et al. (2024) -- "Smaller Models, Bigger Insights" -- smaller teachers sometimes better for distillation
- SOLE proven findings (this project) -- answer-conditioned PPL metric (r=0.811)

## Experimental Design

### Domain Selection (10 from pilot50)

| Category | Domain | Complexity | Rationale |
|----------|--------|------------|-----------|
| Code/Factual | python | Low | Structured output, patterns |
| Code/Factual | sql | Low | Deterministic, rule-based |
| Code/Factual | bash | Low | Pattern-based scripting |
| Code/Factual | physics | Medium | Formula-based, some reasoning |
| Code/Factual | accounting | Low | Procedural, rule-based |
| Reasoning/Nuanced | ethics | High | Multiple perspectives, nuance |
| Reasoning/Nuanced | creative-fiction | High | Style, creativity, narrative |
| Reasoning/Nuanced | causal-reasoning | High | Multi-step logical chains |
| Reasoning/Nuanced | legal | High | Complex argumentation |
| Reasoning/Nuanced | game-theory | High | Strategic + mathematical |

### Configuration (identical for both conditions)

| Parameter | Value |
|-----------|-------|
| Student model | Qwen/Qwen2.5-7B |
| Quantization | 4-bit NF4 (double quant) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q/k/v/o/gate/up/down (all-modules) |
| Training steps | 300 |
| Batch size | 8 (1 * 8 grad accum) |
| Learning rate | 2e-4 |
| Max seq length | 1024 |
| Packing | enabled |
| Optimizer | adamw_8bit |
| Seed | 42 |

### Evaluation Metric

Answer-conditioned PPL (proven at r=0.811, see exp_answer_conditioned_scoring):
- Use 70B-generated data as gold standard test set (last 50 of 1000 examples)
- Measure NTP loss on answer tokens only
- Compare: base, 8B-teacher adapter, 70B-teacher adapter

### Cost Breakdown

| Component | 8B Teacher | 70B Teacher |
|-----------|-----------|-------------|
| Data generation (1000 ex) | $0.02 | $0.19 |
| QLoRA training (300 steps) | $0.04 | $0.04 |
| **Total per expert** | **$0.06** | **$0.23** |
| **10 experts** | **$0.60** | **$2.30** |

## Empirical Results

[PENDING -- GPU queue tasks submitted, waiting for completion]

### Per-Domain Comparison

| Domain | Category | Base PPL | 70B PPL | 8B PPL | 70B Imp% | 8B Imp% | Gap% |
|--------|----------|----------|---------|--------|----------|---------|------|
| python | code/fact | | | | | | |
| sql | code/fact | | | | | | |
| bash | code/fact | | | | | | |
| physics | code/fact | | | | | | |
| accounting | code/fact | | | | | | |
| ethics | reason | | | | | | |
| creative-fiction | reason | | | | | | |
| causal-reasoning | reason | | | | | | |
| legal | reason | | | | | | |
| game-theory | reason | | | | | | |

### Category Averages

| Category | Avg 70B Imp% | Avg 8B Imp% | Avg Gap% |
|----------|-------------|------------|----------|
| Code/Factual (N=5) | | | |
| Reasoning/Nuanced (N=5) | | | |
| Overall (N=10) | | | |

### Mixed Strategy Analysis

| Strategy | Avg PPL | Cost (10 experts) | PPL/$ Efficiency |
|----------|---------|-------------------|-----------------|
| Uniform 70B | | $2.30 | |
| Uniform 8B | | $0.60 | |
| Mixed (8B code/fact + 70B reason) | | $1.05 | |

## Kill Criteria Assessment

### K1: 70B produces <5% better experts than 8B

**Threshold:** Average |gap%| < 5%
**Result:** [PENDING]

### K2: Mixed strategy does not beat uniform 70B

**Threshold:** Mixed avg PPL > Uniform 70B avg PPL
**Result:** [PENDING]

## Micro-Scale Limitations

- 10 of 50 domains (sufficient for category-level analysis, not exhaustive)
- Single seed (no confidence intervals)
- Test set is from 70B teacher data (potentially biases toward 70B adapter)
- Answer-conditioned PPL only (no downstream task accuracy like MMLU/HumanEval)
- Short training (300 steps) -- longer training might change the gap
- Same random seed for both conditions (controls for training noise but limits generalization)

## What Would Kill This

**Micro scale:**
- K1: If 70B-8B gap < 5% across ALL domain categories, teacher size is irrelevant
  and we should use 8B uniformly (massive cost savings)
- K2: If mixed strategy PPL > uniform 70B PPL, domain-specific teacher selection
  is not viable and we must use 70B everywhere

**Macro scale (future validation):**
- If downstream task accuracy (MMLU subset, HumanEval) does not correlate with
  answer-conditioned PPL improvements
- If the gap reverses at longer training (>1000 steps) -- adapter might eventually
  absorb the richer 70B signal given more time
- If 8B teacher produces systematically incorrect training data that the adapter
  learns as "correct" (silent quality degradation)

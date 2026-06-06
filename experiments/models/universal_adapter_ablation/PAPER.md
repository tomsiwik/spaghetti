# Universal Adapter Ablation: Is Routing Even Needed?

## Theorem (from MATH.md)

If adapter k has quality q_k(d) >= alpha * max_i(q_i(d)) for all domains d with
alpha close to 1, then routing provides at most (1-alpha) relative improvement.
When alpha > 1 (the universal adapter is BETTER than domain-specific), routing
actively hurts.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: Code >= 80% of domain on code, math | Code BEATS domain on math (alpha=1.20), matches on code (alpha=1.0) | YES (stronger than predicted) |
| P2a: Code >= 80% on prose if SFT teaches format | Alpha: medical=0.97, legal=1.25, finance=1.11 | YES -- H2a confirmed |
| P2b: Code < 50% on prose if domain knowledge needed | Refuted: alpha > 0.97 on ALL domains | REFUTED |
| P3: If H2a, routing value V < 0.1 | V = -8.8% (routing HURTS) | YES (stronger) |

## Hypothesis

Single code SFT adapter matches or outperforms domain-specific routing on all
5 domains. Routing is unnecessary overhead for these SFT adapters.

**Verdict: SUPPORTED (strongly)**

## What This Experiment Is

A controlled ablation comparing 4 configurations on 50 test queries (10 per domain)
using execution-based evaluation:

1. **Base model** (no adapter)
2. **Code adapter on ALL domains** (universal adapter hypothesis)
3. **Domain-specific adapter per domain** (oracle routing -- best possible)
4. **TF-IDF routed composition** (practical routing at 86% accuracy)

Uses pre-trained SFT adapters from exp_bitnet_sft_generation_v3 on BitNet-2B-4T.

## Key References

- Finding #204: Code SFT adapter is universal improver
- Finding #205: Energy gap routing collapses to code adapter selection
- Finding #203: Routing errors cost only ~13% at PPL level
- Finding #207: TF-IDF routing achieves 90% accuracy but prose domains still degrade
- Shazeer et al. 2017: MoE gating adds value only when experts specialize
- DES-MoE: 43-76% drops from wrong routing (contrast: we see 0% or negative)

## Empirical Results

### Summary Table

| Domain | Base | Code Adapter | Domain Adapter | Routed | Alpha | Winner |
|--------|------|-------------|----------------|--------|-------|--------|
| medical | 0.601 | 0.581 | 0.597 | 0.597 | 0.97 | domain (by 0.016) |
| code | 0.571 | 0.710 | 0.710 | 0.710 | 1.00 | tie |
| math | 0.378 | 0.773 | 0.642 | 0.642 | 1.20 | **code (+20%)** |
| legal | 0.423 | 0.373 | 0.298 | 0.285 | 1.25 | **code (+25%)** |
| finance | 0.369 | 0.352 | 0.316 | 0.315 | 1.11 | **code (+11%)** |
| **TOTAL** | **2.341** | **2.787** | **2.562** | **2.548** | | **code** |

### Key Findings

**1. Code adapter is the universal best adapter (4/5 domains).**
Only on medical does the domain-specific adapter beat the code adapter, and only by
0.016 (not statistically significant at n=10). On math, legal, and finance, the code
adapter outperforms the domain-specific adapter by 11-25%.

**2. Routing HURTS (-8.8% vs code alone).**
The value of routing is negative: V = 1 - 2.787/2.562 = -8.8%. This means
domain-specific adapters are collectively WORSE than the code adapter alone.
Routing to them degrades total quality.

**3. Domain-specific adapters degrade prose domains vs base.**
Legal adapter: 0.298 vs base 0.423 (-30%).
Finance adapter: 0.316 vs base 0.369 (-14%).
The domain-specific SFT adapters for legal and finance are worse than NO adapter.

**4. Code adapter improves structured domains massively.**
Math: 0.378 -> 0.773 (+105%, answer correctness 30%->80%).
Code: 0.571 -> 0.710 (+24%, syntax pass 50%->80%).

**5. All adapters reduce response quality on prose domains.**
Base model has high response_quality (0.77-0.78 on prose) but low factual overlap.
All adapters reduce response_quality while only slightly affecting factual overlap.
This suggests SFT adapters interfere with the base model's general prose generation
capability without adding enough domain knowledge to compensate.

### Execution-Based Metric Breakdown

| Domain | Metric | Base | Code | Domain | Routed |
|--------|--------|------|------|--------|--------|
| code | syntax_valid | 0.50 | **0.80** | 0.80 | 0.80 |
| code | response_quality | **0.74** | 0.50 | 0.50 | 0.50 |
| math | answer_correct | 0.30 | **0.80** | 0.70 | 0.70 |
| math | response_quality | **0.69** | 0.66 | 0.41 | 0.41 |
| medical | factual_overlap | 0.49 | 0.48 | **0.54** | 0.54 |
| medical | response_quality | **0.77** | 0.73 | 0.69 | 0.69 |
| legal | factual_overlap | **0.19** | 0.14 | 0.10 | 0.08 |
| legal | response_quality | **0.77** | 0.71 | 0.59 | 0.59 |
| finance | factual_overlap | 0.10 | **0.10** | 0.07 | 0.08 |
| finance | response_quality | **0.78** | 0.73 | 0.69 | 0.66 |

### Routing Accuracy

TF-IDF routing: 86% overall (43/50 correct).
- medical: 10/10, code: 10/10, math: 10/10, legal: 7/10, finance: 6/10
- Errors: legal misclassified as finance (3), finance misclassified as legal/math (4)

## Kill Criteria

| Kill Criterion | Threshold | Measured | Result |
|---------------|-----------|----------|--------|
| K608: Code >= 50% of routed total | 0.50 | 1.094 (109.4%) | **PASS** |
| K609: 2+ domains where domain > code | 2 | 1 (medical only) | **FAIL** |
| K610: Execution-based metrics | yes | syntax, correctness, factual overlap | **PASS** |

**K609 FAIL means: domain-specific adapters do NOT provide enough specialization
to justify routing.** The code adapter is nearly universal.

## Interpretation

### Why does the code adapter generalize?

Code SFT training teaches:
1. **Instruction following** -- how to parse "### Instruction:" and produce structured output
2. **Sequential reasoning** -- step-by-step problem solving (common in code and math)
3. **Formatting** -- clean, structured responses

These are general skills that transfer across domains. The code training data
(being the most structured) produces the strongest instruction-following capability.

### Why do domain-specific adapters HURT prose domains?

The legal and finance SFT adapters are trained on domain text that:
1. Is longer and more complex (legal: 2.84 SFT val loss vs code: 1.28)
2. Teaches domain-specific patterns that reduce general response quality
3. Has high loss even after training (suggesting underfitting on hard data)

The adapters learn to produce more domain-specific but lower-quality responses.
At lora_scale=20, this overwrites the base model's general prose capability.

### What does this mean for the composition thesis?

**The composition thesis is NOT killed** -- it needs refinement:

1. **Current SFT adapters lack sufficient specialization.** They primarily teach
   format compliance, not domain expertise. Better training (more data, better
   data selection, curriculum learning) could create adapters that genuinely
   specialize.

2. **Routing is premature for under-specialized adapters.** When all adapters
   provide similar (or worse) quality, routing cannot help. Routing becomes
   valuable only when adapters are genuinely specialized.

3. **The code adapter's universality is an artifact of training quality.**
   Code data is well-structured and produces strong instruction-following.
   If other domain adapters were trained equally well, routing might matter.

## Limitations

1. **n=10 per domain.** Small sample; differences < 20pp may not be significant.
2. **Factual overlap metric is approximate.** Entity extraction is not perfect.
   LLM-judge evaluation would be more reliable for prose domains.
3. **Single lora_scale=20.** Not ablated; different scales may change results.
4. **Single model (BitNet-2B-4T).** Results may differ on larger models with
   more domain knowledge in the base weights.
5. **SFT adapters trained for only 300 steps.** Longer training may improve
   domain-specific adapters more than the code adapter.
6. **Factual overlap scores are low across all configs** (0.05-0.54), suggesting
   the metric may be noisy or the model lacks domain knowledge.

## What Would Kill This

1. **Better domain-specific adapters** that achieve alpha < 0.5 on 2+ domains
   would prove routing is needed. Likely requires: more training data, longer
   training, better data selection, or larger base models.
2. **LLM-judge evaluation** showing domain-specific adapters produce more
   factually accurate responses despite lower entity overlap scores.
3. **Larger scale** (7B+ models) where the base model has more domain knowledge
   that domain-specific adapters can unlock.

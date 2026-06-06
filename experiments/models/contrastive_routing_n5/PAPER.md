# Text Classification Routing for SFT Adapters: Proof Verification Report

> **Naming note:** Originally titled "contrastive routing." The actual method is
> TF-IDF + logistic regression. See MATH.md for full naming clarification.

## Design Principle

**NLL-independence (Design Principle 2).** Classifier-based routing accuracy is
independent of adapter generation quality. The classifier f: text -> {1,...,K}
operates on input text only and cannot be corrupted by one adapter having
universally lower NLL. This is a property of the function signature, not a
non-trivial mathematical guarantee.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---|---|---|
| Routing accuracy >= 90% (DP 1, domain separability) | 90% (45/50) | YES |
| Energy gap baseline ~36% | 36% (from Finding #205) | YES (baseline) |
| Per-domain medical >= 70% | 100% (10/10) | YES |
| Per-domain code >= 70% | 100% (10/10) | YES |
| Per-domain math >= 70% | 100% (10/10) | YES |
| Per-domain legal >= 70% | 80% (8/10) | YES |
| Per-domain finance >= 70% | 70% (7/10) | YES (at threshold) |
| Math correctness >= 60% | 80% (8/10) | YES |
| 0/5 prose domains degraded | 3/5 worse | NO |

## Hypothesis

Text-classification-based routing achieves >70% accuracy on 5 SFT domains where
energy gap achieves only 36%, by decoupling routing from adapter NLL.

**Verdict: SUPPORTED for routing, REFUTED for generation quality.**

## What This Model Is

A TF-IDF + logistic regression text classifier that routes input queries to
domain-matched SFT adapters. Trained on 2000 instruction texts (400/domain)
with 5000 TF-IDF features. Classification takes <1ms per query. Zero additional
GPU memory. Replaces the energy gap routing mechanism entirely.

## Key References

- van den Oord et al. (2018): InfoNCE and mutual information lower bound
- Khosla et al. (2020, arXiv:2004.11362): Supervised contrastive learning
- Zhao et al. (2024, arXiv:2402.09997): LoraRetriever - input-aware LoRA retrieval
- Finding #205: Energy gap routing fails at 36% for SFT adapters
- Finding #204: Code adapter is universal improver at 70% math correctness

## Empirical Results

### Phase 1: Classifier Training (0.5s)
- Train accuracy: 99.0% (2000 samples, 400/domain)
- Val accuracy: 93.6% (250 samples, 50/domain)
- Perfect on medical (100%), code (100%), math (98%)
- Weaker on legal (86%), finance (84%) -- these domains share vocabulary

### Phase 2: Routing Accuracy on Test Prompts
| Domain | Contrastive | Energy Gap | Improvement |
|--------|-------------|------------|-------------|
| Medical | 100% | 80% | +20pp |
| Code | 100% | 100% | 0pp |
| Math | 100% | 0% | +100pp |
| Legal | 80% | 0% | +80pp |
| Finance | 70% | 0% | +70pp |
| **Overall** | **90%** | **36%** | **+54pp** |

### Phase 3-4: Generation Quality (Base vs Correctly-Routed)
| Domain | Base Score | Routed Score | Change | Routing |
|--------|-----------|-------------|--------|---------|
| Medical | 0.4746 | 0.4078 | -14.1% | 100% correct |
| Code | 0.3569 | 0.4880 | +36.7% | 100% correct |
| Math | 0.1002 | 0.5784 | +477.6% | 100% correct |
| Legal | 0.4644 | 0.2754 | -40.7% | 80% correct |
| Finance | 0.4715 | 0.3830 | -18.8% | 70% correct |

### Behavioral Metrics
- Math: 8/10 correct answers (vs 1/10 base, vs 7/10 with code adapter)
- Code: 7/10 valid syntax (vs 5/10 base)

### Kill Criteria
| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K605: Routing accuracy | >= 70% | 90% | **PASS** |
| K606: Math correctness | >= 60% | 80% | **PASS** |
| K607: 0 domains worse | 0/5 | 3/5 | **FAIL** |

**Overall: MIXED (2/3 PASS)**

## Critical Finding: Text Classification Routing Works for 5 Well-Separated Domains

Text classification routing achieves 90% accuracy on 5 well-separated domains,
validating the principle that routing should use input features rather than adapter
NLL. The broader routing problem (fuzzy domain boundaries, N>5, production scale)
remains open.

### Decomposition of Results

1. **Input-based routing works at N=5.** TF-IDF + logistic regression achieves 90%
   accuracy (vs 36% energy gap). Input text features are a reliable domain signal for
   5 well-separated domains. This does not generalize to fuzzier boundaries or larger N.

2. **Adapter quality may be a problem for prose domains — but this is inconclusive.**
   Three domains (medical, legal, finance) show degraded keyword density scores.
   However, per Finding #179, keyword density is an unreliable metric. The 3/5
   degradation could reflect genuine adapter quality issues, metric noise, or both.
   This result is **inconclusive** and requires validation with a reliable behavioral
   metric (e.g., LLM-as-judge, execution-based eval) before drawing conclusions.

3. **Energy gap routing may have masked adapter quality problems.** By misrouting
   legal/finance to the code adapter, energy gap routing showed smaller degradation
   on those domains. Correct routing reveals larger degradation on the keyword density
   metric — but the same metric unreliability caveat applies.

4. **Correct routing + math adapter > code adapter for math.** Math correctness
   improved from 70% (code adapter via energy gap) to 80% (math adapter via
   text classification routing). This validates that domain-matched routing helps
   when the adapter is good.

### Integration with Existing Routing

VISION.md reports a softmax router that "matches oracle quality at N=24" for NTP
(next-token-prediction) adapters. This TF-IDF classifier addresses a **different
setting**: SFT (supervised fine-tuned) adapters at N=5. The softmax router operates
on hidden states and was validated with NTP adapters. This experiment validates the
principle that for SFT adapters — where NLL-based routing fails due to single-adapter
dominance — input-feature-based routing is effective. At production scale (N>5, fuzzy
boundaries), a learned embedding model (e.g., BERT-tiny) or the existing softmax
router may be more appropriate than TF-IDF.

### Implications

The K607 result (3/5 domains worse on keyword density) is **inconclusive** due to
the unreliable metric (Finding #179). It may indicate adapter quality issues, but
cannot be confirmed without a reliable behavioral evaluation.

**Next steps:**
- Validate prose domain quality with a reliable metric (LLM-as-judge or execution-based)
- Investigate whether lora_scale=20.0 is too aggressive for prose domains
- If adapter quality is confirmed as the issue, focus on training improvements

## Limitations

1. **n=10/domain test prompts.** Small sample size may not capture all routing failure modes.
2. **Keyword density metric for prose domains is unreliable.** Finding #179 showed
   this metric has weak correlation with actual quality. The K607 failure (3/5 domains
   worse) is **inconclusive** — it could be genuine adapter quality degradation, metric
   noise, or both. No conclusions about adapter quality should be drawn from this
   metric alone.
3. **Single seed.** No statistical significance testing.
4. **TF-IDF is domain-specific.** This classifier was trained on these 5 specific domains.
   Adding a 6th domain or changing domain boundaries requires retraining.
5. **Validation set may overlap with test prompts.** Both come from valid.jsonl.

## What Would Kill This

- If a domain were added where TF-IDF cannot separate it (e.g., "biomedical law" overlapping
  medical + legal), routing accuracy would drop below 70%.
- If the adapter quality issue is not solvable (all domain-matched adapters degrade prose),
  then correct routing is useless and a single universal adapter is better.
- At macro scale, domain boundaries may be fuzzier and TF-IDF may not suffice ---
  a learned embedding model (BERT-based) would be needed.

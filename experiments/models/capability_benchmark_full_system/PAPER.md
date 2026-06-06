# Capability Benchmark: Full System Verification Report

## Theorem (from MATH.md)

**Proposition 1 (Format-Capability Equivalence):** For a task T with
format_dependency f in [0,1]:

  quality(M + s*A, T) - quality(M, T) ~ c*f - delta*(1-f)

Composition helps when f~1 (format IS capability) and hurts when f~0
(knowledge bottleneck). Predicted: GSM8K +10pp, Code +15pp, NER +5pp,
MMLU -5pp to 0pp.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| GSM8K >= +10pp (chain-of-thought) | -15.0pp (30% -> 15%) | NO |
| Code gen >= +10pp (syntax) | -10.0pp (90% -> 80%) | NO |
| Clinical NER >= +5pp (entity F1) | -7.4pp (31.8% -> 24.4%) | NO |
| MMLU -5pp to 0pp (knowledge) | -5.0pp (44% -> 39%) | YES |
| Incoherence < 5% | 0.0% | YES |

**2/5 predictions match. 3/5 falsified. K1 FAIL. K2 FAIL. K3 PASS.**

## Status: KILLED

K1 (#675) FAIL: GSM8K degraded from 30% to 15% (-15pp).
K2 (#676) FAIL: Code gen degraded from 90% to 80% (-10pp).
K3 (#677) PASS: 0% incoherent output.

## Hypothesis

"Composition helps on CAPABILITY benchmarks where FORMAT=CAPABILITY."

**REFUTED.** Composition HURTS on all benchmarks tested, both capability
and knowledge.

## What This Experiment Is

A verification of the full composition system (BitNet-2B-4T + 5 SFT adapters +
oracle top-1 routing + per-domain optimal scales) on standard benchmarks:
- GSM8K (20 problems): chain-of-thought math reasoning
- Code generation (10 problems): Python syntax validity
- Clinical NER (20 examples): medical entity extraction
- MMLU (100 questions, 20 per domain): factual multiple-choice

Two conditions: base model alone vs. composed model with domain-routed adapters.

## Key References

- LIMA (Zhou et al., 2305.11206): Fine-tuning teaches format, not knowledge
- Finding #249: Two behavioral regimes (FORMAT s<=4 vs CAPABILITY s>=20)
- Finding #258: CPT is no-op, adapters encode FORMAT not KNOWLEDGE
- Finding #237: GSM8K +10pp consistent across 3 experiments

## Empirical Results

### Capability Benchmarks (format-dependent)

| Benchmark | Base | Composed | Delta |
|-----------|------|----------|-------|
| GSM8K (accuracy) | 0.300 | 0.150 | -15.0pp |
| Code gen (syntax parse) | 0.900 | 0.800 | -10.0pp |
| Clinical NER (F1) | 0.318 | 0.244 | -7.4pp |

### Knowledge Benchmark (factual recall)

| Benchmark | Base | Composed | Delta | Scale |
|-----------|------|----------|-------|-------|
| MMLU overall | 0.440 | 0.390 | -5.0pp | mixed |
| MMLU medical | 0.400 | 0.350 | -5.0pp | 20.0 |
| MMLU code | 0.400 | 0.400 | +0.0pp | 20.0 |
| MMLU math | 0.500 | 0.400 | -10.0pp | 20.0 |
| MMLU legal | 0.550 | 0.450 | -10.0pp | 4.0 |
| MMLU finance | 0.350 | 0.350 | +0.0pp | 1.0 |

### Incoherence

0% incoherent output across all domains.

## Critical Analysis: Why Did This Fail?

### The Core Problem: In-Distribution vs Out-of-Distribution

Prior experiments (Finding #249, generation_quality_perscale) showed the SFT adapters
improving behavioral metrics. But those experiments used **evaluation prompts from
the same distribution as the training data** (the validation split of the same
dataset). The adapters learned the FORMAT of those specific training examples.

This experiment tests on **out-of-distribution benchmarks**:
- GSM8K: Different distribution from our math training data
- Code generation: Same distribution (our validation set), but the base model already
  achieves 90% syntax rate, leaving little room for improvement
- Clinical NER: A novel task (entity extraction) not in the training distribution
- MMLU: Standard factual recall benchmark

### The SFT Adapters Are Narrow Format Copiers

The SFT adapters learned to reproduce the format of their training data. When
applied to prompts from different distributions:
- The format transformation INTERFERES with the base model's general capability
- At s=20, the perturbation is large enough to degrade generation quality
- The "700% improvement on math" from prior experiments was an in-distribution effect:
  the adapter learned the GSM8K-like format of the training data, and the evaluation
  used prompts from the same distribution

### Scale=20 Is Too High for Standard Benchmarks

MMLU finance (s=1.0) and MMLU code (s=20.0) show +0.0pp delta, while MMLU math
(s=20.0) and MMLU legal (s=4.0) show -10pp. Higher scales cause more interference
with the base model's stored knowledge. The finding that s=1 preserves base
performance is consistent with Finding #249's FORMAT regime.

### Reconciliation with Prior +10pp GSM8K Finding (#237)

Finding #237 reported consistent +10pp on GSM8K across 3 experiments. That finding
used the competitive_benchmark_routed experiment which used **NTP adapters** (from
real_data_domain_experts/adapters/), not the SFT adapters used here. The different
adapter training methods produce different perturbation structures.

This is strong evidence that the adapter type matters critically for benchmark
performance: NTP adapters may produce more general format improvements than SFT
adapters which overfit to their specific training distribution.

## Limitations

1. **Small sample sizes:** 20 GSM8K, 10 code, 20 NER, 100 MMLU.
   Statistical power is limited; some deltas may not be significant.
2. **SFT adapters only:** NTP adapters (which showed +10pp in prior work) were
   not tested. The failure may be specific to SFT adapter quality.
3. **Single seed:** No repeated trials with different random seeds.
4. **Clinical NER is synthetic:** The NER benchmark was constructed from medical
   prompts, not from a standard NER dataset.

## What Would Save This

1. **Test with NTP adapters:** The competitive_benchmark_routed showed +10pp on
   GSM8K with NTP adapters. Repeating this experiment with NTP adapters would
   determine if the failure is adapter-specific or fundamental.
2. **Lower scales for OOD tasks:** The FORMAT regime (s<=4) preserves base
   capability. Testing s=4 uniformly on all benchmarks would show if lower scales
   avoid the degradation.
3. **Distribution-matched evaluation:** If the adapters only help in-distribution,
   that is still valuable for a DEPLOYMENT system where routing matches queries
   to the correct domain adapter. The benchmark must match the deployment scenario.

## What Was Learned

**Finding:** SFT adapters at high scale (s=20) degrade out-of-distribution
benchmark performance across all task types (capability AND knowledge). The
prior in-distribution gains do not transfer to standard benchmarks. The
format-capability equivalence hypothesis is REFUTED: even format-dependent
tasks (GSM8K, code) degrade when the adapter's format does not match the
benchmark's expected format.

**Implication for the project:** The two-regime model (FORMAT vs CAPABILITY)
must be refined. It is not "format tasks benefit, knowledge tasks don't."
It is "in-distribution tasks benefit, out-of-distribution tasks don't,
regardless of format dependency." This means the routing system must not
only select the right domain adapter but also the right SCALE based on
how well the query matches the adapter's training distribution.

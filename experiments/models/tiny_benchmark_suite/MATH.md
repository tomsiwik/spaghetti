# Pierre Tiny Benchmark Suite: Mathematical Framework

## Experiment Type: Verification (Type 1)

This is a benchmark verification experiment, not a new mechanism experiment.
The "proof" is the accumulated mathematical framework from prior findings.

## Prior Mathematical Results Being Verified

### 1. Additive Composition Preserves Base Knowledge (Finding #320)

**Theorem (Davis-Kahan, applied):** For base model W and adapter perturbation
delta_W with ||delta_W||_2 <= epsilon, the rotation of principal knowledge subspaces
is bounded by:

  sin(theta) <= ||delta_W||_2 / spectral_gap(W)

At LoRA scale alpha <= 5 on fp16 models, epsilon is small enough that theta ~ 0,
preserving factual knowledge (MMLU). At alpha = 20, the bound becomes vacuous.

**Prediction:** MMLU with scale <= 5 should match base (0pp degradation).
MMLU with scale = 20 should catastrophically degrade (-40 to -60pp on Qwen3-4B;
effect on BitNet-2B less characterized).

### 2. NTP Training Preserves Reasoning (Finding #262)

**Observation:** NTP loss on all tokens regularizes adapter perturbation on
instruction-like inputs, preserving chain-of-thought reasoning capability.
SFT response-only masking creates format-locked adapters that disrupt OOD reasoning.

**Prediction:** NTP math adapter should improve GSM8K over base (+10pp on prior n=50).
SFT adapter at scale=20 should degrade GSM8K (-18pp on prior n=50).

### 3. DARE Sparsification (Finding #266)

**Theorem (DARE, unbiased estimator):** For adapter B, DARE with drop rate p
produces B_sparse = mask * B / (1-p) where E[B_sparse] = B. The effective
perturbation norm is reduced by sqrt(1-p), reducing OOD corruption while
preserving in-distribution signal in expectation.

**Prediction:** DARE p=0.5 composed adapters should preserve in-domain quality
while reducing OOD degradation.

### 4. Norm-Rescaled Composition (Finding #275)

**Property:** compose_adapters() uses norm-rescaled averaging:
result = mean(Bs) * mean_source_norm / ||mean(Bs)||. This prevents 1/sqrt(N)
norm shrinkage from naive averaging, preserving adapter magnitude.

**Prediction:** Composed N=5 should not shrink to zero even with 5 adapters.

## Predictions for This Experiment

Based on prior findings (all on BitNet-2B-4T unless noted):

| Benchmark | Base | NTP adapter | SFT adapter (s=20) | Composed N=5 |
|-----------|------|-------------|---------------------|---------------|
| MMLU (8Q) | ~38% | ~32-38% | ~10-20% (degraded) | ~25-35% |
| GSM8K (4Q)| ~50% | ~60% (math NTP) | ~20% (degraded) | ~40% |
| Code (4Q) | ~60% | ~50% | ~45% | ~40% |

Note: Prior finding #213 used n=50 questions; this experiment uses smaller n
for speed. Results are directional indicators, not precise estimates.

## Kill Criterion Derivation

**K820: All benchmarks below base model.** This is a binary existence test:
does ANY adapter configuration improve ANY benchmark over base? If composition
is mathematically sound (proven by Findings #320, #323), at least one
configuration should help on at least one task.

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   Additive LoRA perturbation at bounded scale preserves base model knowledge
   subspaces (Davis-Kahan bound).

2. Which existing theorem(s) does the proof build on?
   Davis-Kahan sin-theta theorem; DARE unbiased estimator (Yu et al. 2024).

3. What specific numbers does the proof predict?
   MMLU at scale<=5: 0pp degradation. NTP math GSM8K: +10pp. SFT scale=20: -18pp GSM8K.

4. What would FALSIFY the proof?
   If adapters at scale<=5 still degrade MMLU on BitNet (would indicate ternary
   spectral gap is too small for Davis-Kahan to apply at any scale).

5. How many hyperparameters does this approach add?
   0 new. Using previously validated: alpha=20 for SFT, alpha=1 for MMLU, DARE p=0.5.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is a benchmark of the existing proven system.

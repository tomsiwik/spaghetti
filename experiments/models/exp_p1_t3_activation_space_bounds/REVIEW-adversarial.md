---
reviewer: adversarial
date: 2026-04-10
verdict: PROCEED
---

# REVIEW-adversarial.md — T3.3: Activation-Space Interference Power Law

## Verdict: PROCEED

All kill criteria pass. Prediction-vs-measurement table is complete. Status SUPPORTED is appropriate
for a guided exploration that confirms a bounds result.

---

## Issues Found

### Blocking: None

### Non-Blocking (3 caveats)

**1. MATH.md alpha prediction missed by 2.5x (0.35–0.42 predicted vs 0.15 measured)**

Theorem 1 predicted alpha ≈ 0.35–0.42 (extrapolated from Finding #372 on Qwen3-4B fc1).
Actual measurement: alpha_unnorm = 0.159, alpha_vnorm = 0.145.

The PAPER.md provides a post-hoc correction (`1/sqrt(d_out) = 1/sqrt(2048) = 0.022` argument),
but this correction formula is NOT in MATH.md. The finding is still valid — all kill criteria
used appropriate thresholds (K1057: ≤ 0.40, K1058: < 0.50) that are met with large margin.

For future experiments citing this power law: use alpha=0.15, not 0.38. The MATH.md derivation
should be updated (in a separate pass) to reflect that Gemma 4's larger d_out=2048 (vs Qwen3-4B
fc1's d_out=11008) shifts the regime.

**2. Real-adapter cosine discrepancy is not derived from the theorems**

Real adapters: max_cos = 0.596 at N=5 (7.6× higher than synthetic 0.078).
The PAPER.md explanation (correlated LoRA initialization across checkpoints) is post-hoc
and not a theorem. The Frobenius cosine of A matrices (0.71-0.83 across domains) is
plausible but not measured in this experiment — it's stated as context from an earlier
investigation. This is appropriate for a SUPPORTED finding but Future T4 experiments
should validate the initialization correlation claim directly.

**3. K1056 is definitional ("measurement always has a value")**

K1056 passes by definition — any power-law fit will return c and alpha. It is not
a real falsifiable criterion. This is pre-existing (already in the DB) and non-blocking
for this review, but future kill criteria should avoid tautological checks.

---

## Strengths

- Power law measured cleanly with T=100 trials per N-value; R² ≥ 0.94 for both cases.
- V-norm claim (Theorem 2) confirmed quantitatively: delta_alpha = -0.013.
- Real-adapter observation (0.596 vs 0.078) is the most important result — it grounds
  T3.1's "routing is load-bearing" finding in an activation-space measurement.
- Corollary in MATH.md correctly distinguishes pairwise cosine from SNR-under-simultaneous-activation.
- Caveats section in PAPER.md is honest and complete.

---

## Action

No blocking fixes. Emit review.proceed → Analyst writes LEARNINGS.md.

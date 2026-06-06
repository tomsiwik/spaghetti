# LEARNINGS — E14-full: Grassmannian Activation Orthogonality

## Core Finding

Grassmannian A-matrix orthogonality provides **zero reliable activation decorrelation** at full scale. Smoke's ~33% benefit (3 layers) was upward-biased sampling noise; full run (22 layers) shows mean benefit 0.0018 — indistinguishable from zero (12 positive, 10 negative layers).

## Why

1. **Johnson-Lindenstrauss dominates**: In d=2560, random projections are already nearly orthogonal. Grassmannian's marginal gain over random max overlap (~0.32) is negligible.
2. **B₁ᵀB₂ is the true interference source**: B = W @ Aᵀ shares the base model W, making B-matrix coupling O(1) regardless of A orthogonality. This is the σ_max ≈ 40-50 that makes the bound vacuous.
3. **Smoke bias mechanism**: Layers 0, 6, 20 happened to be on the positive side of a zero-mean distribution. At N=22, the law of large numbers reveals the true center.

## Implications for Remaining Work

1. **E22-full is the final P≤2 experiment**. Its poisoning protection (F#821, 55pp margin) operates via input-space feature isolation under adversarial perturbation — a DIFFERENT mechanism than activation decorrelation. E14-full's kill does NOT threaten E22's result.
2. **Grassmannian's value is behavioral, not metric**: Zero activation decorrelation benefit (E14-full) but 55pp poisoning protection (E22-smoke). This is the clearest F#666 example yet — proxy and behavioral diverge completely.
3. **No further activation-cosine experiments**: F#815 confirmed. The A→B coupling gap is structural (shared W). Only B-matrix orthogonality regularization during training could close it.
4. **Bound tightening is a dead end**: σ_max(BᵀB) ≈ 40-50 is intrinsic to LoRA's B = W @ Aᵀ construction. No post-hoc analysis can make the spectral norm bound informative.

## Antipatterns

None. Researcher correctly identified vacuous proxy, reviewer caught independently. Smoke-to-full methodology was sound — the smoke just happened to sample from the tail.

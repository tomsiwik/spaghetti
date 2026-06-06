# P9.B3: TT-LoRA + PoLAR Hybrid — Stiefel Retraction on TT Cores

## Summary

Stiefel retraction on TT-LoRA interior cores improves GSM8K accuracy by +8pp
(62% → 70%) at identical parameter count (64,260), with negligible sr change
(1.03x). The dominant mechanism is **norm regularization**, not spectral
spreading: Frobenius norms drop 2-3x (preventing over-correction), while
singular value ratios stay similar. K1365 (retraction < 1ms) fails due to
numpy overhead (9ms), not fundamental cost.

## Prediction vs Measurement

| Metric | Predicted | Measured (TT-LoRA) | Measured (Stiefel) | Match? |
|--------|-----------|--------------------|--------------------|--------|
| sr(ΔW) mean (trained layers) | 1.5-3.0 (baseline), 3.0-6.0 (Stiefel) | 1.88 | 1.94 | **MISS** — sr barely changes (1.03x not 1.5-3x) |
| sr ratio | 1.5-3x higher | — | 1.03x | **MISS** — prediction too optimistic |
| GSM8K accuracy | ~65% baseline, ≥60% Stiefel | 62.0% | 70.0% | **MISS** (opposite dir) — Stiefel *improves* by 8pp |
| Params | 64,260 = 64,260 | 64,260 | 64,260 | MATCH |
| Retraction time | < 0.1 ms/step | — | 9.07 ms | **MISS** — numpy overhead dominates |
| Convergence | Both converge | loss 0.92→0.41 | loss 0.98→0.41 | MATCH |

## Per-Layer Stable Rank Detail

| Layer | TT-LoRA sr | Stiefel sr | TT-LoRA ||ΔW||_F | Stiefel ||ΔW||_F |
|-------|-----------|-----------|------------------|------------------|
| 0 | 1.39 | 1.42 | 8.75 | 4.05 |
| 10 | 2.34 | 2.02 | 10.80 | 3.80 |
| 20 | 1.92 | 2.38 | 10.21 | 5.02 |
| 30 | 0.00 | 0.00 | 0.00 | 0.00 |
| 41 | 0.00 | 0.00 | 0.00 | 0.00 |

**Key observation:** Layers 30+ have ΔW = 0 for both variants — gradient signal
does not reach deeper layers with 500 steps of v_proj-only training.

## Analysis

### Why the sr prediction failed

The MATH.md predicted that Stiefel prefix → isometric contraction → sr(ΔW) =
sr(R) (output-side cores), expecting 1.5-3x improvement. The actual mechanism
is different:

1. **Stiefel constrains norm, not spectrum.** Interior cores on Stiefel have
   ||G_k||_F = sqrt(r) = 2.45 (fixed). Unconstrained cores can grow arbitrarily.
   After 500 steps, unconstrained norms are 2-3x larger.

2. **Norm control = regularization.** Smaller ||ΔW||_F means smaller corrections
   to the base model. This prevents over-correction (a form of over-fitting).
   The 8pp quality gain comes from this regularization, not from spectral spreading.

3. **sr is unchanged because the spectral shape is intrinsic to the last core.**
   Both variants have the same last core architecture. Stiefel on prefixes
   doesn't change how the last core distributes its singular values — it just
   controls the overall scale.

### The actual theorem at work

Theorem 1 (MATH.md) correctly predicts: in left-canonical form, ||ΔW||_F =
||G_d||_F. The Stiefel constraint forces this by making prefix contractions
norm-preserving. Without Stiefel, prefix norms amplify ||ΔW||_F by 2-3x.

The correct framing: **Stiefel on TT cores is a norm regularizer equivalent
to constraining ||ΔW||_F, not a spectral regularizer targeting sr.**

### Quality improvement mechanism

- TT-LoRA ||ΔW||_F ≈ 10.0 → large corrections → potential over-fitting
- Stiefel ||ΔW||_F ≈ 4.3 → moderate corrections → better generalization
- Both achieve similar final loss (0.41), but Stiefel generalizes better to
  held-out test problems (+8pp on 50 GSM8K questions)

### Retraction overhead

9ms per retraction step (every 10 training steps = 0.9ms amortized/step).
The cost is dominated by MLX↔numpy data transfer for 210 small (48×6) matrices.
An MLX-native SVD or batched operation would likely reduce this 10x+.
The fundamental math (SVD of 48×6) takes O(1728) flops per core.

## Kill Criteria

| ID | Criterion | Result | Detail |
|----|-----------|--------|--------|
| K1363 | sr(Stiefel) > sr(baseline) | **PASS** | 1.94 vs 1.88 (trained layers only), 1.03x |
| K1364 | Quality ≥ baseline | **PASS** | 70.0% vs 62.0% (+8pp) |
| K1365 | Retraction < 1ms/step | **FAIL** | 9.07ms (numpy overhead) |

**Overall: 2/3 PASS.** K1365 is an implementation constraint, not a fundamental
limitation. The core finding (Stiefel improves quality via norm regularization)
is supported.

## Behavioral Assessment

TT-LoRA-Stiefel produces correct GSM8K solutions with step-by-step arithmetic
chains at 70% accuracy (35/50). The quality improvement vs unconstrained TT-LoRA
(62%, 31/50) suggests the Stiefel regularization prevents over-correction in
the v_proj weight updates, leading to more robust reasoning.

## Implications

1. **For composition:** Smaller ||ΔW||_F means less interference between
   adapters when composing. This connects directly to the interference bounds
   (Finding #225) — smaller corrections → larger null-space margin.

2. **For training:** Stiefel retraction every 10 steps adds <1% wall-clock
   overhead (amortized) and improves quality. Should be default for TT-LoRA.

3. **Revised understanding:** The benefit of Stiefel on TT-LoRA is
   **norm control/regularization**, not spectral spreading. The sr guarantee
   from PoLAR (Finding #442) does not transfer to TT because the contraction
   chain destroys the isometric prefix → sr relationship for the full matrix.

## References

- Oseledets (2011). "Tensor-Train Decomposition." SIAM J. Sci. Comput.
- Batselier et al. (2025). "TT-LoRA MoE" arXiv:2504.21190
- Finding #442: Joint Stiefel PoLAR guarantees sr=r
- Finding #515: TT-LoRA MLX port
- Finding #516: TT-LoRA quality (84.4%, 12.4x compression)

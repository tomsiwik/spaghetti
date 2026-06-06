# PAPER.md — E15: Composition Residual Decomposition

## Abstract

We test whether SVD decomposition of LoRA B matrices reveals a rank threshold separating compositional signal from cross-adapter interference, enabling filtered composition with reduced tau. The hypothesis is falsified: B matrices have near-uniform singular value spectra (top-3 carry 55% vs 50% for perfect uniformity), and rank filtering INCREASES cross-adapter coupling rather than reducing it. The failure is structural: B matrices derived from the same base weight W share output-space directions, so SVD filtering removes correlated dimensions simultaneously, providing no relative decorrelation.

## Predictions vs Measurements

| Quantity | Predicted | Measured | Match |
|---|---|---|---|
| Top-3 SV energy fraction | ≥ 75% | 55.5% | MISS (10× closer to uniform than predicted) |
| Best rank-k coupling ratio | < 0.70 at k=4 | 1.00 at k=6 (no improvement) | MISS (filtering counterproductive) |
| Rank-2 vs Rank-6 coupling | lower at k=2 | higher at k=2 (0.040 vs 0.034) | OPPOSITE DIRECTION |
| Best mean\|cos\| | < 0.030 | 0.034 (full rank is best) | FAIL |
| Norm retention | ≥ 0.50 at useful k | 1.00 (only full rank useful) | N/A (no useful k) |

## Per-Layer Analysis

### Layer 0
- SV spectrum: [16.4, 15.2, 14.6, 14.5, 14.0, 13.2] — near-flat (ratio σ₁/σ₆ = 1.24)
- Top-3 energy: 55.0%
- Cross-adapter U spectral norm: 0.21 (moderate alignment)
- Coupling: rank-2 cos=0.064 → rank-4 cos=0.051 → rank-6 cos=0.026

### Layer 6
- SV spectrum similarly flat (σ₁/σ₆ ≈ 1.27)
- Top-3 energy: 56.1%
- Coupling: rank-2 cos=0.030 → rank-4 cos=0.044 → rank-6 cos=0.045

### Layer 20
- SV spectrum flat (σ₁/σ₆ ≈ 1.18)
- Top-3 energy: 55.3%
- Coupling: rank-2 cos=0.026 → rank-4 cos=0.042 → rank-6 cos=0.030

## Key Finding

**SVD filtering of B matrices cannot reduce composition interference** because:

1. **Near-uniform spectra.** At r=6, the ratio σ₁/σ₆ ≈ 1.2 across all layers. This means adapter capacity is spread evenly — there is no "noise tail" to filter. The intrinsic dimensionality result (Aghajanyan et al.) applies to the full weight perturbation ΔW, not to the learned B matrix within a fixed-rank LoRA — at r=6, all 6 dimensions are load-bearing.

2. **Filtering increases coupling.** Rank-2 filtering produces HIGHER mean|cos| than full rank at 2/3 layers. The mechanism: all B_i = W @ A_i^T project through the same W, so their top SVs (=W's dominant output directions) are correlated. Removing lower SVs concentrates representation in the shared top directions, increasing not decreasing interference.

3. **Cross-adapter U alignment is low but equal across SV indices.** Per-SV alignment U_i[:,k]^T U_j[:,k] ≈ 0.01-0.07 for all k — no systematic pattern where top SVs are more or less aligned. The coupling comes from the collective structure, not individual SV pairs.

## Structural explanation

For B_i = W @ A_i^T (gradient-free construction):
- U_i spans the column space of W projected through A_i^T
- Since all A_i operate on the same W, the U_i share W's dominant output directions
- SVD(B_i) ≈ rotation of SVD(W) restricted to A_i's subspace
- Filtering removes the same output-space dimensions from ALL adapters simultaneously

This makes rank filtering structurally incapable of providing relative decorrelation. The composition residual tau ≈ 0.48 (F#752) is dominated by B-matrix coupling (F#815), but SVD filtering is not the path to reducing it.

## Implications

1. **E16 (tight NRE bounds):** Must account for near-uniform B spectra — bounds that assume spectral decay will be vacuous.
2. **E22 (poisoning robustness):** Grassmannian provides ~33% decorrelation (E14), B-matrix filtering adds nothing — safety mechanism must use different approach.
3. **Alternative paths to reducing tau:**
   - Train B matrices with explicit orthogonality penalty (B_i^T B_j regularizer during SFT)
   - Use independent training data so B matrices develop genuinely different column spaces
   - Post-hoc rotation: learn R_i such that (R_i B_i)^T (R_j B_j) is minimized

## Verdict

**KILLED** (smoke, method-level). All proxy and target KCs fail. Failure is structural, not sample-level — more layers/prompts cannot fix near-uniform spectra or the fundamental issue that B matrices from the same base weight share output-space structure.

- K2045 FAIL: top-3 energy 0.555 < 0.60
- K2045_target FAIL: no rank k achieves coupling ratio < 0.70
- K2046 FAIL: best coupling 0.034 > 0.030 threshold
- K2046_quality PASS: norm retention 1.0 (irrelevant — only full rank is best)

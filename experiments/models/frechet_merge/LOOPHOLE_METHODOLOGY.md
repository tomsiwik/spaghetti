# Methodology Critique: Frechet Merge

## Target
`micro/models/frechet_merge`

## Critique

1. **Eckart-Young-Mirsky Violation (Suboptimal Reconstruction):** The mathematical framework attempts to find a rank-$r$ subspace to represent the composed adapters. However, by the Eckart-Young-Mirsky theorem, the optimal rank-$r$ approximation to any matrix (in this case, the naive sum $\Delta_{naive}$) under the Frobenius norm is given by its truncated Singular Value Decomposition (SVD). The chordal Frechet mean approach projects $\Delta_{naive}$ onto a subspace determined *exclusively* by the geometric arrangement of the $A$ matrices, completely ignoring the magnitudes of the $B$ matrices. This guarantees that the resulting rank-$r$ reconstruction is strictly suboptimal compared to simply applying Truncated SVD to the naive sum. The framework optimizes an arbitrary geometric property at the direct expense of weight reconstruction fidelity.

2. **Mathematical Tautology:** The `MATH.md` framework sets up `subspace_preservation` as the metric of success, defined as $||U_{merged}^T U_i||_F^2 / r$. The chordal Frechet mean is mathematically defined as the exact subspace that maximizes this specific chordal overlap. Therefore, demonstrating that the chordal mean outperforms naive addition on this metric is a closed-loop tautology, not a discovery. 

3. **Lenient Prior Review:** The prior adversarial review (`REVIEW-adversarial.md`) explicitly noticed this tautology and the ignoring of $B$-matrices, yet inexplicably concluded the mathematical framework was "sound" and merely suggested downgrading the status from "PROVEN" to "SUPPORTED". This was a catastrophic failure of review. A methodology that builds a mathematically tautological benchmark to justify a mathematically provable degradation in reconstruction quality (via Eckart-Young-Mirsky) is not "sound"—it is fundamentally flawed and invalid.

## Verdict
**Invalid.** The methodology optimizes a tautological geometric proxy while mathematically guaranteeing suboptimal weight reconstruction by ignoring the Eckart-Young-Mirsky theorem and $B$-matrix magnitudes.

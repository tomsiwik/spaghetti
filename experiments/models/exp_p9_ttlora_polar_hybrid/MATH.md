# TT-LoRA + PoLAR Hybrid: Stiefel Retraction on Tensor Train Cores

TYPE: guided-exploration
PROVEN FRAMEWORK: PoLAR Stiefel → sr=r (Finding #442), TT left-canonical norm concentration (Oseledets 2011)
UNKNOWN: How much of the sr=r guarantee propagates through TT contraction when only interior cores are Stiefel-constrained

## Background

Standard LoRA: ΔW = B @ A ∈ R^{n × m}, rank ≤ r.
- Unconstrained: sr(ΔW) ≈ 1.77 for r=6 (Finding #442 — severe rank collapse)
- PoLAR (B, A ∈ Stiefel): sr(ΔW) = r exactly (Finding #442 — 9× improvement)

TT-LoRA: ΔW = reshape(contract(G_1, ..., G_d)), with d TT-cores.
- Rank ≤ r (Oseledets 2011, Thm 2.1: unfolding rank bounded by TT-rank)
- 12.4× parameter compression vs LoRA at 84.4% quality (Finding #516)
- Open question: does the TT contraction chain cause spectral collapse?

## Theorem 1: Left-Canonical Norm Concentration

(Oseledets, 2011, Property 2.3)

**Statement.** Let T = contract(G_1, ..., G_d) with each core G_k having
left-orthogonal unfolding: G_k^{<T} G_k^{<} = I_{r_k} for k = 1, ..., d-1.
Then ||T||_F = ||G_d||_F, where G_d is the d-th core flattened to a vector.

**Proof sketch.** By induction. The cumulative left contraction
L_k ∈ R^{(s_1...s_k) × r_k} has orthonormal columns at each step k.
For k=1: L_1 = G_1^{<} which is orthogonal by assumption.
For k → k+1: L_{k+1} = (L_k ⊗ I_{s_{k+1}}) · G_{k+1}^{<}, and since
(L_k ⊗ I)^T (L_k ⊗ I) = I ⊗ I = I, the isometric property propagates.
At step d: ||T||_F^2 = ||L_{d-1} @ G_d||_F^2 = ||G_d||_F^2 (since L_{d-1}
preserves norms). □

## Theorem 2: Isometric Prefix → Stable Rank Lower Bound

**Statement.** Under the same left-canonical conditions as Theorem 1, let
ΔW ∈ R^{m × n} be the matrix reshape of T at the input/output split.
Write ΔW = L @ R where:
- L ∈ R^{m × r_p} is the contraction of input-side cores (left of split)
- R ∈ R^{r_p × n} is the contraction of output-side cores (right of split)

If L has orthonormal columns (L^T L = I_{r_p}), then:
1. σ_i(ΔW) = σ_i(R) for all i
2. sr(ΔW) = sr(R) = ||R||_F^2 / ||R||_2^2

**Proof.**
ΔW^T ΔW = R^T L^T L R = R^T R (since L^T L = I).
Therefore the eigenvalues of ΔW^T ΔW equal those of R^T R.
Hence σ_i(ΔW) = σ_i(R), and:
sr(ΔW) = Σ σ_i(ΔW)^2 / max_i σ_i(ΔW)^2 = ||R||_F^2 / ||R||_2^2 = sr(R). □

**Corollary.** sr(ΔW) ≥ 1, with equality iff R has effective rank 1.

## Theorem 3: Stiefel Cores Prevent Chain Collapse

**Statement.** For unconstrained TT-LoRA, the input-side contraction L is NOT
guaranteed to have orthonormal columns. In general:
||ΔW||_2 ≤ Π_k ||G_k^{<}||_2

If any input-side core has condition number κ(G_k) >> 1, the operator norm
Π||G_k||_2 can be much larger than ||R||_2, causing:
sr(ΔW_unconstrained) ≤ ||ΔW||_F^2 / (Π_k ||G_k||_2)^2 · ... (loose bound)

In practice, spectral amplification from ill-conditioned cores compresses sr
toward 1, analogous to how unconstrained B, A in LoRA causes rank collapse.

**Mechanism:** Stiefel constraint on input-side cores forces ||G_k^{<}||_2 = 1
(all singular values = 1), making L an isometry. This eliminates spectral
amplification from the input chain, so sr(ΔW) = sr(R) depends only on the
output-side cores (or last core in fully left-canonical form).

## Application to Gemma 4 E4B

For v_proj (2560 → 512), TT shape [5, 8, 8, 8, 8, 8, 8], TT-rank r = 6:
- Split at index 4: input factors [5, 8, 8, 8] → m = 2560, output [8, 8, 8] → n = 512
- Input-side cores: G_1(1,5,6), G_2(6,8,6), G_3(6,8,6), G_4(6,8,6)
- Output-side cores: G_5(6,8,6), G_6(6,8,6), G_7(6,8,1)

Interior core left unfoldings:
- G_2^{<}: (48, 6) — Stiefel OK (48 ≥ 6)
- G_3^{<}: (48, 6) — Stiefel OK
- G_4^{<}: (48, 6) — Stiefel OK
- G_5^{<}: (48, 6) — Stiefel OK
- G_6^{<}: (48, 6) — Stiefel OK

First core G_1^{<}: (5, 6) — cannot enforce Stiefel (5 < 6). Skip.
Last core G_7: (6, 8, 1) → reshape to (6, 8). Unconstrained (stores learned info).

**Retraction approach:** Polar decomposition via SVD on (48, 6) matrices.
Cost: SVD of 48×6 per core × 5 interior cores × 42 layers = 210 small SVDs.
Each SVD of a 48×6 matrix: O(48 × 6^2) = O(1728) flops. Negligible.

## Predictions

| Metric | TT-LoRA (unconstrained) | TT-LoRA-Stiefel | Basis |
|--------|------------------------|-----------------|-------|
| sr(ΔW) mean | 1.5-3.0 | 3.0-6.0 | Thm 2: Stiefel prefix → sr = sr(R) |
| sr(ΔW) ratio | baseline | 1.5-3× higher | PoLAR achieved 9× vs LoRA (Finding #442) |
| GSM8K accuracy | ~65% | ≥ 60% | Stiefel = regularizer, not capacity loss |
| Params | 64,260 | 64,260 | Same architecture, retraction doesn't add params |
| Retraction time | 0 | < 0.1 ms/step | 210 SVDs of 48×6, O(1728) each |

**Behavioral prediction:** TT-LoRA-Stiefel should produce GSM8K solutions of
similar quality to unconstrained TT-LoRA. The sr improvement means the correction
uses more of its rank-6 capacity, which should manifest as more diverse/robust
reasoning patterns rather than collapsing to a dominant direction.

## Kill Criteria Grounding

| Criterion | Mathematical Basis |
|-----------|-------------------|
| K1363: sr(TT-Stiefel) > sr(TT-LoRA) | Thm 2+3: isometric prefix eliminates spectral amplification |
| K1364: Quality ≥ TT-LoRA | Stiefel constrains direction, not capacity; regularization helps |
| K1365: Retraction < 1ms/step | 210 SVDs of 48×6 matrices, ~O(360K) total flops |

## References

- Oseledets, I. (2011). "Tensor-Train Decomposition." SIAM J. Sci. Comput.
- Batselier et al. (2025). "TT-LoRA MoE" arXiv:2504.21190
- Finding #442: Joint Stiefel PoLAR guarantees sr=r on Gemma 4
- Finding #515: TT-LoRA MLX port (8.3x compression)
- Finding #516: TT-LoRA quality (84.4%, 12.4x compression)

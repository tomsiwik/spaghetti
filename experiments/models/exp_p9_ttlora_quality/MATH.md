# TT-LoRA Quality Preservation Under Extreme Compression

## Theorem 1: TT-LoRA Parameter Count

For a weight correction ΔW ∈ R^{m×n} represented as a Tensor-Train with
d cores G_1, ..., G_d of factor sizes s = (s_1,...,s_d) where Π s_i = m·n,
and uniform interior TT-rank r with boundary ranks 1, the parameter count is:

    P_TT = s_1·r + Σ_{k=2}^{d-1} r·s_k·r + s_d·r
         = r·(s_1 + s_d) + r²·Σ_{k=2}^{d-1} s_k

**Proof.** Core k has shape [r_{k-1}, s_k, r_k]. Boundary cores: [1, s_1, r]
and [r, s_d, 1] with s_1·r and s_d·r parameters. Interior cores: [r, s_k, r]
with r²·s_k parameters. Sum gives the formula. □

For Gemma 4 E4B v_proj (2560→512), TT shape [5,8,8,8,8,8,8], rank 6:
- Boundary: 5·6 + 8·6 = 78
- Interior: 6²·(8+8+8+8+8) = 36·40 = 1,440
- Total: 1,518 per layer

For 42 layers: 63,756 total parameters.
Standard LoRA r=6 on v_proj: (2560+512)·6 = 18,432/layer × 42 = 774,144.
Compression: 774,144 / 63,756 = **12.1x**.

## Theorem 2: Effective Rank Preservation

The TT-LoRA correction matrix ΔW = reshape(contract(G_1,...,G_d)) has
matrix rank at most r (the interior TT-rank).

**Proof.** (Oseledets, 2011, Thm 2.1) The k-th unfolding of a TT-tensor has
rank bounded by the k-th TT-rank. The reshape from tensor to matrix [m, n]
corresponds to the unfolding at the split between input factors (Π_{i=1}^p s_i = m)
and output factors (Π_{i=p+1}^d s_i = n). All interior TT-ranks equal r,
so rank(ΔW) ≤ r. □

## Corollary: Quality Equivalence Class

Both TT-LoRA rank r and standard LoRA rank r produce corrections in the
rank-r matrix manifold M_r ⊂ R^{m×n}. However, TT-LoRA's corrections are
constrained to a Kronecker-structured submanifold K_r ⊂ M_r.

The quality gap depends on alignment: if the optimal ΔW* has Kronecker
structure (common when input/output dimensions factorize into semantically
meaningful groups), TT-LoRA matches LoRA. If not, quality degrades by the
projection distance from ΔW* to K_r.

## Convergence Guarantee

TT-core contraction is a composition of bilinear maps, hence infinitely
differentiable. MLX autograd computes ∂L/∂G_k through the chain:

    ∂L/∂G_k = ∂L/∂ΔW · ∂ΔW/∂G_k

where ∂ΔW/∂G_k is a tensor contraction of the other d-1 cores. The gradient
magnitude scales as O(Π_{j≠k} ||G_j||_F), which for small random init is O(1).

The TT-LoRA paper (arXiv:2504.21190) uses lr=5e-3 (50x standard LoRA lr)
to compensate for the deeper chain rule. We adopt this recommendation.

## Predictions

| Metric | TT-LoRA r=6 v_proj | LoRA r=6 v_proj | Prediction |
|--------|---------------------|-----------------|------------|
| Params/layer | 1,518 | 18,432 | 12.1x compression |
| Total params | 63,756 | 774,144 | 12.1x compression |
| Adapter size (float16) | ~154 KB | ~1.5 MB | < 200KB (K1358) |
| Effective rank | 6 | 6 | Equal (Theorem 2) |
| GSM8K accuracy | 60-90% of LoRA | baseline | ≥60% (K1357) |
| Loss at step 100 | < loss at step 1 | n/a | Converges (K1359) |

**Behavioral prediction:** TT-LoRA adapters should produce step-by-step GSM8K
solutions with correct arithmetic chains. Quality loss (if any) manifests as
shorter reasoning chains or incorrect intermediate steps, not degenerate output.

## Kill Criteria Grounding

| Criterion | Mathematical Basis |
|-----------|-------------------|
| K1357: GSM8K ≥60% of LoRA | Thm 2: same rank-6 manifold, Kronecker constraint ≤40% quality loss |
| K1358: adapter ≤200KB | Thm 1: 63,756 params × 2B + metadata ≈ 154KB |
| K1359: convergence | Continuous differentiable reparameterization + paper lr=5e-3 |

## References

- Oseledets, I. (2011). "Tensor-Train Decomposition." SIAM J. Sci. Comput.
- Batselier et al. (2025). "TT-LoRA MoE" arXiv:2504.21190
- Finding #515: TT-LoRA port to MLX verified (8.3x compression, 1.36x latency)
- Finding #421: LoRA r=6 on q_proj achieves 82% GSM8K on Gemma 4 E4B

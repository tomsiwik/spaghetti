# MATH.md — Stiefel-KAN: hard-orthogonality + additive composition

## Hypothesis

KAN gives us composition-as-coefficient-addition (`c_merged = Σ_k c_k`),
but the *output* at a node is `Σ_i ϕ_ij(z_i)`. Two adapters with
overlapping spline supports still produce non-trivially combined
contributions for shared inputs.

The mathematical fix: **constrain per-adapter spline coefficient vectors
to a Stiefel frame** (rows of an orthonormal matrix). For each edge
(i, j) and K adapters, stack coefficients into

    M_ij = stack([c_1_ij, c_2_ij, ..., c_K_ij])  ∈ ℝ^{K × G}    (G = grid_size)

with the constraint

    M_ij · M_ij^T = I_K                              (rows orthonormal)

This is Pierre's PoLAR Grassmannian-sibling theorem applied at the spline-
coefficient level instead of the matmul-A level.

## What this guarantees

**At init and throughout training**, for any pair (k, j), `c_k_ij ⊥ c_j_ij`
in coefficient space. The B-spline basis is fixed; the basis functions
`B_k(u)` are continuous and bounded. So:

    ⟨ϕ_k, ϕ_j⟩_layer  =  ⟨c_k_ij, c_j_ij⟩  ·  ⟨B_basis, B_basis⟩
                       =  0  ·  Gram(B-basis)
                       =  0

i.e. **adapters' contributions at each edge are orthogonal as functions**.
This is strictly stronger than what TIES, Pico, ACE, or Fisher-Rao
provide — they all manage interference *empirically* via post-hoc
correction. Stiefel-KAN forbids it by manifold constraint.

> **Q1 (FEASIBILITY)**: Can K=7 spline coefficient vectors per edge be
> trained on a Stiefel manifold without losing per-task expressivity?
>
> **Q2 (ORTHOGONALITY-IN-PRACTICE)**: Does the Stiefel constraint
> actually drive cross-contribution to zero on Pierre's benchmark suite?
>
> **Q3 (COMPOSITION SCALING)**: Does Stiefel-KAN composition scale to
> K=7 without degrading per-task accuracy below TIES-B baseline (71.3%)?

## Architecture

For each (layer, q_proj) edge (i, j):

```
Spline coefficients per adapter:
    c_k_ij ∈ ℝ^G          # G = grid_size, e.g. 7 for K=7
Stack:
    M_ij = [c_1_ij; c_2_ij; ...; c_K_ij]  ∈ Stiefel(K, G)
Forward (single adapter k):
    ϕ_k_ij(u) = Σ_g c_k_ij[g] · B_g(u)     # B-spline basis at u
Forward (composed):
    ϕ_merged_ij(u) = Σ_k λ_k · ϕ_k_ij(u)
                   = Σ_k λ_k · Σ_g c_k_ij[g] · B_g(u)
                   = Σ_g (Σ_k λ_k c_k_ij[g]) · B_g(u)
    i.e. c_merged_ij = Σ_k λ_k c_k_ij     (additive at coefficient level)
```

The Stiefel constraint requires K ≤ G. For K=7, set G=8 or G=16. The
"slack" dimension (G − K) provides room for the orthogonal frame to live.

## Training procedure

Standard Riemannian SGD on Stiefel:

1. Initialize each `M_ij` as a random Stiefel frame (QR of random Gaussian).
2. For each minibatch, compute Euclidean gradient `∇M_ij`.
3. Project to Stiefel tangent space: `T = ∇M − M sym(M^T ∇M)`.
4. Retract back to Stiefel via QR: `M_new = QR(M − η T)`.
5. Repeat.

Cost per step: 7 QRs of (K × G) = (7 × 8) matrices per edge = ~50 FLOPs
per edge. Negligible vs the matmul cost dominating training.

## Pre-registered Kill Criteria

- **K1 (FEASIBILITY)** Per-task accuracy of single Stiefel-KAN adapter
  ≥ standard PoLAR adapter − 5pp on its native benchmark.
  PASS → Stiefel constraint doesn't kill expressivity.

- **K2 (ORTHOGONALITY)** Pairwise function-space inner product
  `⟨ϕ_k, ϕ_j⟩` averaged over input distribution ≤ 0.05 (≈ 5% of the
  norm of either function).
  PASS → constraint translates to actual function-space orthogonality.

- **K3 (COMPOSITION)** K=7 composed accuracy on (GSM8K, HumanEval, MedQA)
  ≥ TIES-B baseline (71.3%).
  PASS → hard-orthogonality matches or beats heuristic structured merge.

- **K4 (CROSS-CONTRIBUTION)** Adding K-1 other adapters to adapter k's
  forward perturbs k's output by ≤ 2% on its native task.
  PASS → composition is *behaviorally* superposition. The strongest
  single number we'd care about.

## Verdict logic

| K1 | K2 | K3 | K4 | Outcome |
|----|----|----|----|---------|
| ✓ | ✓ | ✓ | ✓ | **SUPPORTED** — Stiefel-KAN is the math-guaranteed composition. Plan migration. Composition arc closed *with theorem*, not just empirics. |
| ✓ | ✓ | ✓ | ✗ | **SUPPORTED w/ note** — function-space orthogonal but small behavioral cross-effects remain (probably from grid quantization). Adopt. |
| ✓ | ✓ | ✗ | * | **PARTIAL** — orthogonal but composition operator wrong; investigate weighted (λ_k ≠ 1/K) variants. |
| ✓ | ✗ | * | * | **CONTRADICTION** — Stiefel constraint isn't producing orthogonality. Implementation bug; debug before claiming anything. |
| ✗ | * | * | * | **KILLED** — Stiefel constraint kills expressivity. Either G is too small (raise it) or hard-orthogonality is too restrictive at K=7. |

## Implementation status

**SPEC + ALGORITHM — implementation is non-trivial.**

What's needed:
1. **MLX Stiefel retraction** (~80 LoC). QR-based on `mx.linalg.qr`. Apply
   per-edge per training step. Verifiable via `M @ M.T ≈ I` post-step.
2. **Riemannian gradient projection** (~30 LoC). Tangent-space project
   before retract. Standard formula.
3. **KAN block from `exp_pierre_kan_adapter_lagrangian`** (already exists)
   needs the Stiefel constraint added to its `coefficients` parameter.
4. **Training script** (~250 LoC) — multi-task data loader (math + code +
   medical), per-task adapter index selection, Stiefel-aware optimizer.
   Training cost: 7 adapters × ~30 min each on M5 Pro = ~3.5h.
5. **Eval rig** is the existing shared `eval_runner` infrastructure. KAN
   block install path already in `exp_pierre_kan_adapter_lagrangian`.

Estimated total: **6-8h of focused MLX implementation + 3.5h of training**.

## What this experiment is NOT

- Not a from-scratch full architecture replacement. Reuses Pierre's
  shared-A path, just changes B's parameterization to Stiefel-KAN.
- Not a softmax-replacement experiment. Attention is unchanged.
- Not a runtime cost reduction. Inference cost is comparable to KAN
  baseline (per-FLOP slower than matmul, per-byte cheaper).

## Why this is the experiment that matters

If `exp_pierre_kan_compositional_orthogonality` (sibling) finds existing
adapters have low natural support overlap → Stiefel-KAN is incremental
insurance.

If it finds high overlap (likely for similar-domain adapters) → Stiefel-
KAN is the architecture that **mathematically guarantees** what TIES-B
achieves heuristically. The user's stated priorities (mathematical
stability, structural guarantees, scalar-reducible composition) align
with this exact construction.

The thesis: **composition is interference iff parameters are
interfering**. Stiefel constrains parameters to be non-interfering.
Therefore composition is non-interfering. By construction. No TopK,
no sign-elect, no rescaling.

## References

- KAN paper (arxiv 2404.19756)
- PoLAR Grassmannian sibling theorem (Pierre's design doc)
- Riemannian Optimization for LoRA on Stiefel (arxiv 2508.17901) — closest
  prior art for Stiefel-constrained adapter training
- StelLA (arxiv 2510.01938) — Stiefel-LoRA subspace learning
- Sibling: `exp_pierre_kan_adapter_lagrangian` (P1) — KAN parameterization
  expressivity test, must pass first
- Sibling: `exp_pierre_kan_compositional_orthogonality` (P1) — does
  natural support-disjointness exist; if no, Stiefel is needed

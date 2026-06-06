# MATH.md: SLERP B-matrix Composition

## TYPE: verification
## PRIOR MATH: MANIFOLD_COMPOSITION.md §2 (Kavan et al. 2006 dual-quaternion analogy)
## FAILURE MODE: Linear blending of diverse B-matrices collapses composed adapter strength ("candy wrapper")

---

## Problem Statement

Current Pierre composition:
```
h = W·x + Σᵢ αᵢ Bᵢ(Aᵢ·x)
```
Grassmannian A guarantees ε₁ = 0 (parameter-space orthogonality).
BUT: when αᵢ = 1/N equal-weight and we pre-combine B matrices:
```
B_linear = Σᵢ αᵢ Bᵢ
```
the combined B has Frobenius norm that COLLAPSES as adapters become diverse.
This is the **candy wrapper effect** — the same failure mode as linear blend skinning
in skeletal animation (Kavan et al., SCA 2006).

---

## Theorem 1: LERP Norm Collapse

**Theorem.** For normalized B̂₁, B̂₂ ∈ R^{d_out×r} with ||B̂₁||_F = ||B̂₂||_F = 1 and
angle θ = arccos(⟨vec(B̂₁), vec(B̂₂)⟩ / (||B̂₁||_F · ||B̂₂||_F)):

```
||LERP(B̂₁, B̂₂, t)||_F² = (1-t)² + t² + 2t(1-t)cos(θ)
                          = 1 - 2t(1-t)(1 - cos(θ))
```

At t = 0.5:
```
||LERP(B̂₁, B̂₂, 0.5)||_F = √((1 + cos(θ)) / 2)
```

**Proof.**
||LERP||² = ||(1-t)B̂₁ + t·B̂₂||_F²
           = (1-t)²||B̂₁||² + t²||B̂₂||² + 2t(1-t)⟨B̂₁, B̂₂⟩_F
           = (1-t)² + t² + 2t(1-t)cos(θ)
           = 1 - 2t(1-t) + 2t(1-t)cos(θ)    [since (1-t)²+t² = 1-2t(1-t)]
           = 1 - 2t(1-t)(1 - cos(θ))
QED.

**Corollary.** At t=0.5: norm = √((1+cos(θ))/2). Minimum at θ=π: norm = 0 (complete collapse).

**For diverse trained adapters (cos(θ) ≈ 0):** LERP norm = 1/√2 ≈ 0.707 → 29% dip.
**For N=5 equal-weight uniform random B:** LERP norm ≈ 1/√N = 0.447 → 55% dip.

---

## Theorem 2: SLERP Norm Preservation

**Theorem.** For any B̂₁, B̂₂ on the unit Frobenius sphere and any t ∈ [0,1]:
```
||SLERP(B̂₁, B̂₂, t)||_F = 1
```

**Proof.**
SLERP(u, v, t) = sin((1-t)θ)/sin(θ) · u + sin(tθ)/sin(θ) · v

||SLERP||² = sin²((1-t)θ)/sin²(θ) + sin²(tθ)/sin²(θ) + 2sin((1-t)θ)sin(tθ)cos(θ)/sin²(θ)

Using the identity sin((1-t)θ)sin(tθ) = [cos((1-2t)θ) - cos(θ)] / 2:

||SLERP||² = [sin²((1-t)θ) + sin²(tθ) + cos(θ)(cos((1-2t)θ) - cos(θ))] / sin²(θ)
           = [1 - cos²(θ)] / sin²(θ)    [by sin²A + sin²B + 2sin(A)sin(B)cos(θ) identity]
           = sin²(θ) / sin²(θ) = 1
QED.

**Iterative SLERP for N adapters** (sequential pairwise with accumulating weight):
```
B_slerp^(1) = B̂_1
for i = 2..N:
    t_i = w_i / (Σ_{j≤i} w_j)
    B_slerp^(i) = SLERP(B_slerp^(i-1), B̂_i, t_i)
B_slerp = s · B_slerp^(N)    where s = Σᵢ αᵢ ||Bᵢ||_F  (scale blended linearly)
```
By induction on Theorem 2, ||B_slerp^(i)||_F = 1 for all i.

---

## Theorem 3: Quality Bound Under Composition

**Claim.** For equal-weight composition over N diverse adapters (independent B directions):
```
Quality(SLERP) / Quality(LERP) ≈ 1/√N  at large N
```

**Reasoning.** The effective adapter signal strength is proportional to ||B_composed||_F:
- SLERP: ||B_S||_F = Σᵢ αᵢ||Bᵢ||_F = s (preserves scale)
- LERP: ||B_L||_F ≈ s/√N (norm dip from diverse directions)

Since adapter perturbation is ||B||·σ_A (where σ_A = spectral norm of A), the
adapter SNR ratio is SLERP/LERP ≈ √N. At N=5: expected 2.24x stronger signal.

Stronger signal → lower perplexity → better quality.

---

## Experimental Design

**Architecture:**
- D_MODEL=256, N_LAYERS=2, N_HEADS=4, LORA_RANK=4, VOCAB=128

**Adapter structure (SHARED A):**
- All 5 adapters share a COMMON frozen A matrix per layer (random init, Kaiming)
- Each domain has its OWN B matrix (trained by SFT)
- Composition: B_combined = LERP(B₁,...,B₅) or SLERP(B₁,...,B₅), then apply A

**Why shared A:** Makes SLERP on B well-defined — all Bᵢ map from the SAME subspace.
Individual adapter: h += B_i(A·x). Composed: h += B_combined(A·x).

**Domains:** arithmetic, sort, reverse, repeat, parity (5 diverse seq2seq tasks)

**Evaluation:**
- Per-domain perplexity under: (a) base model, (b) single SFT adapter, (c) LERP composed, (d) SLERP composed
- Quality ratio = exp(loss_base - loss_composed) [relative perplexity improvement]
- Norm ratio = ||B_composed||_F / mean_i(||B_i||_F)

---

## Quantitative Predictions

| Metric | LERP (predicted) | SLERP (predicted) | Source |
|--------|-----------------|-------------------|--------|
| B-matrix cos(B_i, B_j) at N=5 | ~0.1 (near-orthogonal) | same | High-dim geometry |
| Norm ratio at N=5 | ~0.45 (1/√5) | ~1.0 (exact) | Theorems 1&2 |
| SLERP/LERP norm ratio | — | ≥ 2.0 | Theorem 2 |
| Quality ratio vs base | baseline | ≥ baseline | Theorem 3 |

**K931 threshold (>30% strength advantage for SLERP):**
SLERP norm / LERP norm > 1.30 ✓ (predicted ≥ 2.0 at N=5)

**K932 threshold (SLERP quality ≥ LERP quality):**
PPL_SLERP ≤ PPL_LERP on held-out mixed domain test ✓

---

## Kill Conditions

**K931 KILLED if:** SLERP norm ratio / LERP norm ratio ≤ 1.30 at N=5.
Impossibility structure: Theorem 2 is provably true — norm = 1 always.
If K931 fails, it means the B-matrices all point in similar directions (cos ≈ 1),
so LERP norm ≈ 1 too. This would mean adapters are NOT diverse, which is a finding
about B-matrix geometry, not a failure of SLERP.

**K932 KILLED if:** SLERP quality < LERP quality at N=5.
Would mean: despite stronger adapter signal, SLERP degrades quality. This would imply
the DIRECTION of composition matters more than the magnitude — an important finding
that would point toward the Polar Decomposition approach (Section 3 of MANIFOLD_COMPOSITION.md).

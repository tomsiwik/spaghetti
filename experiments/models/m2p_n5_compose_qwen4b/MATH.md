# MATH.md — N=5 Domain Composition at 4B: Grassmannian Scaling Verification

## Problem Statement

Finding #404 verified 2-domain Grassmannian composition at 4B (d=2560, r=4, qr=1.3125). 
Finding #393 verified N=50 at 0.6B (d=896). The question is whether N=5 at 4B holds
identically — not a scaling challenge but a direct verification of the capacity bound.

## Theorem 1: Sequential Gram-Schmidt Orthogonality for N=5

**Theorem.** Let A_1 ∈ R^{d×r} (math) and A_2 ∈ R^{d×r} (code) be fixed Grassmannian
matrices (from Findings #403, #404). Define A_3, A_4, A_5 by sequential Gram-Schmidt:

  For k = 3, 4, 5:
    1. Sample Q ∼ N(0, I_{d×r})
    2. Project out prior subspaces: Q ← Q - Σ_{i<k} A_i (A_i^T Q)
    3. Orthonormalize: A_k, _ ← QR(Q); take first r columns

Then for all i ≠ j: A_i^T A_j = 0 (exact, in float64).

**Proof.** By construction at step 3, A_k is in the nullspace of {A_1,...,A_{k-1}}.
The QR decomposition preserves the nullspace constraint. Since A_i is formed in the
nullspace of all prior A, A_i^T A_k = 0 for all i < k. By symmetry, all pairs are
orthogonal. ∎

**Precision corollary.** In bfloat16 (mantissa precision ~3e-4, Finding #404 measured
1.38e-05): max|A_i^T A_j|_F < ε_{bf16} ≈ 1e-4 across all 10 pairs.

**Prediction for K978:** max|A_i^T A_j|_F < 1e-4 for all C(5,2)=10 pairs (PASS).

---

## Theorem 2: Capacity Bound N_max = 640 >> 5

**Theorem.** In R^d, the maximum number of mutually orthogonal rank-r subspaces is:

  N_max = ⌊d / r⌋

For d=2560, r=4: N_max = 640.

**Proof.** Each rank-r subspace consumes r dimensions. Mutual orthogonality requires 
the r-dimensional column spaces to be disjoint. R^d can accommodate at most ⌊d/r⌋ 
such disjoint subspaces. ∎

**Implication.** N=5 uses 5×4=20 of 2560 dimensions (0.78%). Capacity is not a
constraint. All 10 pairwise orthogonality conditions are simultaneously achievable.

---

## Theorem 3: TF-IDF 5-Class Separability

**Theorem.** Let the 5 domains have anchor vocabularies:
- Math: {sold, earned, total, spent, how many, profit, each, altogether, ...}
- Code: {Python, function, def, return, implement, write, output, code, ...}
- Sort: {sort, alphabetically, arrange, alphabetical, order these words, ...}
- Reverse: {reverse, order of these words, flip, backwards, ...}
- Count: {count the words, how many words, phrase, number of words, ...}

These vocabularies are disjoint by construction. By the centroid-based nearest-neighbor
theorem (Finding #390: 100% routing at N=2), with N_train=100 examples per class:

  P(correct routing) ≥ 1 - 5 × exp(-γ² N_train / 2)

where γ = min pairwise centroid separation in cosine space > 0 for disjoint vocabularies.

**Prediction for K979:** Routing accuracy ≥ 95% (K979 threshold: 80%).

---

## Theorem 4: Math Quality Preservation Under N=5 Composition

**Theorem.** Under routed composition with N=5 Grassmannian-isolated adapters, the
math quality_ratio satisfies:

  quality_ratio(math, N=5) = quality_ratio(math, N=2)  (up to routing noise)

**Proof.** Let B̂_math be the M2P-generated B-matrix for the math query. The composed
forward pass applies W_applied = W_base + A_math B̂_math (routing selects math domain).
The remaining 4 domains' adapters are NOT added — routing is exclusive. Therefore,
the computation is identical to the 2-domain case (Finding #404: qr=1.3125). ∎

**Note on additive composition:** Even if all 5 adapters were simultaneously applied,
the interference term is:
  ||Σ_{i≠math} A_i B_i||² / ||A_math B_math||²  →  0

since A_i^T A_math ≈ 0 (Theorem 1) and the cross-terms vanish in the product A_math^T A_i ≈ 0.

**Prediction for K980:** quality_ratio ≥ 0.70 (conservative); expected ≈ 1.31 (N=2 value).

---

## Quantitative Predictions Table

| Prediction | Formula | Expected | Kill Threshold |
|------------|---------|----------|----------------|
| max|A_i^T A_j|_F (10 pairs) | Gram-Schmidt → 0 | < 1e-5 (fp64); < 1e-4 (bf16) | K978: < 1e-4 |
| Routing accuracy (5-class) | Centroid cosine NN | ≥ 95% | K979: ≥ 80% |
| Math quality_ratio under N=5 | Same as N=2 (routed) | ≈ 1.31 | K980: ≥ 0.70 |
| N_max at d=2560, r=4 | ⌊d/r⌋ = 640 | N/A | structural |
| Dimensions consumed (N=5) | 5×r = 20 | 20/2560 = 0.78% | structural |

## Failure Mode Analysis

- **K978 FAIL:** Gram-Schmidt bug or wrong float precision in projection. 
  Fix: use float64 throughout, verify QR preserves nullspace constraint.
- **K979 FAIL:** Domain vocabularies overlap (e.g., "sort" appears in math queries).
  Fix: use domain-specific prefix strings with unique anchor tokens.
- **K980 FAIL:** Code/sort/reverse/count M2P B-matrices accidentally corrupt math computation.
  This would indicate Theorem 1 is violated, or routing is incorrect.
  Structural impossibility: Grassmannian isolation + exclusive routing makes this impossible.

## Prior Art

- Finding #404: N=2, d=2560, r=4, qr=1.3125, all K PASS
- Finding #393: N=50, d=896, r=4, max|A_i^T A_j|=9.50e-08
- LoraRetriever (arXiv:2402.09997): contrastive LoRA retrieval at N=100+ adapters
- He et al. (2016): residual connections preserve initialization quality

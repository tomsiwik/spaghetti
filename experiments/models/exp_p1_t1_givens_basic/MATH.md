# MATH.md — T1.3: Givens Rotation Orthogonality at d=2816

## Setup

**Paper:** qGOFT: Quasi-Orthogonal Fine-Tuning via Givens Rotations (arxiv 2404.04316)
**Context:** P1 adapter architecture uses q_proj only on NoPE dims [128:512] (T0.3/T0.4).
  We need an adapter parameterization that is (1) orthogonal for structural interference prevention,
  (2) O(d) parameters, (3) parallelizable on Apple Silicon (M5 Pro).

Givens rotations are the candidate: each is a 2×2 rotation on a pair of dimensions,
independently parameterized by one angle θ. A block of d/2 non-overlapping pairs covers
all d dimensions in O(d) time and O(d) parameters.

---

## Theorem 1: Single-Layer Givens is Exactly Orthogonal

**Claim:** Let G_k(θ_k) denote a Givens rotation on dimensions (2k, 2k+1) for k = 0, …, d/2−1.
These d/2 rotations act on disjoint pairs of dimensions. Their product

    O = G_0(θ_0) ⊗ G_1(θ_1) ⊗ … ⊗ G_{d/2−1}(θ_{d/2−1})

is an exactly orthogonal matrix: O^T O = I_d.

**Proof:**
Each G_k is a 2×2 rotation matrix, hence G_k^T G_k = I_2. Because the pairs are disjoint,
the full d×d matrix is block-diagonal:

    O = diag(G_0, G_1, …, G_{d/2−1})

Then:
    O^T O = diag(G_0^T, …) · diag(G_0, …) = diag(G_0^T G_0, …) = diag(I_2, …) = I_d.

In exact arithmetic, ‖O^T O − I‖_F = 0. In float32, quantization error O(d × ε_mach)
with ε_mach ≈ 1.2e−7 gives ‖O^T O − I‖_F ≤ √d · ε_mach ≈ √2816 × 1.2e−7 ≈ 6.4e−6. **QED**

---

## Theorem 2: Parallel Execution — d/2 Rotations per Block

**Claim:** All d/2 Givens rotations in a single layer can execute in parallel (zero data
dependency between rotations in the same block).

**Proof:**
Rotation G_k operates exclusively on indices {2k, 2k+1}. For k ≠ l:
{2k, 2k+1} ∩ {2l, 2l+1} = ∅. Therefore no rotation reads an output written by another
rotation in the same block. The block can be vectorized as a single batched 2×2 matrix
multiply over d/2 independent pairs, executing in O(d) time on d/2 parallel units. **QED**

On MLX/Metal: implemented as reshape-to-(N, d/2, 2), broadcast multiply by (d/2, 2, 2)
rotation matrices → single matmul kernel.

---

## Theorem 3: Parameter Count is O(d)

**Claim:** A depth-L Givens adapter has ≤ L × d/2 total parameters.

**Proof:**
Each layer has exactly d/2 angles (one per disjoint pair). L layers give L × d/2 angles.
For d=2816: one layer = 1408 params; L=8 layers = 11264 params. Compare with LoRA r=8:
2 × d × r = 2 × 2816 × 8 = 45056 params. Givens is 4x more parameter-efficient at
equivalent depth. **QED**

---

## Quantitative Predictions

| Metric | Prediction | Kill Criterion |
|--------|-----------|----------------|
| ‖O^T O − I‖_F (single layer, float32) | < 6.4e-6 (theory), < 1e-4 (K1015) | K1015: < 1e-4 |
| Parallel execution (d/2 ops per block) | Confirmed structurally | K1016: structural |
| Total params, L=1 layer | d/2 = 1408 ≤ d = 2816 | K1017: ≤ O(d) |

**Expected outcomes:**
- K1015: ‖O^T O − I‖_F ≈ 5e-6 to 1e-5 in float32 (well below 1e-4 threshold)
- K1016: d/2 rotations implemented as single batched matmul → structurally parallel
- K1017: 1408 params per layer, O(d) confirmed

---

## Behavioral Connection

Exact orthogonality (Theorem 1) means the Givens adapter preserves the Frobenius norm of
the weight update. Combined with T0.3 (NoPE channel isolation) and T0.4 (Q-only KV sharing),
a Givens adapter on q_proj[NoPE] causes zero interference in the KV cache and zero
positional leakage into NoPE dimensions — structural interference prevention for multi-tenant serving.

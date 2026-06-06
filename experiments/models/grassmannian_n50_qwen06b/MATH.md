# MATH.md: Grassmannian Orthogonality at N=50 on Qwen3-0.6B

## Experiment Type
Verification (Type 1) — proof complete, experiment confirms predictions.

## Background

The M2P composition system assigns each domain adapter a slot on the Grassmannian
Gr(r, d_in): the set of all r-dimensional subspaces of R^{d_in}. The A-matrix
for each adapter defines this subspace. Interference requires A_i^T A_j ≠ 0;
orthogonality makes interference exactly zero.

Prior results:
- exp_m2p_2domain_compose_qwen06b: max|A_math^T A_code| = 1.51e-08 at N=2 (verified)
- Theory predicts N_max = floor(d_in/r) = floor(1024/4) = 256 for Qwen3-0.6B

This experiment verifies the capacity claim at N=50 and measures the memory footprint.

---

## Theorem 1: Grassmannian Capacity for Qwen3-0.6B

**Statement.** Let d_in = 1024, r = 4. Construct N Grassmannian A-matrices as follows:
Sample X ∈ R^{d_in × (N·r)} with entries iid N(0,1), compute thin QR: X = QR where
Q ∈ R^{d_in × (N·r)} has orthonormal columns. Set A_i = Q[:, i·r:(i+1)·r] for i=0..N-1.

**Claim.** For N ≤ N_max := floor(d_in/r) = 256:
```
A_i^T A_j = 0  (exact)  for all i ≠ j
```
**Numerical precision:** float32 QR yields max|A_i^T A_j|_F ≤ ε_mach · sqrt(d_in · r)
≈ 1.2e-7 · sqrt(4096) ≈ 7.6e-6.

**Kill criterion K948 threshold of 1e-5:** satisfied with margin (~10x buffer).

**Proof.**
By construction, Q has orthonormal columns: Q^T Q = I_{N·r}. Partitioning into
N blocks: [A_0 | A_1 | ... | A_{N-1}]. The (i,j) block of Q^T Q is A_i^T A_j.
Since Q^T Q = I, all off-diagonal blocks are exactly zero. ∎

For N=50 ≤ N_max=256: we need 200 orthogonal columns in R^{1024}. This is feasible
since 200 < 1024. ∎

---

## Theorem 2: Memory Bound for N=50 Adapters

**Statement.** For N=50 adapters, 28 layers (Qwen3-0.6B), applied to q_proj + v_proj,
stored in float16:

```
Memory = N × n_layers × Σ_{weight_type} (d_in × r + d_out × r) × sizeof(float16)
```

**Calculation (float16 = 2 bytes):**

| Weight type | d_in | d_out | A params | B params |
|-------------|------|-------|----------|----------|
| q_proj      | 1024 | 2048  | 4,096    | 8,192    |
| v_proj      | 1024 | 1024  | 4,096    | 4,096    |
| **Per layer** |    |       | 8,192    | 12,288   |

Total params per adapter: 28 × (8,192 + 12,288) = 28 × 20,480 = 573,440

For N=50 adapters: 50 × 573,440 = 28,672,000 params

Memory: 28,672,000 × 2 bytes = **57.3 MB** ≪ 5GB (K949 threshold).

**Even for all 7 weight types** (q, k, v, o, gate, up, down):
Adding k, o (d_out=1024): +2 × 28 × 50 × (4096 + 4096) × 2 = 45.9 MB
Adding gate, up (d_out=3072), down (d_in=3072 → A=12,288): +3 × 28 × 50 × ~16,384 × 2 ≈ 138 MB

Total for all 7 weight types: ~241 MB ≪ 5GB. ∎

---

## Predictions

| Prediction | Value | Source |
|-----------|-------|--------|
| max|A_i^T A_j|_F (N=50) | ≤ 1e-5 | Theorem 1 (QR precision) |
| Peak memory q+v only | 57.3 MB | Theorem 2 |
| Peak memory all 7 types | ~241 MB | Theorem 2 extended |
| N=50 feasible | Yes | N_max = 256 |
| Pairwise checks N(N-1)/2 | 1225 | N=50 |

---

## Kill Criteria

- **K948:** max|A_i^T A_j|_F < 1e-5 for all N(N-1)/2 = 1225 pairs at N=50
  - PASS: orthogonality holds at real scale (Theorem 1 confirmed)
  - FAIL: numerical precision issue at scale > N=2 (unexpected)
  
- **K949:** Total adapter memory < 5GB
  - PASS: memory well within limit (~57MB for q+v, ~241MB for all 7 types)
  - FAIL: unexpected memory overhead (would require >100x overhead vs theoretical)

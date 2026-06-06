# P5.A0: LoRI Sparse-Mask Composition — Mathematical Proof

## Reference: arXiv:2504.07448 (LoRI: Reducing Cross-Task Interference in Multi-Task LoRA)

## Theorem 1: Disjoint Sparse Masks Guarantee Zero Parameter Interference

**Setup.** Let W_base in R^{d_out x d_in} be a frozen weight matrix. For N adapters,
define LoRI adapters as:

  delta_W_i = B_i * diag(m_i) * A    for i = 1, ..., N

where:
- A in R^{r x d_in} is a FROZEN random projection (shared across all adapters)
- B_i in R^{d_out x r} is trainable per adapter
- m_i in {0, 1}^r is a binary mask (task-specific, non-overlapping)

**Claim.** If supp(m_i) ∩ supp(m_j) = {} for all i != j, then:

  <delta_W_i, delta_W_j>_F = 0

**Proof.**

  <delta_W_i, delta_W_j>_F = Tr(delta_W_i^T delta_W_j)
    = Tr((B_i diag(m_i) A)^T (B_j diag(m_j) A))
    = Tr(A^T diag(m_i) B_i^T B_j diag(m_j) A)

Since m_i and m_j have disjoint supports:
  diag(m_i) diag(m_j) = diag(m_i * m_j) = 0  (elementwise product of disjoint binary vectors)

Therefore:
  diag(m_i) B_i^T B_j diag(m_j)

The k-th row of diag(m_i) B_i^T is zero whenever m_i[k] = 0.
The l-th column of B_j diag(m_j) is zero whenever m_j[l] = 0.
Since supp(m_i) ∩ supp(m_j) = {}, for every non-zero row k (m_i[k]=1),
the corresponding column k in diag(m_j) is zero (m_j[k]=0).

Thus diag(m_i) (B_i^T B_j) diag(m_j) = 0, and <delta_W_i, delta_W_j>_F = 0.  QED.

**Capacity bound.** With rank r, we can support at most N_max = r non-overlapping
masks (one dimension per adapter). At r=6: N_max = 6. For N > r, masks must be
constructed via sparse coding with overlap budget epsilon.

At r=16 (rank-16 adapters): N_max = 16 disjoint slots.
At r=32 with 50% sparsity (each mask uses r/2 dims): N_max = 2*32/16 = 4 fully disjoint.
With structured overlap: N_max ~ r / (sparsity * r) = 1/sparsity.

## Theorem 2: Frozen A as JL-Projection Preserves Task Structure

**Statement (Johnson-Lindenstrauss).** For N task-specific B-matrices in R^{d_out x s}
(where s = |supp(m_i)|), a random Gaussian projection A in R^{r x d_in} with
r >= 4 ln(N) / (epsilon^2/2 - epsilon^3/3) preserves pairwise distances with
probability >= 1 - delta.

**Consequence.** Freezing A does not lose expressiveness — it's a random projection
that preserves the essential structure of the task gradient. Only B needs to adapt.

**Prediction.** At N=5, epsilon=0.1: r_JL = ceil(4*ln(5) / 0.00467) = 138.
Our r=6 is below JL bound, but Finding #440 showed Grassmannian composition works
at r=4 with N=100. The practical question is whether sparse B at r=6 has enough
capacity. Kill criterion K2 tests this.

## Theorem 3: Impossibility of Interference Under Mask Disjointness

**Statement.** Under disjoint masks, NO training procedure can create parameter-space
interference. This is a STRUCTURAL guarantee, not a training-time one.

**Proof.** The mask m_i is applied AFTER training: delta_W_i = B_i * diag(m_i) * A.
Even if B_i has arbitrary values, the mask zeroes out all dimensions outside supp(m_i).
Since supp(m_i) ∩ supp(m_j) = {}, the masked adapters operate in strictly disjoint
subspaces of the rank-r intermediate space. No gradient, no weight update, no
numerical error can violate this — it's enforced by multiplication with zeros.  QED.

**Comparison to Grassmannian:** Grassmannian A-matrices achieve orthogonality at
float64 precision (~1e-16). LoRI masks achieve EXACT orthogonality (true zero,
not approximate). The tradeoff: LoRI has lower capacity per adapter (s < r dims
vs all r dims), but stronger guarantee.

## Kill Criteria Derivation

| K | Criterion | Theorem | Prediction |
|---|-----------|---------|-----------|
| K1 | max\|cos\| < 1e-4 | Theorem 1+3 | = 0.0 exactly (mask disjointness) |
| K2 | Quality >= 90% of LoRA | Theorem 2 | Depends on effective rank s per adapter |
| K3 | 5-adapter composition < 5pp degradation | Theorem 3 | 0pp (structural guarantee) |

## Implementation Notes

Port from LoRI paper's approach:
1. Initialize A as random Gaussian (frozen, shared)
2. Assign each adapter a disjoint mask slice: adapter_i gets dims [i*s : (i+1)*s]
3. Train B_i on domain data with mask applied (only s columns of B are non-zero)
4. Compose: W_merged = W_base + sum_i scale * B_i * diag(m_i) * A

For Gemma 4 E4B on MLX:
- Use QuantizedLinear dimension extraction (proven in C1.1v2)
- Apply mask after each optimizer step: B[:, ~mask] = 0
- Or more efficiently: only allocate and train the s active columns of B

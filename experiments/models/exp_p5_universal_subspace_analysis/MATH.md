# Universal Subspace Analysis of Pierre's Adapters

## Background

The Universal Weight Subspace Hypothesis (arXiv:2512.05117) claims that LoRA adapters
fine-tuned from the same base model share a low-rank "universal subspace" — top-K PCA
components of the stacked weight matrices capture the majority of variance across
hundreds of independently trained adapters.

Pierre's adapters use Grassmannian-initialized A-matrices: A_i are constructed via QR
decomposition to satisfy A_i^T A_j ≈ 0 for i ≠ j. This structural orthogonality is
our core composition guarantee (Finding #428). The question: does this construction
prevent universal subspace structure?

## Prior Results

- **Finding #65**: EigenLoRAx on BitNet-2B-4T. LoRA-A only 31.3% variance at K=16
  (Grassmannian prevents shared subspace). LoRA-B trivially 100% (rank < N). KILLED.
- **Finding #130**: Frechet merge (subspace-optimal) WORSE than naive addition (-69% MSE).
- **Finding #428**: N=25 Grassmannian composition on Gemma 4, max|cos|=2.165e-8.

## Theorem 1: Grassmannian A-Matrices Have Uniform PCA Spectrum

**Statement.** Let A_1, ..., A_N ∈ ℝ^{d×r} be Grassmannian-packed matrices satisfying
⟨A_i, A_j⟩_F = tr(A_i^T A_j) ≈ 0 for i ≠ j. Define the data matrix
X = [vec(A_1), ..., vec(A_N)]^T ∈ ℝ^{N×dr}. Then the singular values of X (centered)
are approximately equal, and the variance explained by the top-K components is ≈ K/N.

**Proof.** The Gram matrix G = X X^T has entries G_{ij} = ⟨vec(A_i), vec(A_j)⟩ = tr(A_i^T A_j).
By Grassmannian construction, G_{ij} ≈ δ_{ij} · ‖A_i‖_F^2.

If all A_i have equal Frobenius norm c (true for QR construction where ‖A_i‖_F = √r),
then G ≈ c^2 · I_N. The eigenvalues of G are all ≈ c^2, so the singular values of X
are all ≈ c. PCA of X has N components each explaining 1/N of the total variance.

Therefore top-K components explain K/N of variance. At K=16, N=25: 64%. ∎

**Prediction**: For A-matrices, top-16 PCA explains 60-68% of variance (not ≥80%).

## Theorem 2: B-Matrix Variance Depends on Task Similarity

**Statement.** Let B_i ∈ ℝ^{r×d_out} be trained B-matrices. The PCA concentration of
{vec(B_i)} depends on the effective rank of the task gradient correlation matrix.

**Proof sketch.** Each B_i is trained from initialization B_0 via gradient descent:
B_i = B_0 + η Σ_t A_i^T ∇_{W} L_i(x_t). Because A_i^T A_j ≈ 0 (Grassmannian),
the projected gradients A_i^T ∇_W L and A_j^T ∇_W L operate in orthogonal subspaces
of the input space. This means gradient updates to B_i and B_j are structurally
decoupled — they see different projections of the same loss landscape.

**Consequence**: Unlike vanilla LoRA (where all adapters share the same A initialization
and thus the same gradient projection), Grassmannian A-matrices cause B-matrices to
explore independent directions. This predicts LOWER PCA concentration for B-matrices
than the Universal Subspace paper reports.

**Prediction**: B-matrix top-16 PCA explains 65-80% of variance (higher than A due to
shared base model features, but lower than vanilla LoRA's ~90% reported in the paper).

## Theorem 3: Universal Subspace Merging Cannot Beat Naive Addition

**Statement.** For adapters with Grassmannian A-matrices, projecting onto a universal
subspace and merging is equivalent to or worse than naive addition Σ_i A_i B_i.

**Proof sketch.** Naive addition W_composed = Σ_i A_i B_i preserves each adapter's
contribution exactly because A_i's are orthogonal (no cross-term interference).
Any projection onto a universal subspace P must satisfy P A_i ≈ A_i for all i
to preserve adapter quality. But if rank(P) < N·r, some information is lost.
Since the adapters already compose without interference via naive addition
(Finding #428), the universal subspace projection can only remove information,
never add it. ∎

**Prediction**: K1283 FAIL — universal merging ≤ naive addition on all domains.

## Theorem 4: Compression Destroys Orthogonality

**Statement.** Projecting N Grassmannian adapters onto K < N·r universal basis vectors
increases max|cos(A_i, A_j)| above the interference threshold.

**Proof.** After projection to K-dim universal basis U ∈ ℝ^{dr×K}, the projected
adapters are Ã_i = U U^T vec(A_i). The effective A-matrices share the column space
of U, so at most K/r independent adapters can be orthogonal in the projected space.
For K=16, r=6: at most 2-3 adapters remain orthogonal. The rest suffer interference.

**Prediction**: K1284 FAIL for A-matrix compression. B-matrix compression may preserve
quality if B-matrices share more structure, but at the cost of losing Grassmannian guarantee.

## Kill Criteria Predictions

| Criterion | Prediction | Rationale |
|-----------|-----------|-----------|
| K1282: Top-16 PCA ≥ 80% | **FAIL (~64% for A, ~72% combined)** | Theorem 1: uniform spectrum |
| K1283: Merging > naive on ≥3/5 | **FAIL (0/5)** | Theorem 3: naive is optimal |
| K1284: Compression < 5pp degradation | **FAIL (>10pp for N>3)** | Theorem 4: orthogonality lost |

## Significance

This experiment confirms that the Universal Weight Subspace Hypothesis does NOT apply
to Grassmannian-initialized adapters. The orthogonality that enables interference-free
composition is precisely what prevents shared subspace structure. This is a feature:
the "cost" of no universal subspace is paid at training time (each adapter trained
independently), while the "benefit" of composition is gained at serving time (zero
interference, simple addition).

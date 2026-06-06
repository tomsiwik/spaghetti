# Grassmannian LoRA-A Initialization: QR Orthogonality Proof

## A. Failure Mode Identification

**Failure mode:** LoRA adapters trained on different domains develop correlated
weight updates, causing interference during additive composition. Specifically,
if two adapters share column space in their A-matrices (the "down-projection"
from hidden dim d to rank r), their composed weight updates
Delta W_i = B_i A_i^T and Delta W_j = B_j A_j^T will interfere:

||Delta W_i^T Delta W_j|| = ||A_i B_i^T B_j A_j^T|| <= ||B_i|| ||A_i^T A_j|| ||B_j||

When A_i and A_j are initialized randomly and independently, E[|cos(vec(A_i), vec(A_j))|]
is O(1/sqrt(dr)) which is small but nonzero for finite d, and training can push
them closer. More critically, standard Kaiming/Gaussian initialization provides
no STRUCTURAL guarantee of orthogonality.

**Root cause:** Random initialization is probabilistically near-orthogonal at
high d, but offers no deterministic guarantee. The disease is not "bad luck in
initialization" but rather the absence of a structural constraint that makes
correlation impossible.

## B. The Right Question (Reframe)

**Wrong question:** "How do we reduce adapter interference after training?"

**Right question:** "What initialization of A-matrices makes column-space
overlap EXACTLY ZERO by construction, for up to N domains?"

**Answer:** This is a classical problem in linear algebra. Given a d-dimensional
space and N rank-r subspaces, we need N*r mutually orthonormal vectors. This is
achievable whenever N*r <= d, via the Gram-Schmidt process or its numerically
stable equivalent, QR decomposition.

## C. Prior Mathematical Foundations

### QR Decomposition (Householder, 1958)

For any matrix M in R^{d x m} with m <= d, the thin QR decomposition yields
M = QR where Q in R^{d x m} has orthonormal columns (Q^T Q = I_m) and
R in R^{m x m} is upper triangular.

**Theorem (Householder 1958):** The QR decomposition exists for any matrix
and the Q factor has exact machine-precision orthonormality when computed via
Householder reflections (as implemented in LAPACK/numpy).

### Grassmannian Packing (Conway, Hardin, Sloane 1996)

The Grassmannian manifold Gr(r, d) is the space of all r-dimensional subspaces
of R^d. The optimal packing problem on Gr(r, d) asks: given N points, how to
arrange them to maximize the minimum chordal distance?

For our application, we need a weaker result: we only need N subspaces with
ZERO mutual coherence (exact orthogonality), not optimal packing. This is
trivially achievable when N*r <= d because R^d has enough dimensions to
accommodate N mutually orthogonal r-planes.

**Capacity bound (folklore):** The maximum number of mutually orthogonal
r-dimensional subspaces in R^d is floor(d/r).

### Johnson-Lindenstrauss Lemma (JL, 1984)

For random Gaussian matrices G in R^{d x r}, the inner product between
two independent random projections concentrates: for any fixed vectors u, v,
|<Gu, Gv>| = O(1/sqrt(r)) with high probability. This provides a
PROBABILISTIC guarantee for random initialization but not an exact one.

The QR approach is STRICTLY STRONGER: it gives |<q_i, q_j>| = 0 exactly
(to machine precision), not approximately.

### Prior Finding #132 (This Project)

"Grassmannian AP skeleton reduces interference." Alternating Projection on
Gr(r, d) produces orthonormal A-matrices. Confirmed on BitNet-2B-4T (d=2560).
B-matrix cosine 0.0298 maps to delta cosine 0.0017 (17x decorrelation filter).

### Prior Finding #317 (This Project)

"Qwen3-4B-4bit validated as Pierre Pro base on M5 Pro 48GB." Same d=2560 as
BitNet-2B-4T. All Grassmannian machinery transfers.

## D. Proof of Guarantee

**Theorem 1 (QR Orthogonality).** Let d, r, N be positive integers with
N*r <= d. Construct matrix M in R^{d x Nr} with i.i.d. Gaussian entries.
Compute the thin QR decomposition M = QR. Partition Q into N blocks of r
columns each: A_i = Q[:, (i-1)*r : i*r] for i = 1, ..., N. Then for all
i != j:

  A_i^T A_j = 0_{r x r}   (exactly, to machine precision)

*Proof.*

By construction of QR decomposition, Q has orthonormal columns:

  Q^T Q = I_{Nr}

This means for any two columns q_a and q_b with a != b:

  q_a^T q_b = 0

Now consider blocks A_i = [q_{(i-1)r+1}, ..., q_{ir}] and
A_j = [q_{(j-1)r+1}, ..., q_{jr}] with i != j.

The (p, q)-th entry of A_i^T A_j is:

  (A_i^T A_j)_{p,q} = q_{(i-1)r+p}^T q_{(j-1)r+q}

Since i != j, the indices (i-1)r+p and (j-1)r+q are distinct (they belong
to non-overlapping blocks). By the orthonormality of Q, this equals zero.

Therefore A_i^T A_j = 0 for all i != j. QED.

**Corollary 1.1 (Flattened Cosine).** Under the conditions of Theorem 1,
the cosine similarity between flattened A-matrices is:

  cos(vec(A_i), vec(A_j)) = tr(A_i^T A_j) / (||A_i||_F ||A_j||_F)
                           = 0 / (sqrt(r) * sqrt(r))
                           = 0

*Proof.* Since A_i has r orthonormal columns, ||A_i||_F = sqrt(r). The
inner product <vec(A_i), vec(A_j)> = tr(A_i^T A_j) = sum of diagonal
entries of the zero matrix = 0. QED.

**Remark:** The cosine between FLATTENED A-matrices (as computed by the
experiment) equals tr(A_i^T A_j) / (||A_i||_F * ||A_j||_F). Since
A_i^T A_j = 0, this is exactly zero. The experiment measures
|sum(A_i * A_j)| / (||A_i|| * ||A_j||) which is the same quantity.

**Theorem 2 (Capacity).** The maximum number of mutually orthogonal
r-dimensional subspaces of R^d is exactly floor(d/r).

*Proof.* Each subspace occupies r dimensions. Since the subspaces are
mutually orthogonal, their direct sum must be a subspace of R^d:

  dim(span(A_1) + span(A_2) + ... + span(A_N)) = N*r <= d

Therefore N <= floor(d/r). This bound is tight: the construction in
Theorem 1 achieves it. QED.

**Theorem 3 (GQA Invariance).** Orthogonality of A-matrices depends only on
the INPUT dimension of the linear projection, not on the output dimension or
attention pattern. For Grouped Query Attention (GQA) with fewer KV heads, the
A-matrix dimension is determined by hidden_dim (the input to q/k/v projections),
not by n_heads * head_dim (the output).

*Proof.* A LoRA adapter for a linear layer W in R^{out x in} has
A in R^{in x r} and B in R^{out x r}. The A-matrix projects from the
input space R^{in}. For all attention projections (q, k, v, o), the input
is the hidden state h in R^{hidden_dim}. Therefore in_features = hidden_dim
for q_proj, k_proj, v_proj. For o_proj, in_features = n_heads * head_dim.

In GQA, k_proj and v_proj have FEWER OUTPUT dimensions
(n_kv_heads * head_dim < n_heads * head_dim), but their INPUT dimension
remains hidden_dim. Since the A-matrix operates in the input space, GQA
does not reduce the orthogonal capacity.

Specifically for Qwen3-4B: hidden_dim = 2560 for all projections (confirmed
by Finding #317). Therefore N_max = 2560/16 = 160 for all attention modules,
regardless of whether they use GQA or MHA. QED.

## D. Quantitative Predictions

| Prediction | Source | Expected Value |
|-----------|--------|---------------|
| P1: Pairwise cos at N=5, all modules | Theorem 1, Corollary 1.1 | 0.0 exactly (< 1e-6 numerical noise) |
| P2: Pairwise cos at N=24, d=2560 modules | Theorem 1 (24*16=384 << 2560) | 0.0 exactly (< 1e-6) |
| P3: N_max for d=2560, r=16 | Theorem 2 | 160 domains |
| P4: GQA modules have same N_max as MHA | Theorem 3 | 160 (same for all 7 module types) |
| P5: Initialization time (36 layers x 7 modules x QR) | O(d * (Nr)^2) per QR | < 10s on M5 Pro |
| P6: in_features uniform across module types | Theorem 3 (Qwen3 arch) | 2560 for all except possibly o_proj |

**Kill criteria derivation:**
- K810: Theorem 1 predicts cos = 0.0 exactly. Threshold 0.05 is 50,000x above
  predicted value. PASS is guaranteed by the proof unless QR implementation is
  broken. Anything > 1e-6 indicates a BUG, not a fundamental failure.
- K811: QR on R^{2560 x 80} takes ~0.5ms. 36 layers x 7 modules = 252 QR
  calls. Total << 1s. Threshold 60s is 60x+ above expected. PASS is trivially
  guaranteed.

## E. Assumptions & Breaking Conditions

1. **A1: N*r <= d for all modules.** If any module has in_features < N*r, QR
   cannot produce N orthogonal subspaces in that module. Breaking: overflow
   modules get random (non-orthogonal) initialization.
   - N=5: 5*16=80 << 2560. Safe with 32x margin.
   - N=24: 24*16=384 << 2560. Safe with 6.7x margin.

2. **A2: NumPy QR uses Householder reflections.** If the implementation used
   modified Gram-Schmidt, numerical errors accumulate. NumPy/LAPACK uses
   Householder (confirmed in NumPy docs), so machine-precision orthogonality
   is guaranteed.

3. **A3: in_features = hidden_dim for all target modules.** If some module has
   a different input dimension (e.g., o_proj input = num_heads * head_dim), the
   capacity for that module differs. The experiment detects this automatically.

4. **A4: float32 precision suffices.** At float32, machine epsilon ~ 1e-7.
   QR orthogonality error is O(epsilon * cond(M)) where cond(M) ~ sqrt(d/Nr)
   for Gaussian M. At d=2560, Nr=384: error ~ 1e-7 * 2.6 ~ 3e-7.

## F. Worked Example (d=8, r=2, N=3)

Generate M in R^{8 x 6} with Gaussian entries, compute QR:

```
M = randn(8, 6)  # random
Q, R = qr(M)     # Q is 8x6 with orthonormal columns

A_0 = Q[:, 0:2]  # first domain
A_1 = Q[:, 2:4]  # second domain
A_2 = Q[:, 4:6]  # third domain

# Verify:
A_0^T A_1 = [[0, 0], [0, 0]]  # exactly zero
A_0^T A_2 = [[0, 0], [0, 0]]  # exactly zero
A_1^T A_2 = [[0, 0], [0, 0]]  # exactly zero

# Flattened cosine:
cos(vec(A_0), vec(A_1)) = tr(A_0^T A_1) / (||A_0||_F * ||A_1||_F)
                        = 0 / (sqrt(2) * sqrt(2)) = 0

# Capacity: floor(8/2) = 4 domains max
# We used 3, so 3*2=6 <= 8. Margin: 1 more domain possible.
```

## G. Complexity & Architecture Connection

**Initialization complexity:**
- Per module per layer: QR decomposition of R^{d x Nr}
- QR via Householder: O(d * (Nr)^2) flops
- Total: n_layers * n_modules * O(d * (Nr)^2)
- For Qwen3-4B (d=2560, N=24, r=16, 36 layers, 7 modules):
  252 * O(2560 * 384^2) ~ 252 * 3.8e8 ~ 9.5e10 flops
  At ~1 TFLOP/s for numpy on M5 Pro: ~0.1s

**Storage:**
- Per skeleton key: d * r * 4 bytes (float32)
- N=5: 36 * 7 * 5 = 1260 keys, each 2560*16*4 = 160KB = 201 MB total
- N=24: 36 * 7 * 24 = 6048 keys = 968 MB total
- Compressed (.npz): ~50% smaller

**Interaction with model:**
- A-matrices are FROZEN during training. Only B-matrices learn.
- The skeleton is architecture-agnostic: it depends only on (d, r, N, n_layers, module_keys).
- Since Qwen3-4B has the same hidden_dim=2560 as BitNet-2B-4T, the same
  skeleton can be shared across base models (modulo n_layers).

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   QR decomposition produces exactly orthonormal columns, making A_i^T A_j = 0
   by construction for columns from different blocks.

2. **Which existing theorem(s) does the proof build on?**
   Householder QR decomposition (Householder 1958, as implemented in LAPACK/NumPy).
   Grassmannian capacity = floor(d/r) (standard linear algebra).

3. **What specific numbers does the proof predict?**
   cos = 0.0 exactly (< 1e-6 numerical noise) for all pairs at N=5 and N=24.
   N_max = 160 for all module types (d=2560, r=16).
   Init time < 1s.

4. **What would FALSIFY the proof (not just the experiment)?**
   The proof is wrong if: QR decomposition does NOT produce orthonormal columns
   (would indicate a NumPy/LAPACK bug), or if in_features varies across module
   types in a way that reduces capacity below N=24.

5. **How many hyperparameters does this approach add?**
   Count: 0. The skeleton is fully determined by (d, r, N, n_layers, module_keys),
   all of which are fixed by the model architecture and desired number of domains.

6. **Hack check:** No. This is a single construction (QR) that provides a single
   guarantee (exact orthogonality). No stacking of fixes.

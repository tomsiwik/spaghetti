# MATH: PiSSA SVD-Init vs Grassmannian Init for LoRA

## 1. Mechanism Definition

### PiSSA Initialization (arxiv 2404.02948)

Given a pretrained weight matrix W in R^{d_out x d_in}, PiSSA computes its
truncated SVD:

  W = U @ Sigma @ V^T

where U in R^{d_out x d_out}, Sigma in R^{d_out x d_in}, V in R^{d_in x d_in}.

PiSSA initializes LoRA as:
  A = V[:, :r]^T  (first r right singular vectors, transposed to R^{r x d_in})
  B = U[:, :r] @ Sigma[:r, :r]  (left singular vectors scaled by singular values, R^{d_out x r})

In our convention (x @ A) @ B:
  A_pissa = V[:, :r]  in R^{d_in x r}  (top-r right singular vectors)
  B_pissa = (Sigma[:r, :r] @ U[:, :r]^T)^T = U[:, :r] @ Sigma[:r, :r]  in R^{r x d_out}

The residual stored in the model: W_res = W - B_pissa @ A_pissa^T

At initialization: W_res + B_pissa @ A_pissa^T = W (exact reconstruction of
the rank-r principal component). Training then fine-tunes in the principal subspace.

### Grassmannian Initialization (current approach)

N orthonormal A matrices A_1, ..., A_N in R^{d_in x r} are pre-computed via
Alternating Projection on Gr(r, d_in) such that:

  A_i^T A_j = 0_{r x r}  for all i != j  (when N*r <= d_in)

Each A_i is frozen. B_i in R^{r x d_out} is initialized to zero and trained.
The LoRA output: delta_y = (x @ A_i) @ ternary(B_i) * scale

### The Fundamental Conflict

**PiSSA gives the SAME A for all adapters on the same weight matrix.**

For weight matrix W_k (e.g., layer 5's q_proj), the SVD is unique (up to sign):
  W_k = U_k @ Sigma_k @ V_k^T

All adapters i targeting W_k get: A_i = V_k[:, :r].
Therefore: A_i^T A_j = V_k[:, :r]^T V_k[:, :r] = I_r for ALL i,j.

This means cos(A_i, A_j) = 1.0 -- maximum possible alignment. The Grassmannian
orthogonality guarantee is completely destroyed.

**PiSSA-frozen-A + multiple adapters per weight matrix is mathematically
incompatible with orthogonal composition.**

### PiSSA-unfrozen: What Happens to Orthogonality

When A is trainable, each adapter's A_i starts at V_k[:, :r] but drifts during
training on domain-specific data. The question: does domain-specific gradient
signal push A matrices apart? Training gradient on A:

  dL/dA = x^T @ (dL/dy @ B^T)

Different domains produce different x distributions and different loss gradients,
so A matrices will diverge. The extent depends on:
  1. Learning rate and number of steps
  2. Diversity of domain data
  3. Magnitude of B (which scales the A gradient through the chain rule)

## 2. Why PiSSA Works (Single Adapter)

PiSSA captures the principal variance of W in the LoRA factorization. The rank-r
SVD approximation minimizes the Frobenius norm error:

  ||W - U_r Sigma_r V_r^T||_F = sqrt(sum_{i=r+1}^{min(d_out,d_in)} sigma_i^2)

By initializing LoRA in this principal subspace, the adapter starts "aware" of
the weight structure rather than in a random direction. The residual W_res
contains the remaining (d-r) singular directions.

**Convergence benefit:** Standard LoRA starts at zero delta (A random, B=0).
PiSSA starts at the rank-r approximation of W. The optimization landscape is
pre-aligned with the dominant gradient directions of the loss.

Fang et al. (2404.02948) report 10-20% faster convergence and 1-3% better
final quality on LLaMA-2-7B across several benchmarks.

## 3. What Breaks It

### 3a. Ternary Weight SVD Degeneracy

BitNet-2B weights are ternary: W in {-1, 0, +1}^{d_out x d_in}. The SVD of
ternary matrices has distinctive properties:

- Singular values are more uniformly distributed (less spectral concentration)
- For a random ternary matrix with p(0) ~ 0.42: sigma_max ~ sqrt(d * (1-p_0))
- The rank-r approximation captures LESS variance than for float weights
- Specifically: if singular values are nearly flat, then
  sum_{i=1}^r sigma_i^2 / sum_all sigma_i^2 ~ r / rank(W)

This means PiSSA's advantage (capturing principal variance) is diminished for
ternary weights. The principal subspace is less "principal."

### 3b. Frozen-A Composition Failure

With frozen A from PiSSA, all N adapters share the same A matrix per weight.
The interference bound becomes:

  ||Delta_W_i^T Delta_W_j|| = (scale/r)^2 * ||B_i|| * ||A_i^T A_j|| * ||B_j||
                             = (scale/r)^2 * ||B_i|| * ||I_r||_F * ||B_j||
                             = (scale/r)^2 * sqrt(r) * ||B_i|| * ||B_j||

Compare to Grassmannian: ||A_i^T A_j|| = 0, so interference = 0.

**Kill criterion K2 directly follows:** PiSSA-frozen cosine = 1.0 >> 0.1 threshold.

### 3c. Unfrozen-A Optimization Landscape

When A is unfrozen, we lose the Grassmannian guarantee but gain:
- Per-adapter principal subspace alignment
- Faster single-adapter convergence
- But no composition guarantee

The composition ratio depends on how much A matrices diverge during training.
If they remain close to the shared SVD init (likely for short training), composition
will be severely degraded.

## 4. Assumptions

1. **SVD of ternary weights is meaningful.** Justified: any real matrix has an SVD.
   Risk: if eigenvalue spectrum is flat, the "principal" subspace captures minimal
   extra variance vs random. Tested in this experiment.

2. **200 training steps sufficient for A divergence.** Justified: prior experiments
   show significant B-matrix training in 200 steps. Risk: A matrices may not
   diverge enough in 200 steps if B gradients dominate. Tested.

3. **Character-level toy data is directionally valid.** The experiment tests the
   MECHANISM (SVD init quality, orthogonality preservation) not absolute quality.
   Justified by all prior micro experiments using same setup.

## 5. Complexity Analysis

SVD computation per weight matrix: O(d_in * d_out * min(d_in, d_out)).
At d=64: O(64^3) = O(262K) -- negligible.
At d=2560: O(2560^3) ~ O(16.7B) -- about 10-30 seconds per matrix, with 210
projections = 35-100 minutes total. This is a one-time cost.

Grassmannian AP at N*r <= d: O(N * d * r) for QR -- negligible.

Runtime cost of PiSSA init is dominated by SVD of all weight matrices.

## 6. Worked Example (d=64, r=8, N=4)

### Grassmannian Init:
- Generate 4 orthonormal frames in R^{64 x 8} via QR.
- Since N*r = 32 <= 64 = d, perfect orthogonality achieved.
- A_i^T A_j = 0 for all i != j.
- After training B_i, interference bound = 0.

### PiSSA Init:
- For weight W in R^{64 x 64} (ternary):
  - Compute SVD: W = U Sigma V^T
  - All 4 adapters get A = V[:, :8] (identical!)
  - A_i^T A_j = I_8 for all i,j.
  - B_i initialized to Sigma[:8,:8] @ U[:,:8]^T (all identical!)
- After training: B matrices diverge, A matrices stay identical (frozen) or
  slowly diverge (unfrozen).
- Interference: (scale/r)^2 * sqrt(8) * ||B_i|| * ||B_j|| >> 0.

### Expected Variance Capture:
For random ternary W (42% zeros, 29% each for +1,-1):
  E[sigma_1^2] ~ d*(1-p_0) = 64*0.58 = 37.12
  E[sum sigma_i^2] = ||W||_F^2 ~ d*d*(1-p_0) = 2375
  Rank-8 capture ~ 8/64 * 2375 = 297 (if flat) vs 37.12*8 = 297 (consistent)
  Fraction captured ~ 8/64 = 12.5% (if flat)

For float weights with power-law decay sigma_i ~ i^{-alpha}:
  Rank-8 capture is much higher (e.g., 40-60% for alpha~0.5).

This confirms the ternary SVD degeneracy concern: PiSSA captures far less
variance on ternary weights than on float weights.

## 7. Connection to Architecture

This experiment tests whether the initialization strategy for A matrices matters
for the Grassmannian skeleton. The result informs:

- If PiSSA-frozen fails (as predicted by the math): confirms Grassmannian
  orthogonality is the correct choice, and the init strategy is settled.
- If PiSSA-unfrozen works despite non-orthogonality: suggests the constructive
  transfer mechanism (already observed in OSRM findings) may be more important
  than orthogonality at small N.
- Either way: helps decide whether to pursue data-aware init strategies vs
  purely geometric init.

Reference: DeepSeek-V3 uses random init for expert gating with auxiliary-loss-free
balancing. Qwen3 uses standard random LoRA init. No production model uses
SVD-based expert initialization, suggesting the mechanism has not been validated
at scale.

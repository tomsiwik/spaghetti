# LoRA Merging Bakeoff: Mathematical Foundations

## Notation

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| W_0 | (d, d') | Frozen base weight matrix |
| dW_k | (d, d') | LoRA delta for domain k: (alpha/r) * A_k @ B_k |
| A_k | (d, r) | LoRA down-projection for domain k |
| B_k | (r, d') | LoRA up-projection for domain k |
| r | scalar | LoRA rank |
| alpha | scalar | LoRA scaling factor |
| N | scalar | Number of domains |
| tau_k | (D,) | Flattened task vector for domain k (all deltas concatenated) |
| D | scalar | Total number of delta parameters (sum of all d*d' across layers) |

## Method Definitions

### 1. Simple Average (Task Arithmetic)

The merged delta is the element-wise mean of all task vectors:

    tau_merged = (1/N) * sum_{k=1}^{N} tau_k

Applied to weights: W_merged = W_0 + tau_merged

**Assumptions**: Linearity of weight space. Works well when task vectors
are near-orthogonal (our cos ~ 0.014 at N=2).

**Complexity**: O(N * D) additions, O(D) divisions. Negligible.

### 2. TIES-Merging (Trim, Elect Sign, Merge)

**Step 1: TRIM**. For each task vector tau_k, keep only the top rho%
parameters by magnitude:

    tau_k^trim[j] = tau_k[j]   if |tau_k[j]| >= threshold_k(rho)
                    0           otherwise

where threshold_k(rho) is the (1-rho)-quantile of |tau_k|.

**Step 2: ELECT SIGN**. For each parameter position j, elect the
dominant sign by majority:

    gamma[j] = sign( sum_{k=1}^{N} tau_k^trim[j] )

**Step 3: DISJOINT MERGE**. Average only the trimmed values matching
the elected sign:

    tau_merged[j] = sum_{k: sign(tau_k^trim[j]) = gamma[j]} tau_k^trim[j]
                    / |{k: sign(tau_k^trim[j]) = gamma[j]}|

**Complexity**: O(N * D * log D) for sorting (trim step), O(N * D) for
elect and merge.

**Hyperparameter**: density rho in [0, 1]. We use rho = 0.2 (keep top 20%).

### 3. DARE (Drop And REscale)

For each task vector tau_k, randomly drop a fraction p of parameters,
then rescale:

    m_k[j] ~ Bernoulli(1 - p)     (keep with probability 1-p)
    tau_k^dare[j] = tau_k[j] * m_k[j] / (1 - p)

Then merge by simple average:

    tau_merged = (1/N) * sum_{k=1}^{N} tau_k^dare

**Key property**: E[tau_k^dare] = tau_k (unbiased estimator).

**Complexity**: O(N * D) random draws + O(N * D) multiplications.

**Hyperparameter**: drop rate p in [0, 1). We use p = 0.9 (drop 90%).

### 4. DARE-TIES

Apply DARE sparsification first, then TIES sign election and merge
(skipping the TIES trim step since DARE already handles sparsification):

    1. tau_k^dare = Drop-and-Rescale(tau_k, p)
    2. gamma[j] = sign( sum_k tau_k^dare[j] )
    3. tau_merged[j] = mean of tau_k^dare[j] where sign matches gamma[j]

### 5. Concat + Calibrate (Our Method)

Instead of merging into a single delta, keep all N deltas as separate
experts and learn a router:

    h_merged = sum_{k in top-K} w_k * MLP(h; W_0 + dW_k)

where w_k = softmax(h @ W_router^T)[k] masked to top-K experts.

Router W_router is trained on mixed-domain data (100 steps).

**Complexity**: O(K * cost(MLP)) inference + O(D_router * cal_steps) calibration.
D_router = d * N per layer.

## Worked Example (d=64, r=8, N=2)

- Per-layer delta: A (64, 8) @ B (8, 256) = dW (64, 256). 16,384 params per layer-sublayer.
- 4 layers x 2 sublayers = 8 delta matrices. Total D = 8 * 16,384 = 131,072.
- Simple avg: 131,072 additions + 131,072 divisions. ~microseconds.
- TIES (rho=0.2): keep 26,214 params per task, sort 131,072 values per task. ~milliseconds.
- DARE (p=0.9): 131,072 random draws per task, rescale by 10x. ~milliseconds.
- Concat+cal: 100 steps x 32 batch x forward+backward. ~seconds.

## Theoretical Analysis: Why Simple Average Dominates

When deltas are near-orthogonal (cos(tau_i, tau_j) ~ 0):

1. **Sign conflicts are rare**. With orthogonal deltas, parameter j being
   positive in tau_i says almost nothing about its sign in tau_j. TIES's
   sign election is solving a problem that barely exists.

2. **Trimming destroys signal**. At rho=0.2, TIES discards 80% of each
   delta. When deltas are small and distributed (typical of LoRA with
   low rank), trimming removes genuine signal, not noise.

3. **DARE's high drop rate amplifies noise**. At p=0.9, only 10% of
   parameters survive, rescaled 10x. This amplifies any sampling noise
   in the surviving parameters. With already-small LoRA deltas, this is
   pure noise amplification.

4. **Simple average is the optimal linear estimator** when vectors are
   orthogonal. The mean minimizes MSE over all affine combinations.

The concat+calibrate method can outperform averaging because it preserves
all information (no merging) and routes per-token to the relevant expert.
But this comes at the cost of N-fold inference compute and calibration data.

## Connection to Prior Findings

- **Natural orthogonality**: cos(tau_A, tau_B) ~ 0.014 at N=2 (confirmed
  in lora_procrustes experiment). This makes sign conflicts nearly
  non-existent, explaining why TIES adds noise rather than resolving
  interference.

- **Dead capsule pruning analog**: DARE's random dropping is analogous to
  our dead capsule pruning, but DARE drops randomly while our method
  profiles actual activation patterns. Prior work (Exp 10) showed random
  pruning is competitive but not superior to targeted pruning.

- **Composition gap**: The +1.08% gap of concat+cal at N=2 is consistent
  with the +0.8% measured in the lora_procrustes experiment. The function-
  space gap is real but small at N=2.

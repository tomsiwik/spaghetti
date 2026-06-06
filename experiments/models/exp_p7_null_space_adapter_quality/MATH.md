# P7.A1: Adapter Restricted to Null Space -- Quality Preserved?

## Type: Verification

## Motivation

Finding #493 (P7.A0) confirmed that Gemma 4 E4B local-attention v_proj has 2048
structural null dimensions per layer (from dimension mismatch 2560 - 512 = 2048).
This gives 341 adapter slots at r=6 -- 13x our 25-domain target.

The question: does restricting an adapter to live in null(W_v) degrade its ability
to learn useful domain features? If quality is preserved, null-space projection
gives us **zero-interference composition by construction**.

## Prior Work

- **Null-LoRA** (arXiv:2512.15233): Projects LoRA updates into null(W) to preserve
  base model capabilities. Reports <1% accuracy loss on multiple benchmarks.
- **Finding #493**: v_proj local layers have null_dim = 2048, effective_rank = 512
  (full rank at epsilon=1e-4). Null space is structural, not from quantization.
- **Finding #492**: Module-disjoint LoRA fails; same-module composition needed.

## Mathematical Framework

### Setup

Let W_v in R^{d_out x d_in} be the v_proj weight matrix at a local layer.
- d_out = 512 (GQA: fewer value heads than query heads)
- d_in = 2560 (hidden size)
- rank(W_v) = d_out = 512 (full rank, confirmed by P7.A0)
- dim(null(W_v)) = d_in - rank(W_v) = 2560 - 512 = 2048

SVD: W_v = U * Sigma * V^T, where V in R^{d_in x d_in}.
Define Q = V[:, 512:] in R^{d_in x 2048} (null-space basis columns).
Define P_null = Q @ Q^T in R^{d_in x d_in} (null-space projector).

### Theorem 1 (Null-Space Reparameterization)

**Statement:** Define a null-space LoRA layer as:

    Delta_y = B @ A_null @ Q^T @ x

where A_null in R^{r x d_null} and B in R^{d_out x r} are learnable, and Q is
the frozen null-space basis. Then:

1. W_v @ Q = 0 (by construction of Q as null-space basis)
2. The effective A-matrix is A_eff = A_null @ Q^T in R^{r x d_in}, and W_v @ A_eff^T = 0
3. The adapter has d_null = 2048 input directions to learn from

**Proof:**
(1) Q consists of right singular vectors with zero singular values:
    W_v @ Q = U @ Sigma @ V^T @ V[:, 512:] = U @ Sigma @ [0; I][:, 512:] = 0.

(2) A_eff = A_null @ Q^T. Then:
    W_v @ A_eff^T = W_v @ Q @ A_null^T = 0 @ A_null^T = 0.

(3) A_null has d_null = 2048 input features, far exceeding the r = 6 rank
    requirement. The adapter selects r directions from a 2048-dim subspace. QED.

### Theorem 2 (Gradient Retention Bound)

**Statement:** Let g in R^{d_in} be the gradient of the loss w.r.t. A's input at
a single position. The fraction of gradient energy retained under null-space
projection is:

    ||P_null @ g||^2 / ||g||^2 >= d_null / d_in = 2048/2560 = 0.80

in expectation, when g is isotropically distributed.

**Proof:** For isotropic g ~ N(0, I), the expected squared norm in any d_null-dim
subspace is d_null * sigma^2, and total expected squared norm is d_in * sigma^2.
The ratio is d_null / d_in. By concentration of measure (chi-squared tail),
the fraction is within [0.75, 0.85] with probability > 0.95 for d_in = 2560. QED.

**Caveat:** Real gradients are NOT isotropic. They concentrate along task-relevant
directions. If these directions happen to align with W_v's row space, null-space
projection loses more than 20%. If they align with null space (as Null-LoRA
empirically observes), the loss is much less.

### Theorem 3 (Base Model Output Preservation)

**Statement:** For input x, the base model's v_proj output is W_v @ x. With a
null-space adapter loaded, the output becomes:

    (W_v + B @ A_null @ Q^T) @ x = W_v @ x + B @ A_null @ (Q^T @ x)

The adapter adds a rank-r perturbation to the output. The base model's output
W_v @ x is exactly preserved (not approximately -- exactly).

**Proof:** Direct computation. The adapter contribution B @ A_null @ Q^T @ x is
additive, not multiplicative. W_v's computation is unchanged.

**Note on K1298:** The adapter DOES change the model's final output (that's its
purpose). K1298 tests whether this change degrades performance on tasks the base
model already handles (general knowledge). The null-space guarantee means the
adapter cannot corrupt W_v's existing feature extraction. But through the
attention mechanism (Q*K*V interaction), the additional value features still
propagate to the output. Whether this helps or hurts general knowledge is
empirical -- the theorem bounds the mechanism, not the downstream effect.

## Predictions

**P1 (K1297 proxy -- quality retention):**
Null-space LoRA final training loss <= 1.25x unrestricted LoRA final loss.
Equivalently, quality ratio >= 0.80. From Theorem 2 lower bound.

**P2 (Post-hoc projection):**
Projecting an unrestricted adapter's A-matrices into null space retains >= 70%
of the PPL improvement. The unrestricted adapter may place some update in W_v's
row space, which projection removes.

**P3 (K1298 -- base model preservation):**
Perplexity on general-knowledge prompts with null-space adapter loaded differs
from base model by < 1pp. From Theorem 3: adapter adds but doesn't corrupt.

**P4 (K1299 -- orthogonality):**
max|W_v @ A_eff^T| < 1e-4 for all layers. From Theorem 1, this is exactly 0
up to numerical precision.

## Kill Criteria Mapping

- K1297: Quality ratio >= 0.80 (loss-based proxy for GSM8K accuracy ratio)
- K1298: General-knowledge perplexity delta < 1pp
- K1299: max|W_v @ A_eff^T| < 1e-4

## Architectural Constraint: Gemma 4 Shared-KV Layers

**Critical discovery (REVISE iteration):** Gemma 4 E4B uses KV-sharing for its last
`num_kv_shared_layers` layers. Layers 24-41 receive pre-computed KV from layers 22/23
via the `shared_kv` parameter. When `shared_kv is not None`, the attention module
skips `k_proj(x)` and `v_proj(x)` entirely.

This means **v_proj is dead code on layers 24-41**. LoRA on these layers has zero
effect — zero gradients, zero logit delta, zero training. The first run targeted
layers 34-41, producing vacuously identical results for both adapter types.

**Correct targets:** layers 16-23 (last 8 non-shared layers that compute their own KV).
These have the same v_proj shapes and null-space dimensions as found in P7.A0.

## Experiment Design

1. Identify non-shared layers via `previous_kvs` mapping (layers where `prev[i] == i`)
2. Target last 8 non-shared layers: [16, 17, 18, 19, 20, 21, 22, 23]
3. Compute null-space bases Q for v_proj at these layers via SVD
4. Train unrestricted LoRA (v_proj, r=16, 500 iters, math instruction data)
5. Train null-space LoRA using reparameterization A = A_null @ Q^T (same setup)
6. Evaluate: training loss, held-out perplexity, orthogonality, base preservation

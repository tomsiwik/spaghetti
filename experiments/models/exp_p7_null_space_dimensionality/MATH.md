# P7.A0: Null-Space Dimensionality of Gemma 4 E4B Per Layer

## Type: Verification (measurement)

## Motivation

Null-LoRA (arXiv:2512.15233) projects adapter updates into the null space of the
pre-trained weight matrix, guaranteeing zero interference with the base model's
learned function. Before we can use null-space projection for interference-free
composition, we need to know: **how large is the null space at each layer?**

This is a prerequisite for exp_p7_null_space_adapter_quality (P7.A1).

## Prior Work

- **Null-LoRA** (arXiv:2512.15233): Projects LoRA updates into null(W) to preserve
  base model capabilities during fine-tuning.
- **Finding #49**: Grassmannian A-matrix isolation achieves cos < 0.01 between adapters,
  but operates in the full input space, not restricted to null(W).
- **Finding #492** (Standing Committee): Module-disjoint LoRA fails because attention
  Q*V coupling is first-order. Same-module composition (both adapters on same weight
  matrix, orthogonal subspaces) is the correct approach.

## Mathematical Framework

### Definition (Null Space)

For W in R^{m x n}, the null space is:

    null(W) = {x in R^n : Wx = 0}

    dim(null(W)) = n - rank(W)

### Theorem (Effective Null-Space Capacity)

**Statement:** For a weight matrix W in R^{m x n} with singular value decomposition
W = U * diag(sigma_1, ..., sigma_p) * V^T where p = min(m,n), define the effective
rank at threshold epsilon as:

    r_eff(epsilon) = |{i : sigma_i / sigma_1 > epsilon}|

Then the effective null space has dimension:

    d_null(epsilon) = n - r_eff(epsilon)

and can accommodate floor(d_null / r) independent rank-r adapters whose updates
are orthogonal to each other AND to the base model's significant directions.

**Proof:** The right singular vectors V_{r_eff+1}, ..., V_n span the effective
null space. Any vector x in this subspace satisfies ||Wx|| <= epsilon * sigma_1 * ||x||.
Partitioning these vectors into groups of r yields floor(d_null/r) orthogonal
adapter slots, each guaranteed to perturb the base model's output by at most
epsilon * sigma_1 * ||delta||. QED.

### Predictions for Gemma 4 E4B

Architecture (from config):
- hidden_size (d_model) = 2560
- num_hidden_layers = 42
- head_dim = 256 (local), 512 (global)
- q_proj: R^{m x 2560} where m depends on layer type

**For local-attention layers (q_proj in R^{2048 x 2560}):**
- Theoretical null dim = 2560 - 2048 = 512 (if full rank)
- With 4-bit quantization reducing effective rank, expect d_null >= 512
- At r=6: capacity >= 85 adapter slots

**For global-attention layers (q_proj in R^{4096 x 2560}):**
- Theoretical null dim = max(0, 2560 - 4096) = 0
- But quantization + training dynamics reduce effective rank
- Effective null dim depends on sigma threshold

**K1294 prediction:** d_null >= 100 for local layers (conservative; theory says >= 512)
**K1295 prediction:** effective d_null >= 50 for global layers (from quantization effects)
**K1296 prediction:** std(d_null) / mean(d_null) < 0.20 (weight matrices trained similarly)

### What This Means for Composition

If d_null is large (>> 100), then null-space projection is a viable path to
interference-free adapter composition: each adapter lives in a subspace that
literally cannot affect the base model's output. Combined with Grassmannian
slot assignment within the null space, we get both base-model preservation
AND inter-adapter orthogonality.

## Kill Criteria

- K1294: Null space dim >= 100 for local layers (enough for 16+ adapters at r=6)
- K1295: Effective null space (sigma < 1e-3 * sigma_max) >= 50 dims for global layers
- K1296: Null space stable across layers (std of null_dim < 20% of mean)

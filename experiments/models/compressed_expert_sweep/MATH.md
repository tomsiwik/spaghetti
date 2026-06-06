# Compressed Expert Sweep: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {64, 256, 896} (micro) |
| d_ff | MLP intermediate dimension | 4d |
| L | Number of MLP layers | 2 |
| r | LoRA rank | 8 (micro), 16 (production) |
| N | Number of expert adapters | 8 (experiment), 50 (geometric test) |
| W | Pretrained weight matrix | (d, d_ff) or (d_ff, d) |
| dW | Weight perturbation (adapter delta) | same shape as W |
| U_r | Top-r left singular vectors of W | (d, r), orthonormal |
| V_r | Top-r right singular vectors of W | (d_ff, r), orthonormal |
| Sigma_r | Top-r singular values of W | (r,) |
| M | LoRA-XS learnable mixing matrix | (r, r) |
| B_s, A_s | VeRA shared random matrices | (d, r) and (r, d_ff) |
| lambda_b | VeRA per-expert output scaling | (d,) |
| lambda_d | VeRA per-expert rank scaling | (r,) |

## 2. Three Adapter Formats

### 2.1 Standard LoRA

    dW = B @ A,    B: (d, r),  A: (r, d_ff)

Each expert has INDEPENDENT A, B matrices. The column space of dW is
span(B), an r-dimensional subspace drawn (approximately) uniformly from
the Grassmannian Gr(r, d). The row space is span(A^T) in Gr(r, d_ff).

**Parameters per expert per layer:** 2 * d * r (for a single weight matrix)
**Total per expert (Qwen2.5-7B, 28 layers, 7 modules):** ~45 MB

### 2.2 LoRA-XS (Balazy et al. 2024)

    dW = U_r @ M @ V_r^T

where U_r, V_r are the top-r singular vectors of the PRETRAINED weight W,
and M is a learnable (r x r) matrix.

**Key constraint:** ALL experts share the SAME U_r and V_r. The only
per-expert parameter is M. This means:

    col(dW_i) subset of span(U_r)  for ALL experts i

ALL expert deltas live in the SAME r-dimensional column subspace.
The row space is similarly constrained to span(V_r).

**Parameters per expert per layer:** r^2
**Total per expert (Qwen2.5-7B):** ~100 KB
**Compression vs LoRA:** 448x

### 2.3 VeRA (Kopiczko et al. 2024)

    dW = diag(lambda_b) @ B_s @ diag(lambda_d) @ A_s

where B_s and A_s are shared random matrices (frozen, same for all experts),
and lambda_b, lambda_d are per-expert learned scaling vectors.

**Key constraint:** The column space of dW is constrained to SCALED versions
of the columns of B_s. Specifically:

    dW = [lambda_b[0] * B_s[0,:] * lambda_d, ..., lambda_b[d-1] * B_s[d-1,:] * lambda_d]^T @ A_s

The effective B matrix for expert i is diag(lambda_b_i) @ B_s, which spans
the same r-dimensional subspace as B_s but with per-row rescaling.

**Parameters per expert per layer:** d + r
**Total per expert (Qwen2.5-7B):** ~1.4 MB (+ ~45 MB shared)
**Compression vs LoRA:** 32x

## 3. Why Shared Bases Destroy Orthogonality

### 3.1 LoRA Orthogonality (Structural Guarantee)

For standard LoRA, each expert i has independent random A_i, B_i matrices.
The flattened delta vector v_i = vec(B_i @ A_i) lies in a random subspace
of R^D where D = d * d_ff.

By concentration of measure on the Grassmannian:

    E[|cos(v_i, v_j)|] ~ sqrt(2 / (pi * D_eff))

where D_eff >> d (the effective dimensionality accounting for the
rank-r structure). Empirically at d=896: mean|cos| = 0.0002.

The key property is that each expert's subspace is drawn INDEPENDENTLY
from the full ambient space, so they are near-orthogonal with high
probability.

### 3.2 LoRA-XS: All Experts in Same r^2-Dimensional Subspace

**Theorem (LoRA-XS Subspace Collapse).** For LoRA-XS, ALL expert deltas
lie in the same r^2-dimensional subspace of R^{d * d_ff}.

*Proof.* The delta dW_i = U_r @ M_i @ V_r^T. The vec operation gives:

    vec(dW_i) = (V_r kron U_r) @ vec(M_i)

where kron denotes the Kronecker product. The matrix (V_r kron U_r) has
shape (d*d_ff, r^2) and is FIXED for all experts. Therefore:

    vec(dW_i) in col(V_r kron U_r)  for ALL i

This is an r^2-dimensional subspace of R^{d*d_ff}. All expert delta
vectors live in this same subspace.

**Consequence:** The pairwise cosine between LoRA-XS expert deltas is
governed by the geometry of vectors in R^{r^2}, NOT R^{d*d_ff}:

    E[|cos(v_i, v_j)|] ~ sqrt(2 / (pi * r^2)) = sqrt(2 / (pi * 64)) = 0.099

This is INDEPENDENT OF d. No matter how large d is, LoRA-XS experts
live in an r^2 = 64-dimensional subspace, and their cosines are bounded
below by ~0.10. This is catastrophic for SOLE composition.

**Empirical validation:**

| d | LoRA mean|cos| | LoRA-XS mean|cos| | Ratio |
|---|---------------|--------------------|-------|
| 64 | 0.006 | 0.103 | 16x |
| 256 | 0.001 | 0.099 | 67x |
| 896 | 0.0005 | 0.099 | 218x |

LoRA-XS cosine is ~0.10 regardless of d, exactly matching the
sqrt(2/(pi*r^2)) prediction. The ratio grows without bound as d increases.

### 3.3 VeRA: Partial Subspace Sharing

VeRA is intermediate. The shared B_s and A_s fix the r-dimensional
column and row subspaces, but the per-expert lambda_b vector provides
d degrees of freedom in how the columns are scaled.

The effective delta is:

    dW_i = diag(lambda_b_i) @ B_s @ diag(lambda_d_i) @ A_s

The column space of dW_i is span(diag(lambda_b_i) @ B_s), which is a
ROTATED version of span(B_s). The rotation depends on lambda_b_i.

Two experts' deltas have overlap determined by:

    cos(v_i, v_j) proportional to correlation of lambda_b_i, lambda_b_j

Since lambda_b has d components and is learned independently, VeRA experts
benefit from the d-dimensional diversity of the scaling vectors:

    E[|cos(v_i, v_j)|] ~ O(1/sqrt(d))

**Empirical validation:**

| d | LoRA mean|cos| | VeRA mean|cos| | Ratio |
|---|---------------|----------------|-------|
| 64 | 0.006 | 0.029 | 4.7x |
| 256 | 0.001 | 0.015 | 9.9x |
| 896 | 0.0005 | 0.008 | 17.8x |

VeRA cosines decrease with d (unlike LoRA-XS), but slower than LoRA.
The ratio grows as sqrt(d) because VeRA scales as 1/sqrt(d) while
LoRA scales as 1/d.

### 3.4 Summary of Orthogonality Scaling

| Format | E[|cos|] scaling | At d=896, r=8 | At d=3584, r=16 (predicted) |
|--------|-----------------|---------------|---------------------------|
| LoRA | ~ 1/D ~ 1/d^2 | 0.0003 | ~0.00002 |
| LoRA-XS | ~ 1/r (d-independent) | 0.099 | ~0.063 |
| VeRA | ~ 1/sqrt(d) | 0.008 | ~0.004 |

LoRA-XS cosines are PERMANENTLY stuck at ~1/r regardless of model size.
VeRA improves with d but never matches LoRA.

## 4. Signal Retention Analysis

### 4.1 Why LoRA-XS Has Near-Zero Signal Retention

The signal retention experiment generates random rank-r perturbations and
measures how well each format can approximate them.

LoRA: uses truncated SVD, which PERFECTLY reconstructs rank-r matrices
(signal retention = 1.000).

LoRA-XS: projects the target onto the span of (U_r, V_r) -- the top-r
SVD of the PRETRAINED weight. A random rank-r perturbation has negligible
overlap with the top-r singular subspace of W:

    E[||proj_{U_r}(target)||^2 / ||target||^2] = r^2 / (d * d_ff)

For d=256, d_ff=1024: r^2/(d*d_ff) = 64/262144 = 0.0002. This matches
the observed 0.01% signal retention.

**Key insight:** LoRA-XS assumes domain knowledge is encoded in the SAME
directions as the pretrained model's most important singular vectors. This
is a strong assumption that only holds if fine-tuning modifies the existing
principal components rather than adding new orthogonal directions.

### 4.2 Caveat: Trained vs Fitted

This experiment FITS adapters to synthetic targets (post-hoc approximation).
Real LoRA-XS TRAINING would learn M to minimize loss directly within the
constrained subspace. The signal retention numbers here measure the
GEOMETRIC CAPACITY of each format, not training effectiveness.

In practice, LoRA-XS has been shown to achieve reasonable quality on
standard benchmarks (Balazy et al. 2024 report within 1-2% of LoRA on
GLUE). This suggests that domain knowledge, when learned through gradient
descent, CAN be partially captured in the top-r SVD subspace -- but
the format is fundamentally limited to modifications along the pretrained
model's existing singular directions.

## 5. Inference Overhead

All three formats compute dW and add it to W. The computation:

| Format | Operations | FLOPs |
|--------|-----------|-------|
| LoRA | B @ A: (d, r) @ (r, d_ff) | 2*d*r*d_ff |
| LoRA-XS | U_r @ M: (d, r) @ (r, r) then @ V_r^T: (d, r) @ (r, d_ff) | 2*d*r^2 + 2*d*r*d_ff |
| VeRA | Scale B, then (d, r) @ (r, d_ff) | d*r + 2*d*r*d_ff |

At large d, all three are dominated by the (d, r) @ (r, d_ff) matmul,
so overhead is negligible. At small d (d=64), the fixed costs of
additional operations dominate, causing 40-80% overhead.

**Empirical at d=896:** LoRA-XS +3.9-18.9%, VeRA +4.8-7.0%.
This is noisy (CPU timing) but within the 10% threshold at production scale.

**In production:** All three formats are pre-merged into base weights
(dW computed once, added to W). Inference overhead is ZERO for pre-merge.
The overhead only matters for dynamic top-k routing (rare path).

## 6. Worked Numerical Example

### d=256, r=8, single weight matrix (d, d_ff) = (256, 1024):

**Parameter counts:**
- LoRA: B(256,8) + A(8,1024) = 2048 + 8192 = 10,240
- LoRA-XS: M(8,8) = 64
- VeRA: lambda_b(256) + lambda_d(8) = 264

**Compression:** LoRA-XS = 10240/64 = 160x. VeRA = 10240/264 = 39x.

**Orthogonality (50 random experts):**
- LoRA: mean|cos| = 0.0015, max = 0.009. 100% below tau=0.01.
- LoRA-XS: mean|cos| = 0.099, max = 0.427. Only 7.6% below tau.
- VeRA: mean|cos| = 0.015, max = 0.084. 46% below tau.

**At production scale (d=3584, r=16):**
- LoRA: mean|cos| ~ 0.00002 (extrapolated)
- LoRA-XS: mean|cos| ~ 0.063 (bounded by 1/sqrt(r^2) = 1/16)
- VeRA: mean|cos| ~ 0.004 (extrapolated via 1/sqrt(d))

## 7. Assumptions and Limitations

1. **Synthetic perturbations, not trained experts.** Signal retention
   measures geometric capacity (subspace overlap), not training-time
   quality. Real LoRA-XS training minimizes loss WITHIN the constrained
   subspace, which may capture more signal than post-hoc projection.

2. **Single weight matrix per layer.** Real models have 7 modules
   (q, k, v, o, gate, up, down). Cross-module diversity may help
   VeRA but not LoRA-XS (each module has its own fixed SVD basis).

3. **Random M/lambda initialization.** The geometric orthogonality
   test uses random expert parameters. Gradient-trained experts
   may have higher correlation (as shown for standard LoRA).

4. **CPU timing.** Inference overhead measurements are noisy and
   platform-dependent. GPU kernels (fused operations) would change
   the relative costs. Pre-merge makes overhead moot.

5. **MLP only, no attention.** Standard LoRA applied to attention
   (q, k, v, o) may show different signal retention for LoRA-XS
   because attention weight singular structure differs from FFN.

## 8. Connection to SOLE Architecture

The central finding is that SOLE's structural orthogonality guarantee
DEPENDS on each expert having an INDEPENDENT random subspace. This is
precisely what LoRA provides and what compressed formats sacrifice.

| Property | LoRA | LoRA-XS | VeRA |
|----------|------|---------|------|
| Independent subspaces | Yes | No (shared SVD) | Partial (shared B,A) |
| Orthogonality scaling | O(1/d^2) | O(1/r) constant | O(1/sqrt(d)) |
| Signal capacity | rank-r in d dims | rank-r in r dims | rank-r, scaled |
| Composition safety | Safe at any d | UNSAFE | Marginal |
| Storage/expert | 2*d*r | r^2 | d+r |

**Conclusion:** LoRA-XS is fundamentally incompatible with SOLE because
it forces all experts into the same subspace, destroying orthogonality
regardless of model dimension. VeRA is marginally compatible at large d
but with significantly higher interference than standard LoRA.

The storage advantage of compressed formats is real (448x for LoRA-XS)
but the orthogonality cost is fatal for composable architectures. For
SOLE, the correct storage optimization path is quantization (INT4/INT8
of standard LoRA) rather than subspace compression.

# Effective-Delta Cosine: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 2560 (BitNet-2B-4T) |
| r | LoRA rank | 16 |
| N | Number of adapters | 5 |
| L | Number of transformer layers | 30 |
| M | Number of LoRA modules per layer | 7 (q,k,v,o,gate,up,down) |
| A_i^{l,m} | LoRA A matrix for adapter i, layer l, module m | (d_in, r) |
| B_i^{l,m} | LoRA B matrix for adapter i, layer l, module m | (r, d_out) |
| DW_i^{l,m} | Effective weight delta: B_i^{l,m} @ A_i^{l,m} | (d_out, d_in) -- wait, note ordering below |

**Important convention note**: In MLX LoRALinear, the forward pass computes:
```
y = W @ x + (scale) * (lora_b @ (lora_a @ x))
```
So the effective delta per module is: DW = scale * lora_b @ lora_a, where
lora_a: (d_in, r) and lora_b: (r, d_out). The product lora_b @ lora_a has shape
(r, r) which is wrong -- actually lora_a @ x maps (d_in,) -> (r,), then lora_b @ that
maps (r,) -> (d_out,). So the effective delta in weight space is:

    DW = scale * (lora_b^T) @ (lora_a^T)^T -- NO.

Let me be precise. For y = Wx + scale * B(Ax):
- A: (d_in, r), maps x: (d_in,) to (r,) via A^T x
- B: (r, d_out), maps (r,) to (d_out,) via B^T h

Wait, MLX uses (out, in) convention for Linear. Let me re-check.

Actually in the saved weights:
- lora_a: (d_in, r) -- e.g. (2560, 16)
- lora_b: (r, d_out) -- e.g. (16, 2560)

The effective delta in weight-space (applied to output):
    DW_i = lora_b_i @ lora_a_i  -- shape (r, d_in) ... no

Let me just flatten: for each module m in each layer l, the effective delta is:
    delta_i^{l,m} = vec(B_i^{l,m} @ A_i^{l,m}^T)  -- whatever the correct matrix product

For the cosine computation, the EXACT matrix product formula matters less than
consistency. We compute: for each (layer, module), form the matrix product of the
B and A weight tensors in the order that produces the weight-space perturbation,
then flatten and concatenate across all layers/modules.

**The key insight**: regardless of the exact ordering convention, the cosine between
vec(DW_i) and vec(DW_j) measures the alignment of the actual perturbations applied
to the model weights. This is what matters for composition interference.

## 2. Two Cosine Metrics

### 2.1 Raw Parameter Cosine (current proxy)

For adapter i, concatenate all LoRA parameters into a single vector:

    p_i = concat_{l,m}( vec(A_i^{l,m}), vec(B_i^{l,m}) )

Dimension: D_raw = sum_{l,m} (d_in * r + r * d_out)
         = L * M * r * (d_in + d_out)  -- varies by module type

Raw parameter cosine:
    cos_raw(i,j) = (p_i . p_j) / (||p_i|| ||p_j||)

This is what tools/orthogonality.py currently measures.

### 2.2 Effective-Delta Cosine (correct metric)

For adapter i, compute the effective weight perturbation per module, then concatenate:

For each (l, m) pair, let:
    delta_i^{l,m} = B_i^{l,m} matmul A_i^{l,m}

where matmul uses the shapes as stored (B: (r, d_out), A: (d_in, r)):
    B @ A does not work: (r, d_out) @ (d_in, r) -- shape mismatch

The correct product that gives the weight-space delta depends on the convention.
In standard LoRA: h = W x + alpha * B A x, where:
- x: (d_in,)
- A: (r, d_in) maps to (r,)
- B: (d_out, r) maps to (d_out,)
- DW = alpha * B @ A: (d_out, d_in)

But the STORED shapes are transposed in MLX:
- lora_a stored as (d_in, r) = A^T
- lora_b stored as (r, d_out) = B^T

So the effective delta is:
    DW = (lora_b)^T @ (lora_a)^T = B @ A
where B = lora_b^T: (d_out, r), A = lora_a^T: (r, d_in).

Equivalently: DW = lora_b.T @ lora_a.T, shape (d_out, d_in).

In code: `effective_delta = lora_b.T @ lora_a.T`

Concatenate across all layers and modules:
    delta_i = concat_{l,m}( vec(DW_i^{l,m}) )

Dimension: D_eff = sum_{l,m} d_out * d_in  (much larger than D_raw)

Effective-delta cosine:
    cos_eff(i,j) = (delta_i . delta_j) / (||delta_i|| ||delta_j||)

## 3. Why Effective-Delta Cosine Should Be Lower

### 3.1 The Filtering Property

The inner product between two effective deltas for a single module:

    vec(DW_i)^T vec(DW_j) = tr(DW_i^T DW_j)
                           = tr((B_i A_i)^T (B_j A_j))
                           = tr(A_i^T B_i^T B_j A_j)

If A_i and A_j are nearly orthogonal (A_i^T A_j approx 0, guaranteed by the
Grassmannian skeleton), then:

    tr(A_i^T B_i^T B_j A_j) approx 0

regardless of B_i^T B_j. The A matrices ACT AS A FILTER on the B-matrix correlation.

### 3.2 Bound

    |vec(DW_i)^T vec(DW_j)| = |tr(A_i^T B_i^T B_j A_j)|
                             <= ||A_i^T A_j||_F * ||B_i^T B_j||_F  (Cauchy-Schwarz on Frobenius)

More precisely, using submultiplicativity:
    <= ||B_i|| * ||A_i^T A_j||_F * ||B_j||

where ||.|| denotes the operator norm.

For the norms in the cosine denominator:
    ||vec(DW_i)|| = ||B_i A_i||_F >= sigma_min(B_i) * ||A_i||_F = sigma_min(B_i) * sqrt(r)

(if A_i has orthonormal columns, ||A_i||_F = sqrt(r))

So:
    |cos_eff(i,j)| <= (||B_i|| * ||A_i^T A_j||_F * ||B_j||) / (sigma_min(B_i) * sqrt(r) * sigma_min(B_j) * sqrt(r))
                    = (||B_i||/sigma_min(B_i)) * (||B_j||/sigma_min(B_j)) * ||A_i^T A_j||_F / r
                    = kappa(B_i) * kappa(B_j) * ||A_i^T A_j||_F / r

where kappa(B) = ||B||/sigma_min(B) is the condition number.

**Key prediction**: If A-matrices are nearly orthogonal (||A_i^T A_j||_F small)
and B-matrices have moderate condition number, effective-delta cosine is
bounded by a product of the A-orthogonality and B condition numbers.

### 3.3 Ratio Prediction

The raw parameter cosine includes both A-A and B-B correlations additively:
    p_i . p_j = sum_{l,m} [vec(A_i)^T vec(A_j) + vec(B_i)^T vec(B_j)]

The effective-delta cosine filters B-B through A-A:
    delta_i . delta_j = sum_{l,m} tr(A_i^T B_i^T B_j A_j)

When A_i^T A_j approx 0, the effective-delta kills the B-B contribution entirely.
The raw metric does not — it ADDS the B-B term.

Therefore: |cos_eff| << |cos_raw| when the skeleton provides good A-orthogonality.

The prior finding at toy scale (d=64): B-matrix cos 0.0298 -> delta cos 0.0017 (17x).
At d=2560, the A-orthogonality should be even better (more dimensions), so we
expect an equal or larger filtering ratio.

## 4. Worked Numerical Example

BitNet-2B-4T: d=2560, r=16, L=30, M=7 modules per layer.

### 4.1 Dimensions

Per attention module (q, k, v, o projections vary):
- q_proj: A (2560, 16), B (16, 2560) -> DW (2560, 2560) = 6,553,600 elements
- k_proj: A (2560, 16), B (16, 640) -> DW (640, 2560) = 1,638,400 elements
- v_proj: A (2560, 16), B (16, 640) -> DW (640, 2560) = 1,638,400 elements
- o_proj: A (2560, 16), B (16, 2560) -> DW (2560, 2560) = 6,553,600 elements

Per MLP module:
- gate_proj: A (2560, 16), B (16, 6912) -> DW (6912, 2560) = 17,694,720 elements
- up_proj: A (2560, 16), B (16, 6912) -> DW (6912, 2560) = 17,694,720 elements
- down_proj: A (6912, 16), B (16, 2560) -> DW (2560, 6912) = 17,694,720 elements

Per layer: 6.55M + 1.64M + 1.64M + 6.55M + 17.69M + 17.69M + 17.69M = 69.5M
Total D_eff = 30 * 69.5M = ~2.08 billion elements per adapter delta vector.

For comparison:
D_raw = L * sum of (d_in*r + r*d_out) per module
      = 30 * 7 * (avg_d_in * 16 + 16 * avg_d_out) -- this is much smaller
      = total LoRA params = 21,626,880 (from results.json)

So D_eff / D_raw ~ 2.08B / 21.6M ~ 96x. The effective delta vectors live in a
much higher-dimensional space.

### 4.2 Expected Cosine Values

From cosine_convergence: raw |cos| mean = 0.00125 at convergence.

In high-dimensional random vectors, cos ~ O(1/sqrt(D)). Since D_eff >> D_raw,
the baseline random cosine is even lower for effective deltas.

The filtering bound says:
    |cos_eff| <= kappa(B)^2 * ||A_i^T A_j||_F / r

If kappa(B) ~ 10 (typical for trained small matrices) and ||A_i^T A_j||_F ~ 0.1
(low A-coherence from random init at d=2560):
    |cos_eff| <= 100 * 0.1 / 16 = 0.625 (upper bound -- very loose)

But the actual value should be much lower because:
1. The bound is over all singular values, not just top
2. A_i at d=2560, r=16 are in the trivial packing regime (Nr = 80 << d = 2560)
3. ||A_i^T A_j||_F ~ r/sqrt(d) = 16/sqrt(2560) = 0.316 (Haar-random expectation)

More realistically, from the cosine_convergence data, raw |cos| = 0.00125.
The effective-delta should be even lower -- perhaps 0.0001 to 0.001 range.

## 5. Assumptions

1. **LoRA weight convention**: lora_a (d_in, r) and lora_b (r, d_out) as stored in NPZ.
   Effective delta: lora_b.T @ lora_a.T gives (d_out, d_in) weight perturbation.

2. **Scale factor**: The LoRA scale (alpha/r = 20.0) is constant across all adapters
   and cancels in the cosine ratio. We can ignore it.

3. **Concatenation order**: Both metrics concatenate across layers and modules in
   sorted key order. The ordering is consistent across adapters.

4. **Independence**: We assume the 5 trained adapters used independent random A
   initialization (standard LoRA init, not Grassmannian). This is the WORST CASE
   for A-orthogonality -- the Grassmannian skeleton would only improve things.

5. **200-step vs 2000-step**: We use both 200-step adapters (from bitnet_2b_real_composition)
   and retrain 2000-step adapters. The mechanism should be the same at both
   training durations.

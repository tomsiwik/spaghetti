# Attention Layer Removal Safety: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 896 (Qwen 0.5B) |
| d_ff | MLP intermediate dimension | 4864 (Qwen 0.5B) |
| r | LoRA rank | 16 (production) |
| N | Number of expert adapters | {10, 20, 50} |
| k | Index of expert to remove | 0 <= k < N |
| L | Number of transformer layers | 24 (Qwen 0.5B) |
| delta_i^attn | Flattened attention delta of expert i | (D_attn,) |
| delta_i^mlp | Flattened MLP delta of expert i | (D_mlp,) |
| delta_i | Full flattened delta: concat(attn, mlp) | (D_full,) |
| D_attn | Attention delta dimension: 4*d*d | 3,211,264 |
| D_mlp | MLP delta dimension: 3*d*d_ff | 13,074,432 |
| D_full | Total: D_attn + D_mlp | 16,285,696 |
| cos_attn | Cosine similarity between attention deltas | ~0.85 (related domains) |
| cos_mlp | Cosine similarity between MLP deltas | ~0.001 (SOLE regime) |

## 2. Problem: Two Cosine Regimes in One Expert

### 2.1 Empirical Observation

From ffn_only_vs_all_modules (proven at micro), attention layers for related
domains have cos=0.85 while MLP layers have cos=0.59. At production scale
with SOLE structural orthogonality, MLP layers achieve cos~0.001. Attention
layers remain higher because they capture shared positional and syntactic
patterns.

### 2.2 Cosine of Concatenated Deltas

When we concatenate attention and MLP deltas into a single vector:

    delta_i = [delta_i^attn ; delta_i^mlp]

The cosine between two concatenated experts is:

    cos(delta_i, delta_j) = (delta_i^attn . delta_j^attn + delta_i^mlp . delta_j^mlp) /
                            (||delta_i|| * ||delta_j||)

If attention and MLP deltas have comparable norms:

    cos_full ~ (D_attn * cos_attn + D_mlp * cos_mlp) / D_full

With D_attn/D_full ~ 0.20 and D_mlp/D_full ~ 0.80:

    cos_full ~ 0.20 * 0.85 + 0.80 * 0.001 = 0.171

This places the concatenated delta in the "recomputation required" regime
(cos > 0.1) even though 80% of the delta (MLP) is nearly orthogonal.

### 2.3 The Problem with Joint GS

Joint Gram-Schmidt on concatenated deltas [attn; mlp] orthogonalizes in the
full D_full-dimensional space. The projection:

    proj(delta_i, delta_j) = (delta_i . delta_j / delta_j . delta_j) * delta_j

mixes attention and MLP components. A high attention cosine (0.85) causes
the projection to subtract signal from both the attention AND MLP portions,
even though the MLP portions are already nearly orthogonal.

This over-orthogonalization of MLP components degrades their signal.
Conversely, it under-orthogonalizes the attention components because the
MLP portion dilutes the joint cosine below the true attention cosine.

## 3. Per-Layer GS: The Correct Approach

### 3.1 Separate Composition

Instead of joint GS on concatenated deltas, compose each layer type separately:

    W_merged^attn = W_base^attn + sum GS(delta_1^attn, ..., delta_N^attn)
    W_merged^mlp  = W_base^mlp  + sum GS(delta_1^mlp, ..., delta_N^mlp)

### 3.2 Per-Layer Removal

To remove expert k:

**Attention layers** (cos=0.85): GS recompute required.

    W_new^attn = W_base^attn + sum GS(delta_1^attn, ..., delta_{k-1}^attn,
                                       delta_{k+1}^attn, ..., delta_N^attn)

**MLP layers** (cos=0.001): Naive subtraction sufficient.

    W_new^mlp = W_merged^mlp - delta_k'^mlp

    where delta_k'^mlp is the GS-orthogonalized MLP delta (approximately
    equal to delta_k^mlp due to near-orthogonality).

### 3.3 Hybrid Error Analysis

The hybrid strategy's total error is dominated by the MLP naive subtraction:

    E_hybrid = [E_attn ; E_mlp]

    E_attn = 0  (GS recompute is exact)
    E_mlp ~ cos_mlp * (N - k - 1) * ||delta^mlp||  (from parent MATH.md)

For N=50, removing middle expert (k=25):

    ||E_mlp|| / ||delta^mlp|| ~ 0.001 * 24 = 0.024 = 2.4%

    ||E_hybrid||_F = sqrt(0 + ||E_mlp||^2) ~ 0.024 * ||delta^mlp||

    Relative error ~ 0.024 * (D_mlp/D_full)^(1/2) ~ 0.024 * 0.89 = 2.1%

In practice, measured hybrid error is 0.06-0.09%, well below the theoretical
bound because: (a) MLP cosines are lower than 0.001 at d=896, and (b) the
bound is conservative (it assumes worst-case alignment).

## 4. Timing Analysis

### 4.1 GS Recompute Complexity

GS recomputation for N experts with D-dimensional deltas:

    T_GS = O(N^2 * D)

For attention-only removal:

    T_attn = O(N^2 * D_attn) = O(N^2 * 4d^2)

For MLP naive subtraction:

    T_mlp_naive = O(D_mlp) = O(3 * d * d_ff)

Hybrid total:

    T_hybrid = T_attn + T_mlp_naive = O(N^2 * 4d^2 + 3 * d * d_ff)

### 4.2 Measured Timings

At d=896, r=16, D_attn=3,211,264:

| N | GS recompute attn (s) | Naive MLP (ms) | Hybrid total (s) |
|---|----------------------|----------------|-------------------|
| 10 | ~0.28 (extrapolated) | <1 | ~0.28 |
| 20 | ~0.94 (extrapolated) | <1 | ~0.94 |
| 50 | 5.84 (measured) | <3 | ~5.84 |
| 100 | ~16.3 (extrapolated) | <3 | ~16.3 |

K2 threshold (10s) is reached at approximately N~65 for attention-only
recompute at actual D_attn dimension. However, this is the single-layer
timing. For per-layer attention removal across L=24 layers, the work is
parallelizable (each layer independent).

### 4.3 Comparison: Hybrid vs Full Recompute

Full recompute (all modules): O(N^2 * D_full) = O(N^2 * (4d^2 + 3*d*d_ff))

Hybrid: O(N^2 * 4d^2 + 3*d*d_ff) -- GS only on attention, naive on MLP

Speedup factor:

    D_full / D_attn = (4d^2 + 3*d*d_ff) / (4d^2)
                    = 1 + 3*d_ff / (4*d)
                    = 1 + 3*4864 / (4*896)
                    = 1 + 4.07 = 5.07

Hybrid is approximately 5x faster than full GS recompute for Qwen 0.5B
architecture. At d=4096 (Qwen 7B) with d_ff=11008:

    Speedup = 1 + 3*11008 / (4*4096) = 1 + 2.01 = 3.01

Still 3x faster. The speedup diminishes at larger d because the d^2 attention
term grows faster relative to d*d_ff.

## 5. Cosine Sweep: Regime Boundary Refinement

The parent experiment identified cos<0.01 as "naive OK" and cos>0.1 as
"recompute required." This experiment refines with a full sweep:

| cos range | Naive recon error (N=20) | Regime |
|-----------|-------------------------|--------|
| 0.01 | 1.8% | MARGINAL |
| 0.05 | 6.1% | RECOMPUTE |
| 0.10 | 8.6% | RECOMPUTE |
| 0.20 | 10.8% | RECOMPUTE |
| 0.30 | 11.8% | RECOMPUTE |
| 0.50 | 12.5% | RECOMPUTE |
| 0.70 | 12.5% | RECOMPUTE |
| 0.85 | 11.8% | RECOMPUTE |
| 0.95 | 9.8% | RECOMPUTE |

The error curve is non-monotonic, peaking around cos=0.5-0.7. This is because:

- At low cosine: error ~ cos * (N-k-1) (linear in cos)
- At high cosine: GS orthogonalization removes more signal, so the
  orthogonalized deltas are smaller. Subtracting a smaller delta produces
  a smaller absolute error. But the denominator (||W_gt||) also shrinks,
  so relative error can go either way.

The peak at cos~0.5 is where the product of "projection magnitude" and
"number of affected experts" is maximized.

## 6. Partial Removal: Signal Preservation

When removing only the attention component of an expert while keeping MLP:

    W_new = [W_attn^{without k} ; W_mlp^{with k}]

This preserves the MLP contribution of expert k (its domain-specific FFN
knowledge) while removing its attention patterns. This is viable when:

1. The attention component is the source of interference (cos=0.85)
2. The MLP component is valuable and non-interfering (cos=0.001)
3. The expert needs to be partially deactivated (e.g., during a tournament
   where attention patterns conflict but FFN knowledge is correct)

Measured: removing only attention yields 11.8% recon error on the attention
portion but 0% on MLP. The total model-wide impact depends on how attention
errors propagate through subsequent layers (tested separately in
multilayer_removal_cascade).

## 7. Assumptions

1. **Cosine similarity is uniform within layer type.** In reality, Q/K/V/O
   matrices may have different cosine distributions. The experiment uses a
   single D_attn vector representing all 4 attention matrices. Per-matrix
   GS would be even cheaper (4 separate small GS operations instead of 1
   large one).

2. **cos=0.85 is the production attention cosine.** This comes from
   non-converged micro models (ffn_only_vs_all_modules). Real production
   adapters at d=4096 may have different attention cosines. The structural
   orthogonality proof covers MLP but not attention explicitly. If
   attention cosines are lower at macro scale, the hybrid strategy becomes
   even more favorable (less work for attention GS).

3. **Single-layer simulation.** Multi-layer LoRA has one delta per layer.
   Per-layer GS is the natural extension: run GS independently on each
   layer's flattened deltas. This is embarrassingly parallel.

4. **Synthetic experts.** Real LoRA experts have structured weight patterns.
   The cosine statistics are the critical variable, and we match them to
   measured values (cos=0.85 for attention, cos=0.001 for MLP).

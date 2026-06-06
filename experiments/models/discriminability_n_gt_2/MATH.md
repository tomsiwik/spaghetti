# Discriminability at N>2: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| N | scalar | Number of experts (N=8 in this experiment) |
| k | scalar | Top-k selection count (k=2) |
| W_base | (d, d') | Frozen base model weight matrix |
| Delta_i | (d, d') | LoRA delta for expert i, i in {1,...,N} |
| x | (B, T, d) | Input hidden states |
| s_i(x) | scalar | Router logit for expert i on token x |
| p_i(x) | scalar | Softmax probability: exp(s_i)/sum_j exp(s_j) |
| T_k(x) | set | Top-k indices: {i : s_i(x) is among k largest} |
| w_i(x) | scalar | Renormalized weight: p_i/sum_{j in T_k} p_j for i in T_k, else 0 |
| f_i(x) | (V,) | Expert i's output logits for token x |
| D_ij(x) | scalar | Pairwise discriminability: ||f_i(x) - f_j(x)||_2 |
| D_mean | scalar | Mean discriminability: (2/N(N-1)) sum_{i<j} E_x[D_ij(x)] |
| g_R | (N, d) per layer | Router weight gradient (full matrix) |

## From N=2 to N>2: What Changes

### N=2, k=2 (parent experiment)

At N=2 with top_k=2, both experts are always selected. The router only controls
the mixing ratio w_1(x) vs w_2(x) = 1 - w_1(x). The composed output is:

```
f_comp(x) = w_1(x) * f_1(x) + w_2(x) * f_2(x)
```

The router gradient for this case:

```
dL/ds_1 = dL/df_comp * (dw_1/ds_1) * (f_1(x) - f_2(x))
```

The gradient is proportional to (f_1(x) - f_2(x)), which IS the discriminability.
This is why discriminability perfectly predicts gradient magnitude at N=2.

### N=8, k=2 (this experiment)

At N=8 with top_k=2, the router must both:
1. **Select** which 2 of 8 experts to activate (discrete decision)
2. **Mix** the selected pair's outputs (continuous weight)

The composed output involves a masking operation:

```
f_comp(x) = sum_{i in T_k(x)} w_i(x) * f_i(x)
```

where T_k(x) = argmax_k(s(x)) contains the indices of the top-k logits.

### Gradient Through Top-k Selection

The critical difference is the gradient flow through top-k. For expert i:

**If i in T_k(x) (selected):**
```
dL/ds_i = dL/df_comp * sum_{j in T_k} (dw_i/ds_i) * f_i(x)
        + dL/df_comp * sum_{j in T_k} w_j * (dw_j/ds_i) * f_j(x)
```
These gradients reflect MIXING dynamics (same as N=2 for the selected pair).

**If i not in T_k(x) (not selected):**

In standard implementation (as used here), the masking is differentiable through
the softmax probabilities that are multiplied by a binary mask. The gradient for
non-selected experts flows through the softmax normalization:

```
dL/ds_i = 0    (hard mask zeros out non-selected experts)
```

Wait -- in our RoutedDeltaGPT implementation, the mask is computed from the
scores, then masked_probs = probs * mask, then renormalized. The gradient of
masked_probs w.r.t. the original softmax probs IS nonzero for selected experts
through the renormalization denominator. But for non-selected experts where
mask[i] = 0, their contribution to the forward pass is zero, so dL/ds_i gets
only the indirect path through the renormalization.

The actual gradient path in MLX:
1. `scores = router(h)` -- logits for all N experts
2. `probs = softmax(scores)` -- probabilities for all N
3. `top_vals = topk(scores, k)` -- top-k logit values
4. `mask = (scores >= min(top_vals))` -- binary mask
5. `masked_probs = probs * mask` -- zero out non-selected
6. `masked_probs = masked_probs / sum(masked_probs)` -- renormalize

The gradient flows back through steps 6->5->4->3->2->1.
Step 4 uses `>=` which has zero gradient w.r.t. scores in general.
But step 3 (softmax) creates coupling: d(probs_i)/d(scores_j) is nonzero
for all i,j pairs, so even non-selected experts get SOME gradient.

### Net Effect on Gradient Magnitude

The total router gradient at N=8, k=2 is attenuated compared to N=2, k=2:

1. **Magnitude reduction**: Only k/N = 2/8 = 25% of experts contribute
   directly to the forward pass per token. The effective gradient is
   ~k/N of what N=2 sees.

2. **Selection noise**: Different tokens select different expert pairs.
   The gradient for expert i averages over:
   - Tokens where i is selected: strong discriminability-driven gradient
   - Tokens where i is not selected: near-zero gradient (indirect path only)

3. **Discriminability still matters**: For tokens where experts i,j are the
   selected pair, the gradient is still proportional to D_ij(x). Mean
   discriminability across ALL pairs predicts how strong these per-token
   gradients are on average.

### Expected Correlations

**Mean-curve prediction:** r^2(D_mean, ||g_R||) should be above 0.3 because
discriminability controls gradient magnitude for the selected pairs, and
mean discriminability captures the average signal strength across all
possible selections.

**Attenuation prediction:** The gradient ratio (cos=0.0 vs cos=0.9) should
be smaller than at N=2 because selection noise dilutes the discriminability
signal. The phase transition should be softer.

**Shape distortion prediction:** The gradient-vs-cosine profile at N=8
should be noisier than at N=2 because:
- Which experts are selected varies stochastically
- The indirect gradient path through softmax creates secondary effects
- With synthetic experts, pairwise cosine variance is large (std~0.4)

## Empirical Results

| Cosine | D_mean (N=8) | ||g_R|| (N=8) | ||g_R|| (N=2) | N8/N2 ratio |
|--------|-------------|---------------|---------------|-------------|
| 0.0    | 7.45        | 0.0416        | 0.2938        | 0.14x       |
| 0.1    | 7.10        | 0.0433        | 0.3205        | 0.14x       |
| 0.3    | 6.70        | 0.0736        | 0.2943        | 0.25x       |
| 0.5    | 5.99        | 0.0627        | 0.2242        | 0.28x       |
| 0.7    | 5.07        | 0.0422        | 0.0958        | 0.44x       |
| 0.9    | 3.18        | 0.0068        | 0.0155        | 0.44x       |

**Gradient attenuation factor:** N=8 gradients are 3.6-7.1x smaller than N=2,
consistent with k/N = 0.25 dilution plus selection noise.

**Phase transition:** Still present at N=8 but weaker:
- N=2: 19.0x gradient ratio (cos=0.0 vs cos=0.9)
- N=8: 6.1x gradient ratio (cos=0.0 vs cos=0.9)
- Parent experiment: 15.5x at N=2

**Mean-curve correlations:**
- r^2(D_mean, ||g_R||) = 0.46 at N=8 (PASSES 0.3 threshold)
- r^2(D_mean, ||g_R||) = 0.95 at N=2 (strong)
- r^2 drops by 0.49 going from mixing-only to selection+mixing

## Computational Cost

Per cosine level at N=8:
- Expert generation: O(N * D) where D = dimensionality of flat deltas
- Discriminability measurement: O(N^2 * B * T * V) -- 28 pairs x 5 batches
- Calibration: O(steps * N * B * T * d^2) -- 300 steps with 8 expert forwards
- Total per seed: ~85s
- Full experiment (3 seeds x 6 cosines x 2 conditions): ~4.3 min

## Assumptions

1. **Synthetic experts approximate real multi-domain experts.** We generate
   8 experts from 2 trained experts using geometric projection. Real experts
   trained on 8 different domains would have more varied structure.

2. **Mean pairwise discriminability is the right summary statistic.** With
   N=8 experts and k=2 selection, the router gradient depends on the
   discriminability of the SELECTED pair, not all pairs. Mean pairwise
   discriminability is a proxy.

3. **Pairwise cosine control is imprecise at N=8.** The Gram-Schmidt
   projection controls cosine with expert A, but pairwise cosines among
   generated experts vary substantially (std=0.42 at target cos=0.0).
   This adds noise relative to N=2 where cosine is precisely controlled.

4. **Standard top-k routing (not straight-through estimator).** Some
   MoE implementations use straight-through gradient estimators for top-k.
   Our implementation uses soft-masked routing where gradients flow through
   the softmax + mask path, which provides SOME gradient to non-selected
   experts but not a full dense signal.

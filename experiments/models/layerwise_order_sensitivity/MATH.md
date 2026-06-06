# Layer-Wise Merge Order Sensitivity: Mathematical Foundations

## 1. Setup

We extend the merge_order_dependence analysis from flattened vectors to
per-layer-type decomposition.

A transformer expert delta d_k in R^D is decomposed into sublayer components:

  d_k = [d_k^(1,wq), d_k^(1,wk), ..., d_k^(L,fc1), d_k^(L,fc2)]

where each sublayer component d_k^(l,s) is in R^d_s (dimension of that sublayer).
The full delta has dimension D = L * (|S_attn| * d_attn + |S_ffn| * d_ffn).

In our setup: L=4, S_attn = {wq, wk, wv, wo}, S_ffn = {fc1, fc2},
d_attn = d_ffn = 256, so D = 4 * (4 + 2) * 256 = 6144.

## 2. Two Orthogonalization Strategies

### 2.1 Flattened GS (parent experiment)

Flatten all sublayers into a single vector, apply GS to the N flattened vectors.
The orthogonalization sees all sublayers simultaneously and can "trade off"
corrections between layer types.

### 2.2 Layer-wise GS

Apply GS independently per sublayer. Each sublayer group is orthogonalized
in isolation. No cross-sublayer correction.

## 3. Per-Sublayer Cosine Independence

The key mathematical question: given that attention sublayers have pairwise
cosine c_attn and FFN sublayers have pairwise cosine c_ffn, does the
order sensitivity (variation ~ slope * c) have different slopes?

### 3.1 Theoretical Analysis

For a single sublayer (l, s) with expert vectors v_1, ..., v_N in R^d_s,
the GS process removes projections sequentially. The order dependence for
the merged average is:

  ||A(sigma) - A(tau)||_2 / ||A(sigma)||_2 = O(c)

where c is the pairwise cosine.

**The slope depends on the geometry of the subspace, not the layer type.**
Specifically, the variation scales as:

  variation ~ f(N, d_s) * c

where f(N, d_s) depends on the number of experts N and the sublayer
dimension d_s. When N and d_s are the same for attention and FFN sublayers
(as in our setup), the slope f is identical.

### 3.2 Empirical Verification

Measured slopes from the cosine sweep:
  - Attention: 61.9 (R^2 = 0.882)
  - FFN:       61.6 (R^2 = 0.884)
  - Ratio:     1.01x (within noise)

The slope differs from the parent experiment's ~80 because:
1. Individual sublayers have dim=256, not the full D=6144
2. Each sublayer has its own shared direction (independent, not correlated)
3. The flattened cosine is a weighted average of per-sublayer cosines,
   not a uniform cosine across all dimensions

## 4. Flattened Cosine as Weighted Average

When attention layers have cos_attn and FFN layers have cos_ffn, the
flattened cosine is approximately:

  cos_flat ~ (n_attn * cos_attn + n_ffn * cos_ffn) / (n_attn + n_ffn)

where n_attn = L * |S_attn| = 16 sublayers and n_ffn = L * |S_ffn| = 8 sublayers.

Numerical check:
  cos_flat = (16 * 0.85 + 8 * 0.59) / 24 = (13.6 + 4.72) / 24 = 0.763

Measured: 0.763. Exact match.

## 5. The Masking Effect

The flattened analysis does mask layer-specific effects, but NOT in the
direction the hypothesis predicted.

### 5.1 What Flattening Masks

Flattened GS applies a **uniform** orthogonalization across all sublayers.
When cosines differ by layer type, this creates:

- **Over-correction of FFN layers:** FFN sublayers have cos=0.59, but
  the flattened correction is driven by the higher cos=0.85 of attention.
  Flattened GS removes more FFN signal than necessary.

- **Under-correction of attention layers:** Conversely, the attention
  correction is diluted by the lower FFN cosine.

Empirical evidence (Phase 3):
  - Flattened GS retains 51.7% of attention signal (mean)
  - Layer-wise GS retains 51.3% of attention signal (mean)
  - Flattened GS retains 76.0% of FFN signal (mean)
  - Layer-wise GS retains 75.2% of FFN signal (mean)

The differences are small (< 1 percentage point). This is because GS
operates on the cross-expert overlap, not on the within-expert structure.
The per-sublayer shared directions are independent, so flattened GS
does not significantly cross-contaminate between sublayer types.

### 5.2 What Flattening Does NOT Mask

The hypothesis predicted that flattened analysis would hide layer-specific
order sensitivity. In fact:

1. The per-sublayer scaling law is identical (slope_attn / slope_ffn = 1.01)
2. The absolute variation differs because cosines differ, not because
   the mechanism differs
3. Flattened variation (52.3%) is HIGHER than either layer-wise group
   (attn: 49.4%, FFN: 47.2%) because the flattened vector has higher
   effective dimensionality

## 6. Divergence Between Methods

Flattened and layer-wise GS produce different merged vectors:
  - Cosine similarity: 0.975
  - Relative L2 difference: 22.4%

This divergence arises because flattened GS creates cross-sublayer
projections (removing an attention component from FFN parameters and
vice versa), while layer-wise GS projects only within each sublayer.

The divergence is real but does NOT translate to different order sensitivity.
Both methods show the same variation ~ 62 * cos scaling.

## 7. Implications

### 7.1 Order Sensitivity is Cosine-Determined, Not Layer-Type-Determined

The variation ~ slope * cos relationship holds independently per sublayer
with the same slope. The attention-layer concern from the parent experiment
was a correct observation of higher cosines, not a different sensitivity
mechanism.

### 7.2 Practical Consequence

At production scale (d=4096, r=16):
- Per-sublayer cos < 0.001 for unrelated domains
- Per-sublayer variation < 0.06% (well below any threshold)
- Even for related domains (cos=0.85), the variation is ~50% of the
  merged vector -- but this only matters if GS is used
- Simple averaging (no GS) is order-invariant and recommended for SOLE

### 7.3 When Layer-Wise GS Would Be Better

Layer-wise GS is preferable when:
1. Attention cosines differ significantly from FFN cosines (true in practice)
2. GS is actually needed (only when cos > 0.06, which is rare)
3. The cross-sublayer contamination of flattened GS is unacceptable

In practice, condition 2 is never met at production scale, so the choice
between flattened and layer-wise GS is moot.

## 8. Assumptions

1. **Synthetic experts with independent shared directions per sublayer.**
   Real transformers may have correlated shared directions across
   sublayers (e.g., attention and FFN learning related features).
   This would change the flattened vs layer-wise comparison but not
   the per-sublayer scaling law.

2. **Uniform sublayer dimension.** Real architectures have different
   dimensions for different sublayers (e.g., wk/wv may be smaller with
   GQA). The slope f(N, d_s) depends on d_s, so sublayers with different
   dimensions would have slightly different slopes.

3. **No nonlinear interactions.** We measure vector-space variation,
   not model output variation. Nonlinear interactions (ReLU, softmax)
   could amplify small vector differences differently for attention
   vs FFN layers.

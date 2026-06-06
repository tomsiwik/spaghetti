# Layer-Wise Merge Order Sensitivity: Research Digest

## Hypothesis

Attention-layer merge order sensitivity exceeds FFN at high domain overlap,
and flattened-vector analysis masks this layer-specific effect.

**Falsifiable:** If attention-layer order CV < 1% at cos=0.85 (K1 kill), or
if FFN and attention show identical order sensitivity scaling (K2 kill).

---

## What This Model Is

This experiment decomposes the parent merge_order_dependence analysis from
flattened vectors to per-layer-type components. The motivation: the
ffn_only_vs_all_modules experiment showed that attention layers have much
higher pairwise cosine (0.85) than FFN layers (0.59) for related domains.
Since the parent experiment showed variation ~ 80 * cos, the prediction was
that attention layers would show ~68% variation while FFN shows ~47%.

The experiment creates synthetic experts with controlled per-layer-type
cosines (matching production measurements) and runs three analyses:

1. **Phase 1:** Compare per-sublayer order sensitivity between attention and
   FFN at production cosines (3 seeds).
2. **Phase 2:** Sweep cosine from 0.01 to 0.90 to fit the scaling law
   separately for attention and FFN sublayers.
3. **Phase 3:** Compare flattened GS vs layer-wise GS to quantify the
   masking effect.

---

## Lineage in the Arena

```
gpt
 `-- lora_gpt
      `-- gram_schmidt_composition
           `-- merge_order_dependence (parent)
                `-- layerwise_order_sensitivity (this experiment)
```

---

## Key References

- **merge_order_dependence** (this project): Established variation ~ 80*cos
  with threshold at cos > 0.06. Used flattened vectors.
- **ffn_only_vs_all_modules** (this project): Measured attention cos=0.85
  vs FFN cos=0.59 for math-medical pair on Qwen2.5-7B adapters.
- **Golub & Van Loan, Matrix Computations:** GS order dependence is a known
  property; the per-component decomposition is novel in the LoRA context.

---

## Empirical Results

### Phase 1: Production-Relevant Analysis (3 seeds)

| Layer Type | Cosine | Mean Variation% | Predicted (80*cos) |
|:-----------|:-------|:----------------|:-------------------|
| Attention  | 0.850  | 48.5            | 68.0               |
| FFN        | 0.589  | 46.6            | 47.2               |
| Flattened  | 0.763  | 51.8            | 61.0               |
| **Ratio (attn/FFN)** | -- | **1.04x +/- 0.01** | 1.44x |

The predicted 1.44x ratio (68/47.2) does NOT materialize. The actual ratio
is 1.04x -- attention and FFN sublayers show effectively identical order
sensitivity per unit of cosine. The absolute difference comes from different
input cosines, not different sensitivity mechanisms.

### Phase 2: Cosine Sweep Scaling Law

| Metric | Attention | FFN | Parent (flat) |
|:-------|:----------|:----|:--------------|
| Slope (variation/cos) | 61.9 | 61.6 | ~80 |
| R-squared | 0.882 | 0.884 | ~0.99 |
| Slope ratio | 1.01x | -- | -- |

Both layer types follow the same linear law. The slope (~62) is lower than
the parent's ~80 because individual sublayers (dim=256) have independent
shared directions, reducing the effective dimensionality relative to the
parent's correlated flattened vectors (dim=4096).

### Phase 3: Flattened vs Layer-Wise GS

| Metric | Flattened GS | Layer-Wise GS |
|:-------|:-------------|:--------------|
| Attn retention (mean) | 51.7% | 51.3% |
| FFN retention (mean) | 76.0% | 75.2% |
| Order variation% | 52.3% | 47.5% |
| **Merged cos similarity** | -- | **0.975** |
| **Relative L2 diff** | -- | **22.4%** |

Flattened and layer-wise GS produce meaningfully different merged vectors
(22.4% L2 difference), but the order sensitivity is similar for both methods.
The retention differences between methods are small (< 1pp).

---

## Kill Criteria Assessment

### K1: Attention-layer order CV < 1% at cos=0.85

**PASS.** Attention variation is 48.5% (49x above 1% threshold). Attention
layers absolutely show order sensitivity -- but so does every layer type
at cos=0.85. This is a property of the cosine, not the layer type.

### K2: FFN and attention show identical order sensitivity scaling

**KILLED.** Attn/FFN slope ratio = 1.01x. Scaling is identical within noise.
The 1.04x absolute variation ratio reflects different input cosines, not
different mechanisms.

---

## Verdict: KILLED (K2) -- Order Sensitivity is Cosine-Determined

The hypothesis that attention layers have *intrinsically higher* order
sensitivity than FFN layers is killed. Both layer types follow the same
scaling law (variation ~ 62 * cos) with identical slopes.

The parent experiment's observation was correct in absolute terms: attention
layers DO have higher variation at production settings (49% vs 47%). But this
is entirely explained by higher input cosines (0.85 vs 0.59), not by any
layer-type-specific sensitivity mechanism.

---

## What We Learned (Despite the Kill)

1. **The variation ~ slope * cos law is universal across layer types.** This
   strengthens the parent finding: you need only measure pairwise cosine to
   predict order sensitivity for any layer group.

2. **Flattened and layer-wise GS diverge meaningfully (cos=0.975).** While
   order sensitivity is similar, the actual merged vectors differ by 22%.
   If GS were used in production (which it is not -- simple averaging is
   preferred), the choice of flattened vs layer-wise would matter.

3. **The per-sublayer slope (~62) is lower than the flattened slope (~80).**
   This makes sense: individual sublayers have lower effective dimensionality
   and independent shared directions. The flattened analysis overestimates
   per-sublayer sensitivity.

4. **Layer-wise composition strategies are NOT motivated by order sensitivity.**
   The hypothesis suggested that GS for attention + simple sum for FFN would
   be beneficial. Since the mechanism is identical, this decomposition offers
   no order-sensitivity advantage.

---

## Micro-Scale Limitations

1. **Synthetic experts, not real LoRA deltas.** Real transformer sublayers
   may have correlated shared directions across layer types, which would
   change the flattened vs layer-wise comparison (but not the per-sublayer
   scaling law).

2. **No model quality evaluation.** We measure vector-space variation, not
   NTP loss or generation quality. Nonlinear interactions (softmax attention,
   ReLU/SiLU) could amplify attention-layer variations more than FFN variations.

3. **Uniform sublayer dimension (256).** Real architectures have different
   dimensions (GQA makes wk/wv smaller). The slope f(N, d) depends on d,
   so sublayers with different dimensions would have slightly different slopes.

4. **Independent shared directions.** Each sublayer has its own shared
   direction (simulating independent parameter structure). In reality,
   wq and wk share more structure than wq and fc1, which could create
   correlated order effects within the attention block.

---

## What Would Kill This (at Macro)

This experiment is already killed (K2). The surviving question for macro
validation:

- **Nonlinear amplification:** Even though vector-space order sensitivity
  is identical per unit cosine, attention-layer variations might have
  disproportionate impact on model outputs due to the softmax nonlinearity
  and the quadratic Q*K interaction. This would require running GS with
  real LoRA experts at macro scale and measuring NTP loss variance across
  orderings -- but since GS is not used in SOLE (simple averaging preferred),
  this is academic.

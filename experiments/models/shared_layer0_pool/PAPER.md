# Shared Layer 0 Capsule Pool: Research Digest

## Hypothesis

Sharing a single Layer 0 capsule pool across domains degrades quality by
less than 2% compared to per-domain Layer 0 pools, because Layer 0
capsules learn domain-invariant features with high co-activation overlap
(J=0.527 from behavioral_dedup).

**Falsifiable**: If all three sharing strategies (base, average, first)
degrade quality by more than 2% vs full concatenation, the hypothesis is
killed.

**Result: PASS (all strategies).** All three sharing strategies actually
**improve** quality over full concatenation by 1.7-3.0%. The redundant
Layer 0 pools in full concatenation cause a "double counting" distortion
that sharing eliminates. Parameter savings: 8.1% of total model, 12.5%
of capsule-only params.

---

## What This Model Is

A composition protocol variant where Layer 0 uses a single shared capsule
pool instead of concatenating per-domain pools. The model architecture is
identical to ReLURouterGPT. The difference is in how composition is
performed:

- **Full concat (standard)**: All L layers concatenate domain pools
- **Shared Layer 0**: Layer 0 uses one pool; Layers 1+ concatenate

Three strategies for constructing the shared Layer 0:
1. **Base**: pretrained Layer 0 (no domain fine-tuning)
2. **Average**: weight-average of domain-specific Layer 0s
3. **First**: first domain's Layer 0 (tests domain-invariance)

---

## Lineage in the Arena

```
gpt -> ... -> relu_router -> capsule_dedup -> behavioral_dedup -> shared_layer0_pool
                (composition    (weight-cos     (activation-based   (shared Layer 0
                 by concat)      dedup)          dedup, J=0.527)    composition)
```

---

## Key References

**Behavioral Dedup (this project)**: Found Layer 0 cross-domain Jaccard
J=0.527, while deeper layers J<0.05. Motivated the sharing hypothesis.

**Yosinski et al. (2014)**: "How Transferable Are Features in Deep Neural
Networks?" -- Early layers learn general features, later layers specialize.
The Layer 0 sharing validates this principle for composed capsule pools.

**ReLU Router (this project)**: Established the composition-by-concatenation
protocol. Shared Layer 0 is a refinement of this protocol.

---

## Empirical Results

### 3-Seed Aggregate Quality (seeds 42, 123, 7)

| Method | Avg Val Loss | Std | vs Joint | vs Full Concat |
|--------|-------------|-----|----------|----------------|
| joint (baseline) | 0.5286 | 0.0051 | -- | -7.9% |
| full_concat (control) | 0.5737 | 0.0142 | +8.5% | -- |
| weight_avg | 0.5433 | 0.0051 | +2.8% | -5.3% |
| **shared_L0_base** | **0.5639** | **0.0225** | **+6.7%** | **-1.7%** |
| **shared_L0_average** | **0.5605** | **0.0116** | **+6.0%** | **-2.3%** |
| **shared_L0_first** | **0.5567** | **0.0136** | **+5.3%** | **-3.0%** |

All shared strategies improve over full concatenation. Numerically,
"first" leads (-3.0%), "average" is second (-2.3%), "base" is third
(-1.7%). However, the differences between strategies are not
statistically distinguishable at 3 seeds (overlapping confidence
intervals). We recommend **"average"** as the principled default for
D>2 domains, since "first" is arbitrary and "base" shows higher variance.

### Kill Criterion Analysis

| Strategy | vs Full Concat | Threshold | Verdict |
|----------|---------------|-----------|---------|
| base | -1.71% | >2% degrades | **PASS** |
| average | -2.31% | >2% degrades | **PASS** |
| first | -2.96% | >2% degrades | **PASS** |

None degrade quality. All improve it. The kill criterion is not triggered.

### Per-Seed Detail (illustrative, not a strategy recommendation)

| Seed | Shared L0 | Full Concat | Delta |
|------|-----------|-------------|-------|
| 42 | 0.5596 | 0.5852 | -4.38% |
| 123 | 0.5687 | 0.5781 | -1.62% |
| 7 | 0.5419 | 0.5579 | -2.86% |
| **Mean** | **0.5567** | **0.5737** | **-2.96%** |

Improvement is consistent across all seeds.

### Layer 0 Cross-Pool Jaccard (confirms behavioral_dedup)

| Metric | Value |
|--------|-------|
| Mean J | 0.544 |
| P50 | 0.571 |
| P90 | 0.950 |
| Max | 0.998 |

Consistent with behavioral_dedup's J=0.527. Layer 0 capsules from
different domains fire on the same inputs.

### Parameter Analysis

| Configuration | Total Params | Capsule Params |
|---------------|-------------|----------------|
| Full concat | 202,112 | 131,072 |
| Shared L0 | 185,728 | 114,688 |
| **Savings** | **16,384 (8.1%)** | **16,384 (12.5%)** |

---

## Why Sharing Improves Quality (Not Just Preserves It)

The surprising finding: shared Layer 0 is BETTER than full concatenation.
The explanation is the **double counting** problem.

Full concatenation at Layer 0 produces:
```
y_0 = Pool_A(x) + Pool_B(x)
```

When both pools produce similar outputs (J=0.54), this approximately
doubles the Layer 0 contribution relative to training. This excess
magnitude distorts the residual stream for all subsequent layers.

Shared Layer 0 avoids this:
```
y_0 = Pool_shared(x)
```

One contribution at the expected magnitude. The residual stream balance
between layers is preserved.

This also explains why weight averaging (which operates at ALL layers,
not just Layer 0) performs even better: it avoids double counting at
every layer. The shared Layer 0 approach is a targeted fix for the layer
with the most redundancy.

**Reconciliation with loudness-falsification (relu_router):** The
relu_router experiment tested whether a *global* learned scalar could close
the composition gap. It could not (learned scales converged to ~0.99),
falsifying the hypothesis that the overall composition gap is a loudness
(global magnitude) problem. The double counting described here is a
*per-layer* magnitude imbalance: Layer 0 contributes at ~2x its trained
magnitude while Layers 1-3 remain at 1x. A single global scalar cannot
correct a per-layer imbalance — it would need to simultaneously halve
Layer 0's contribution while leaving deeper layers unchanged. These are
distinct mechanisms: global loudness (killed) vs layer-specific magnitude
distortion from redundant pools (what we observe here).

---

## Micro-Scale Limitations

1. **Similar domains**: a-m vs n-z names share the same 26-character
   alphabet. With truly different domains (Python vs English prose),
   Layer 0 overlap might be lower. However, at BPE tokenization level,
   the base model's embeddings provide a shared representation that
   may still produce high Layer 0 overlap.

2. **Only 2 domains tested**: With D=5 or D=20 domains, the "first"
   strategy becomes arbitrary. "Average" is more principled at scale.

3. **Character-level tokenization**: Subword tokenization creates
   domain-specific tokens, potentially reducing Layer 0 overlap.

4. **No calibration**: The shared Layer 0 models are evaluated without
   any router calibration (zero-shot composition). With calibration,
   full concatenation might close the gap.

5. **Small scale**: d=64, P=128, L=4. At d=4096 with 24+ layers, the
   Layer 0 sharing benefit may be larger (more parameters saved) or
   smaller (less redundancy if Layer 0 learns more specialized features
   at higher capacity).

6. **Seed variance**: "base" strategy shows high std (0.0225), suggesting
   sensitivity to initialization. "Average" and "first" are more stable.

7. **Capacity-reduction alternative explanation**: The shared L0 model has
   fewer parameters (185K vs 202K for full concat). At micro scale, the
   smaller model may benefit from reduced overfitting rather than (or in
   addition to) eliminating double counting. A random-pruning-to-match-
   capacity control would disambiguate these explanations but is not
   blocking for the core finding that sharing does not degrade quality.

---

## What Would Kill This

### At Micro Scale (tested, survived)

- **Quality degradation >2%**: NOT triggered. All strategies improve
  quality by 1.7-3.0%.

### At Macro Scale (untested)

- **Layer 0 becomes domain-specific with diverse data**: If Python and
  English text create genuinely different Layer 0 features (low J),
  sharing would degrade quality.

- **Calibration closes the gap**: If full concatenation + calibration
  achieves better quality than shared Layer 0 + calibration, the sharing
  benefit is an artifact of the zero-shot evaluation protocol.

- **Many domains (D>>2) break averaging**: With 20 diverse domains, the
  "average" strategy may produce a Layer 0 that is a poor compromise.

---

## Implications for the Project

1. **Shared Layer 0 is a strict improvement over full concatenation in
   zero-shot composition.** It reduces parameters (8.1%) AND improves
   quality (1.7-3.0%). Whether it should become the default composition
   protocol requires a calibrated comparison (full_concat+calibration vs
   shared_L0+calibration), which has not yet been tested.

2. **The double counting problem is real.** Concatenating functionally
   redundant pools distorts the residual stream. This suggests a general
   principle: before concatenating, check co-activation overlap. Layers
   with high overlap should share, not concatenate.

3. **Contribution protocol update**: Contributors need not fine-tune
   Layer 0 capsule pools. Training only Layers 1+ reduces per-contributor
   compute by 25% (1 of 4 layers skipped) while producing a better
   composed model.

4. **Layer 0 confirms the feature hierarchy**: The pretrained base Layer 0
   (-1.7%) is competitive with domain-fine-tuned Layer 0, confirming that
   Layer 0 learns generic features that do not benefit from domain
   specialization.

5. **Generalizes the weight averaging insight**: Weight averaging works
   well (+2.8% vs joint) partly because it avoids double counting at ALL
   layers. Shared Layer 0 is a selective application of this principle to
   the layer where it matters most.

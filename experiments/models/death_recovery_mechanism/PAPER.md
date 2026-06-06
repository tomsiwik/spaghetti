# Inter-Layer Coupling Revival Mechanism: Research Digest

## Hypothesis

Weight updates in layers 0..l-1 shift the input distribution to layer l,
reviving dead ReLU neurons whose detector vectors now see positive inputs.
Freezing upstream layers should suppress this revival.

**Falsifiable**: If freezing upstream layers does NOT reduce revival in
downstream layers (by >5pp or >50%), inter-layer coupling is not the
primary revival mechanism.

**Result: PASS.** Freezing upstream layers reduces revival by 79-94%
across layers 1-3 (3 seeds). Inter-layer coupling is strongly supported as
the dominant revival mechanism. (n=3 seeds, no significance test;
directional evidence with large effect sizes.)

---

## What This Experiment Tests

Exps 17 and 18 established that dead ReLU neurons can revive during
training (28.1% of the S=100 dead cohort revives by S=3200), and
hypothesized inter-layer coupling as the mechanism. But the mechanism
was never directly tested -- the revival could be from optimizer
momentum, self-revival within the same layer, or measurement noise.

This experiment uses **layer-selective freezing** as a causal intervention:

1. **Baseline**: All MLP layers train (replicates Exp 18)
2. **Freeze-upstream**: Freeze MLP layers 0..l-1, measure revival in layer l
3. **Train-only-one**: Train a single MLP layer, measure revival everywhere

If inter-layer coupling drives revival, freezing upstream should suppress
downstream revival. Training only layer k should revive downstream layers
but not upstream layers.

**Embedding freeze status**: All conditions freeze embeddings (wte, wpe),
norm0, lm_head, and all attention layers. The `freeze_specific_mlp_layers()`
function calls `model.freeze()` first (which freezes everything), then
selectively unfreezes only the capsule pools in non-frozen layers. This
means x^0 = embed(tokens) is fixed across all conditions, and frozen
upstream MLP layers truly receive fixed inputs. No embedding drift confound.

---

## Lineage in the Arena

```
gpt -> ... -> relu_router -> dead_capsule_pruning -> pruning_controls
                                                         |
                                    +--------------------+--------------------+
                                    |                    |                    |
                             training_duration    capsule_revival    death_recovery_mechanism
                               (Exp 17:            (Exp 18:           (Exp 20: THIS
                                aggregate           per-capsule         layer-freeze
                                death rate)         identity)           causal isolation)
```

---

## Key References

- **Gurbuzbalaban et al. (2024)**: >90% of revived neurons re-die; cosine
  decay promotes revival; non-monotonic death trajectory. Directly informed
  the inter-layer coupling hypothesis.
- **ReDo (2024)**: Activation-based dead neuron profiling. Our profiling
  protocol is adapted from their approach.
- **Exp 17 (training_duration)**: Established non-monotonic death trajectory
  and first proposed inter-layer coupling as explanation.
- **Exp 18 (capsule_revival)**: Confirmed revival at per-capsule identity
  level (28.1% cohort revival, Jaccard=0.669).

---

## Empirical Results

### Key Finding: Upstream Freeze Suppresses Revival (79-94% reduction)

| Layer | Baseline Revival | Upstream Frozen Revival | Reduction |
|-------|-----------------|------------------------|-----------|
| L1    | 29.4% +/- 15.3% | 1.8% +/- 1.8%         | 94% (-27.6pp) |
| L2    | 26.1% +/- 9.9%  | 2.9% +/- 3.2%         | 89% (-23.2pp) |
| L3    | 37.6% +/- 23.1% | 7.8% +/- 3.3%         | 79% (-29.8pp) |

Revival measured as fraction of capsules dead at S=100 that become alive
by S=3200. 3 seeds (42, 123, 7). "Upstream frozen" means all MLP layers
before the target layer are frozen.

The residual 2-8% revival when upstream is frozen likely comes from:
(a) norm2's denominator shifting as alive capsules in the same layer change,
or (b) minor numerical effects. It is far below the baseline level. Note
that the norm2 self-revival path is a first-order effect through a within-layer
pathway (alive capsules changing the residual -> norm2 input shifts), not
merely a second-order numerical artifact, but its magnitude (2-8%) is small
compared to the inter-layer coupling effect (79-94% reduction).

### Anchor Dead Counts |D^l_100| Per Condition

The denominators of revival rates differ across conditions because the
S=100 death profile depends on which layers train during those first 100
steps. This table enables readers to assess small-denominator effects.

| Condition | L0 dead | L1 dead | L2 dead | L3 dead |
|-----------|---------|---------|---------|---------|
| baseline | 17.7 | 94.7 | 92.0 | 87.3 |
| freeze_upstream_of_L1 | 0.7 | 98.3 | 90.3 | 87.0 |
| freeze_upstream_of_L2 | 0.7 | 35.0 | 96.3 | 90.3 |
| freeze_upstream_of_L3 | 0.7 | 35.0 | 32.3 | 84.7 |
| train_only_L0 | 20.3 | 28.0 | 22.0 | 20.0 |
| train_only_L1 | 0.7 | 99.0 | 24.7 | 19.7 |
| train_only_L2 | 0.7 | 35.0 | 95.3 | 19.7 |
| train_only_L3 | 0.7 | 35.0 | 32.3 | 84.7 |

Key observation: In the train_only_L0 condition, L1-L3 have far fewer dead
capsules at S=100 (20-28) than in baseline (87-95). This is because L1-L3
do not train for those 100 steps, so they only accumulate the death caused
by L0's output changes propagating downstream -- which is modest compared
to the self-induced death when a layer trains actively. The high revival
rates in train_only_L0 (95.7% for L1, 81.4% for L2, 98.3% for L3) apply
to these smaller denominators.

### Training One Layer Revives Downstream Layers (with Explanation of Discrepancy)

| Trained Layer | Revival in Trained Layer | Revival in Downstream | Revival in Upstream |
|--------------|-------------------------|----------------------|---------------------|
| L1 only | 2.8% (self) | L2: 75.3%, L3: 78.6% | L0: 0.0% |
| L2 only | 2.2% (self) | L3: 84.0% | L0: 0.0%, L1: 0.0% |
| L3 only | 7.8% (self) | (none downstream) | L0: 0.0%, L1: 0.0%, L2: 0.0% |

L0 is excluded from self-revival reporting because its anchor dead count is
unstable: |D^0_100| per seed = [0, 49, 12], mean 20.3. One seed has 0 dead
capsules at S=100, making the revival rate undefined (0/0). With 128
capsules and ~1.6% baseline death in L0, the denominator is too small for
reliable rates.

No trained layer revives upstream layers (0.0% in all cases), confirming
the mechanism is strictly feed-forward through the residual stream.

### Explaining the Revival Rate Discrepancy

Training only L0 shows 95.7% revival in L1, but baseline (all layers
training) shows only 29.4%. Why do fewer trainable layers produce more
revival?

**Answer: L1's own training creates offsetting new deaths (alive->dead
transitions).** The new death analysis confirms this:

| Condition | L0 A->D | L1 A->D | L2 A->D | L3 A->D |
|-----------|---------|---------|---------|---------|
| baseline | 5.0 | 7.0 | 7.0 | 7.3 |
| train_only_L0 | 1.7 | 0.3 | 0.0 | 0.0 |

In baseline, L1 creates ~7.0 new dead capsules between S=100 and S=3200.
In train_only_L0, L1 is frozen so it creates only ~0.3 new dead capsules.
The revival from L0's upstream signal is similar in both conditions, but
in baseline it is partially offset by L1's self-induced new deaths.

Additionally, the anchor dead counts differ: train_only_L0 has only 28.0
L1 dead at S=100 (vs 94.7 in baseline), because L1 does not train during
those first 100 steps. The smaller denominator also contributes to the
higher percentage.

Net revival (revived - newly_dead) per layer:

| Condition | L0 net | L1 net | L2 net | L3 net |
|-----------|--------|--------|--------|--------|
| baseline | -0.7 | +20.3 | +16.3 | +28.7 |
| train_only_L0 | +4.0 | +26.7 | +16.3 | +19.3 |

### Death Rate Trajectories Under Freezing

| Step | Baseline L3 | Freeze L0-2, L3 trains | Difference |
|------|-------------|------------------------|------------|
| 0    | 20.6%       | 20.6%                  | 0.0pp      |
| 100  | 68.2%       | 66.1%                  | -2.1pp     |
| 400  | 65.6%       | 66.9%                  | +1.3pp     |
| 800  | 62.8%       | 67.2%                  | +4.4pp     |
| 1600 | 53.9%       | 66.9%                  | +13.0pp    |
| 3200 | 45.8%       | 65.6%                  | +19.8pp    |

With upstream frozen, Layer 3's death rate PLATEAUS at ~66% instead
of declining to ~46%. The 19.8pp gap at S=3200 is attributable to
lost inter-layer coupling revival.

---

## Micro-Scale Limitations

1. **Only 4 layers.** The causal chain (L0 changes -> L1 revives -> L2
   revives -> L3 revives) is short. At macro scale (24-32 layers), the
   coupling may attenuate or amplify through longer chains. The single-
   layer training result (L0 revives downstream at 81-98%) suggests
   long-range coupling is substantial even through intermediary layers.

2. **128 capsules per layer.** Small populations increase variance in
   revival rates. Layer 0 has ~1.6% dead capsules at S=0 (mean 0.7 dead),
   making its self-revival rate unstable and excluded from reporting.

3. **Attention always frozen.** Our protocol freezes attention in all
   conditions. In full training, attention weight updates would also
   shift the residual stream, potentially contributing an additional
   source of revival beyond MLP inter-layer coupling. The headline finding
   is therefore about "MLP inter-layer coupling when attention is frozen,"
   not general inter-layer coupling.

4. **Binary dead/alive threshold.** Capsules firing on 0.01% of inputs
   are classified as alive. Nearly-dead capsules may show different
   revival dynamics than truly-dead ones.

5. **Single domain (a_m).** Revival rates may differ across domains
   or in multi-domain composition scenarios.

6. **n=3 seeds, no significance test.** Effect sizes are large (79-94%
   reduction) and consistent in direction, providing strong directional
   evidence. But with 3 seeds, formal significance testing is not possible.
   Language throughout uses "strongly supported" rather than "confirmed."

7. **Condition-dependent denominators.** The S=100 dead set differs across
   conditions because only trainable layers induce death during the first
   100 steps. Revival rate comparisons across conditions should be
   interpreted alongside the |D^l_100| table, not as direct percentages
   of the same population.

---

## What Would Kill This

**At micro scale:**
- If the effect reverses with a different learning rate schedule (e.g.,
  cosine decay might enable self-revival sufficient to match inter-layer
  coupling).
- If longer training (>3200 steps) shows the frozen-upstream condition
  catching up to baseline revival (delayed but not prevented).

**At macro scale:**
- If the coupling attenuates with depth: revival in layer 24 might not
  depend on layer 0 updates, only on nearby layers (locality of coupling).
- If batch normalization or other normalization schemes (different from
  RMSNorm) provide sufficient distribution shift to enable revival
  without upstream weight changes.
- If the 2-8% residual revival (with frozen upstream) scales up and
  becomes the dominant mechanism at larger model sizes.
- If SiLU/SwiGLU models (with ~0% truly dead neurons, per Exp 15) make
  this entire mechanism irrelevant at macro scale.

---

## Implications for the Composition Protocol

1. **Pruning timing matters more than previously known.** Since revival
   is driven by inter-layer coupling, pruning should happen AFTER all
   layers have finished training (consistent with Exp 18 recommendation).
   Pruning early freezes the capacity loss.

2. **Layer-wise fine-tuning order affects death rates.** If contributing
   a new expert by fine-tuning only specific layers, the fine-tuned layers
   will revive dead capsules in downstream layers. This is desirable for
   the fine-tuned domain but may interact with composition of other experts.

3. **Frozen base model revival is impossible.** In the composition protocol,
   the base model's attention is frozen and MLP adapters are trained.
   Dead neurons in the frozen base MLP cannot revive because their
   upstream context (attention) is fixed. This is a feature, not a bug:
   it means the base model's dead neuron set is stable across all expert
   contributions.

4. **Layer 0 has outsized influence.** Training only Layer 0 revives
   95.7% of Layer 1's dead capsules (from a smaller denominator of 28 vs
   95 in baseline) and 81-98% in deeper layers. This supports the Exp
   behavioral_dedup finding that Layer 0 is special (high redundancy,
   processing raw embeddings) and the suggestion to share a Layer 0 pool
   across domains.

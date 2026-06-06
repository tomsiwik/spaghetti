# Inter-Layer Coupling Revival Mechanism: Mathematical Foundations

## 1. Problem Statement

Experiments 17 and 18 established that dead ReLU neurons can revive during
training: 28.1% of capsules dead at S=100 revive by S=3200 (Exp 18). The
hypothesized mechanism is **inter-layer coupling**: weight updates in layers
0..l-1 shift the input distribution to layer l, pushing previously-negative
inputs above zero for dead neurons.

But this hypothesis was never directly tested. The revival could instead be
caused by:
- **Optimizer momentum**: Adam's first moment carries gradient history
  from before death, providing non-zero updates for ~10 steps after death.
- **Self-revival**: Weight updates within layer l itself (to alive neurons)
  change the residual stream, which feeds back through the layer's own
  RMSNorm to potentially revive dead capsules in the same layer.
- **Noise/sampling**: Different training batches shift the profiling result.

**Q: Is inter-layer coupling the dominant revival mechanism?**

Falsifiable prediction: If upstream layers are frozen during fine-tuning,
revival in downstream layers should decrease substantially (>5pp or >50%
reduction). If not, inter-layer coupling is not the mechanism.

---

## 2. Notation

All notation follows capsule_revival/MATH.md and training_duration/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S         -- number of fine-tuning steps

a_i^l(S)  -- detector vector for capsule i in layer l after S steps
             a_i^l is row i of A^l, where A^l is (P, d)
b_i^l(S)  -- expansion vector for capsule i in layer l after S steps

x^l(S)    -- input to layer l's MLP at step S
             x^l depends on weights in layers 0..l-1

D^l_S     -- set of dead capsule indices in layer l at step S
A^l_S     -- set of alive capsule indices in layer l at step S

F         -- set of frozen layer indices (MLP weights fixed)
T         -- set of trainable layer indices (T = {0..L-1} \ F)

r^l(S_1, S_2; F) -- revival rate in layer l from S_1 to S_2
                     under freeze condition F
                   = |D^l_{S_1} & A^l_{S_2}| / |D^l_{S_1}|
```

---

## 3. Inter-Layer Coupling Model

### 3.1 The Residual Stream

In a transformer with ReLU MLP:

```
x^0 = embed(tokens)
x^{l+1} = x^l + Attn^l(norm1(x^l)) + B^l @ ReLU(A^l @ norm2(x^l + Attn^l(norm1(x^l))))
```

The input to layer l's ReLU MLP is:

```
h^l = norm2(x^l + Attn^l(norm1(x^l)))
```

where x^l depends on the outputs of ALL layers 0..l-1.

### 3.2 How Upstream Changes Revive Downstream Neurons

A dead neuron i in layer l satisfies:

```
a_i^{lT} h^l <= 0   for all h^l in support(data)
```

When layers 0..l-1 are trainable, their weight updates change x^l:

```
x^l(S+1) = x^l(S) + delta_x^l(S)
```

where delta_x^l arises from:
1. Changed attention weights in layers 0..l-1 (frozen in our protocol)
2. Changed MLP weights in layers 0..l-1 (the key variable)

This shifts h^l, potentially creating some h^l_new where:

```
a_i^{lT} h^l_new > 0
```

Note: a_i^l does NOT change (dead neuron receives zero gradient).
The revival comes entirely from input distribution shift.

### 3.3 Freezing as Causal Intervention

By freezing MLP layers in F, we fix delta_x^l = 0 for the contribution
of layers in F. If F = {0, ..., l-1} (all upstream frozen):

```
x^l(S) = x^l(0)   for all S   (upstream contribution fixed)
```

**Important**: This equality holds because `freeze_specific_mlp_layers()`
calls `model.freeze()` first, which freezes ALL parameters including
embeddings (wte, wpe), norm0, and lm_head. Only capsule pools in
non-frozen layers are then unfrozen. Therefore x^0 = embed(tokens) is
constant across training steps, and frozen upstream MLP layers truly
produce fixed outputs. There is no embedding drift confound.

The only source of input change to layer l is from layer l's own alive
capsules feeding back through the residual stream. If a_i^l is dead,
it receives exactly zero gradient, so a_i^l stays fixed. The remaining
path to revival is through norm2's denominator changing (from alive
capsules in layer l contributing differently to the residual). This is
a first-order effect through a within-layer pathway, not merely a
second-order numerical artifact, but empirically it accounts for only
2-8% revival vs 26-38% in the unfrozen baseline.

Prediction:

```
r^l(100, 3200; {0..l-1}) << r^l(100, 3200; {})
```

### 3.4 Downstream Revival from Single-Layer Training

An even stronger test: train only layer k and measure revival in layer l > k.

If inter-layer coupling is real:
- Revival in layers l > k should be substantial (upstream distribution shifts)
- Revival in layers l < k should be near zero (no upstream change)
- Revival in layer k itself should be small (self-revival only)

Prediction:

```
r^l(100, 3200; {0..L-1}\{k}) >> 0   for l > k   (downstream revival)
r^l(100, 3200; {0..L-1}\{k}) ~ 0    for l < k   (no upstream change)
r^k(100, 3200; {0..L-1}\{k}) ~ 0    for l = k   (self-revival only)
```

---

## 4. Experimental Design

### 4.1 Conditions

| Condition | Frozen MLP | Trainable MLP | Purpose |
|-----------|------------|---------------|---------|
| baseline | none | {0,1,2,3} | Replicate Exp 18 |
| freeze_upstream_of_L1 | {0} | {1,2,3} | Isolate L0's effect on L1 |
| freeze_upstream_of_L2 | {0,1} | {2,3} | Isolate L0-1's effect on L2 |
| freeze_upstream_of_L3 | {0,1,2} | {3} | Isolate L0-2's effect on L3 |
| train_only_L0 | {1,2,3} | {0} | L0's downstream reach |
| train_only_L1 | {0,2,3} | {1} | L1's downstream reach |
| train_only_L2 | {0,1,3} | {2} | L2's downstream reach |
| train_only_L3 | {0,1,2} | {3} | L3 self-revival only |

### 4.2 Protocol

Identical to Exp 17/18 except for selective MLP freezing:

1. Pretrain base model on ALL data (300 steps, attention + MLP)
2. For each condition and step count S in {0, 100, 400, 800, 1600, 3200}:
   a. Start from pretrained base (deepcopy)
   b. Freeze attention (all conditions) + specified MLP layers
   c. Fine-tune remaining MLP weights for S steps
   d. Unfreeze all, profile per-capsule activation frequencies
   e. Record per-layer dead/alive masks

Note: attention is ALWAYS frozen across all conditions (consistent with
Exp 17/18). The only variable is which MLP layers are frozen.

### 4.3 Anchor and Measurement

- Anchor: S=100 (matches Exp 18 death peak)
- Measurement: S=3200 (matches Exp 18 endpoint)
- Per-layer revival rate: fraction of S=100 dead capsules alive at S=3200

### 3.5 Why Single-Layer Training Shows Higher Revival Than Baseline

An apparent paradox: training only L0 revives 95.7% of L1's dead capsules,
but baseline (all layers training) revives only 29.4%. Fewer trainable
layers producing more revival seems contradictory but is explained by two
complementary effects:

**Effect 1: Offsetting new deaths (alive->dead transitions).** In baseline,
L1 trains and creates new dead capsules (alive->dead transitions, mean 7.0
per seed from S=100 to S=3200). In train_only_L0, L1 is frozen so it
creates only ~0.3 new deaths. The revival signal from upstream L0 is
similar in both conditions, but in baseline it is partially offset by L1's
self-induced new deaths. This is the dominant explanation.

**Effect 2: Different anchor dead sets.** The S=100 dead population
|D^l_100| differs across conditions because the death profile at S=100
depends on which layers trained during those first 100 steps. In
train_only_L0, |D^1_100| = 28.0 (L1 does not train, so it accumulates
fewer dead capsules). In baseline, |D^1_100| = 94.7 (L1 trains actively
and kills many capsules). The smaller denominator in train_only_L0
amplifies the percentage, even if the absolute number of revivals is
similar.

Formally, define:
```
r_net^l(S_1, S_2; F) = (|D^l_{S_1} & A^l_{S_2}| - |A^l_{S_1} & D^l_{S_2}|) / |D^l_{S_1}|
```

This net revival accounts for offsetting new deaths and is more comparable
across conditions with different training dynamics.

---

## 5. Kill Criteria

| Criterion | Threshold | What it means |
|-----------|-----------|---------------|
| Upstream freeze does NOT reduce revival | reduction < 5pp AND < 50% | Inter-layer coupling not the mechanism |

Specifically: for at least one of layers 1, 2, 3, the difference
r^l(100, 3200; {}) - r^l(100, 3200; {0..l-1}) must exceed 5pp OR
the freeze revival must be less than 50% of baseline revival.

---

## 6. Worked Numerical Example

At d=4, P=4, L=3:

### Baseline (all layers train)

```
S=100: Layer 1 dead = {cap0, cap1}, Layer 2 dead = {cap2, cap3}
S=3200: Layer 1 dead = {cap0},      Layer 2 dead = {cap3}

Layer 1 revival: cap1 revived (1/2 = 50%)
Layer 2 revival: cap2 revived (1/2 = 50%)
```

How cap1 revived: Layer 0 MLP learned new features, shifting x^1.
RMSNorm(x^1_new) now has a positive projection onto a_{cap1}^1.

### Freeze upstream of L2 (freeze layers 0, 1)

```
S=100: Layer 2 dead = {cap2, cap3}  (same as baseline)
S=3200: Layer 2 dead = {cap2, cap3} (NO revival -- frozen upstream)

Layer 2 revival: 0/2 = 0%
```

Why: x^2 is fixed because layers 0 and 1 are frozen. Layer 2's own
alive capsules change, but their contribution feeds back through norm2's
denominator, which is a weak signal (second-order effect).

Revival reduction: 50% - 0% = 50pp (100% reduction). Kill NOT triggered.

---

## 7. Assumptions

1. **Attention and embeddings always frozen.** All conditions freeze
   attention weights, embeddings (wte, wpe), norm0, and lm_head. This is
   consistent with Exp 17/18 and the fine-tuning protocol. Only MLP
   (capsule pool) weights are the treatment variable. The embedding freeze
   is verified: `freeze_specific_mlp_layers()` calls `model.freeze()` first
   (freezing everything), then selectively unfreezes only capsule pools.

2. **Same training trajectory per condition.** All conditions use the same
   base model, same training data, same seed. The only difference is which
   MLP layers' gradients are zeroed (via freeze).

3. **Revival is measured by per-capsule identity.** A capsule is identified
   by (layer, index). Revival = dead at S=100, alive at S=3200. Same binary
   threshold (f_i = 0) as Exp 17/18.

4. **Profiling protocol matches Exp 17/18.** 20 batches, 32 samples each,
   on domain validation data. Exp 12 confirmed this protocol has < 4%
   measurement noise.

5. **Frozen layers maintain their pretrained weights.** Freezing prevents
   gradient updates but does not prevent the frozen layer from participating
   in the forward pass. The frozen layer's output still contributes to the
   residual stream, just with fixed weights.

6. **The residual stream is the primary coupling path.** Inter-layer
   coupling occurs through the residual stream: layer l's MLP output is
   added to x^l, which becomes x^{l+1}'s input. There is no skip connection
   or other pathway between non-adjacent layers except through the residual.

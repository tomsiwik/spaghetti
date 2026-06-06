# Pre-Composition Pruning Pipeline: Mathematical Foundations

## 1. Problem Statement

Given N independently-trained ReLU domain models with a shared base,
validate that pruning dead capsules BEFORE composition produces equivalent
quality to the established compose-then-prune pipeline. The key operation
order is:

```
Pipeline A (baseline):  Compose -> Profile -> Prune -> Calibrate
Pipeline B (proposed):  Profile -> Prune -> Compose -> Calibrate
```

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
N_d       -- number of domains (2 for binary split)
L         -- number of transformer layers (4)

A_k in R^{P x d}     -- detector matrix for domain k (rows are a_{k,i}^T)
B_k in R^{d x P}     -- expansion matrix for domain k (columns are b_{k,i})

D_k                   -- domain k's validation dataset
D_joint               -- joint validation dataset (union of all domains)

f_{k,i}^{own}        -- activation freq of capsule i in domain k model on D_k
f_{k,i}^{joint}      -- activation freq of capsule i in domain k model on D_joint
f_{c,i}^{joint}      -- activation freq of capsule i in composed model on D_joint

S_k^{own}            -- dead set from own-domain profiling: {i : f_{k,i}^{own} = 0}
S_k^{joint}          -- dead set from joint profiling: {i : f_{k,i}^{joint} = 0}
S_c                   -- dead set from composed-model profiling: {i : f_{c,i}^{joint} = 0}
```

---

## 2. Theoretical Foundation

### 2.1 Composition Preserves Dead Capsule Identity

**Theorem (Exp 16)**: The dead capsule sets from single-domain profiling
and composed-model profiling have high overlap:

```
J(S_k^{own}, S_c^{k-half}) = 0.895   (Jaccard similarity, 3-seed mean)
O(S_k^{own}, S_c^{k-half}) = 0.986   (Overlap coefficient)
```

where S_c^{k-half} is the subset of S_c corresponding to domain k's capsules.

The overlap coefficient of 0.986 means: 98.6% of capsules dead in
single-domain models remain dead after composition.

### 2.2 Why Pipeline B Works

**Claim**: If capsule i is dead in single-domain model k, it remains
dead in the composed model with probability >= 0.986.

**Proof sketch**: The composed model's forward pass for layer l is:

```
y_composed(x) = sum_{k=1}^{N_d} B_k @ ReLU(A_k @ x_l)
```

where x_l is the input to layer l. The key observation: A_k @ x_l is
identical for domain k's capsules regardless of whether other domains'
capsules are present, because ReLU is element-wise. The only difference
is that x_l itself changes due to the residual contributions of other
domains' capsules in earlier layers.

For capsule (k, i) with a_{k,i}^T x_l << 0 (deeply dead), the additive
residual perturbation from other domains' capsules in earlier layers
typically does not flip the sign:

```
a_{k,i}^T (x_l + delta_l) = a_{k,i}^T x_l + a_{k,i}^T delta_l
```

where delta_l is the cumulative perturbation. If |a_{k,i}^T x_l| >> |a_{k,i}^T delta_l|,
the capsule remains dead.

### 2.3 Pipeline B Prunes MORE Aggressively

**Observation**: Pipeline B (own-domain profiling) finds MORE dead capsules
than Pipeline A (composed-model profiling).

```
Pipeline A: 55.2% pruned (3-seed mean)
Pipeline B: 61.2% pruned (3-seed mean)
Gap:        +6.0pp more pruning in Pipeline B
```

**Explanation**: In Pipeline A, the composed model profiles on joint data,
so each domain's capsules see BOTH domains' inputs. Some capsules that are
dead for their own domain may fire on the other domain's inputs (cross-domain
activation). Pipeline B profiles each domain on only its own data, where
these cross-domain activations are absent.

This is NOT a quality risk because:
1. Cross-domain activations are incidental (not trained for)
2. Calibration (100 steps on joint data) compensates for any missing contributions
3. The empirical delta is +0.01% (effectively zero)

### 2.4 Quality Equivalence

**Theorem**: For ReLU capsule pools, the quality difference between
Pipeline A and Pipeline B after calibration is bounded by:

```
|L_A - L_B| <= C * |S_B \ S_A| * epsilon_margin
```

where:
- |S_B \ S_A| is the number of capsules pruned by B but not A
  (extra pruning from own-domain profiling)
- epsilon_margin is the mean activation magnitude of these capsules
- C is a constant depending on calibration effectiveness

Since capsules in S_B \ S_A fire only on cross-domain data (which they
were not trained for), their contribution is noise. Calibration eliminates
this noise regardless of whether it's pruned or retained.

---

## 3. Three Profiling Strategies for Pipeline B

### 3.1 Own-Domain Profiling (Pipeline B)

Profile domain k model on D_k:

```
S_k^{own} = {i : (1/|D_k|) sum_{x in D_k} 1{a_{k,i}^T x > 0} = 0}
```

**Properties**:
- Most conservative (fewest dead): only capsules dead for own domain
- Most parallelizable: each domain contributor profiles independently
- No joint data needed for profiling

### 3.2 Cross-Domain Profiling (Pipeline B2)

Profile domain k model on D_{-k} (other domain's data):

```
S_k^{cross} = {i : (1/|D_{-k}|) sum_{x in D_{-k}} 1{a_{k,i}^T x > 0} = 0}
```

**Properties**:
- More aggressive: capsules specialized for own domain fire less on cross data
- Detects capsules that would be dead in a composed setting
- Requires access to other domain's data

### 3.3 Joint-Data Profiling (Pipeline B3)

Profile domain k model on D_joint:

```
S_k^{joint} = {i : (1/|D_joint|) sum_{x in D_joint} 1{a_{k,i}^T x > 0} = 0}
```

**Properties**:
- Best matches Pipeline A's profiling distribution
- Requires joint data (partial parallelism loss)
- Expected to be closest to Pipeline A's pruning decisions

---

## 4. Computational Cost Analysis

### 4.1 Profiling Cost

Pipeline A (composed, joint data):
```
Cost_A = n_batches * B * T * L * (P * N_d) * d * 2
       = 20 * 32 * 32 * 4 * 256 * 64 * 2
       = 5.4G FLOPs
```

Pipeline B (per-domain, own data, parallelizable):
```
Cost_B_per_domain = n_batches * B * T * L * P * d * 2
                  = 20 * 32 * 32 * 4 * 128 * 64 * 2
                  = 2.7G FLOPs per domain

Cost_B_total = 2 * 2.7G = 5.4G FLOPs (sequential)
             = 2.7G FLOPs (parallel across 2 domains)
```

Pipeline B achieves 2x speedup when profiling is parallelized.

### 4.2 Composition Cost

Pipeline A composes full P*N_d capsules per layer, then prunes:
```
Composed model size: P * N_d = 256 capsules per layer
After pruning: ~109 capsules per layer (57% pruned)
```

Pipeline B composes already-pruned models:
```
Pre-pruned: ~94 capsules per domain (62% pruned)
Composed model size: ~188 capsules per layer
```

Pipeline B's composed model is 26% smaller than Pipeline A's pre-pruning
composed model. This saves memory during calibration.

### 4.3 Calibration Cost

Same for both pipelines: 100 steps of gradient descent on joint data.
But Pipeline B operates on a smaller model (188 vs 256 capsules per layer
before pruning, 188 vs 109 after pruning). The FLOPs per calibration step
differ by the capsule count ratio.

### 4.4 Total Pipeline Cost

```
Pipeline A:
  Profile (composed): 5.4G
  Prune: <1ms
  Calibrate (pruned): 100 steps * ~14K FLOPs/token * B * T = ~1.4G
  Total: ~6.8G FLOPs

Pipeline B (parallel profiling):
  Profile (per-domain, parallel): 2.7G (wall-clock)
  Prune: <1ms
  Compose: <1ms
  Calibrate: 100 steps * ~24K FLOPs/token * B * T = ~2.5G
  Total: ~5.2G FLOPs (wall-clock), ~7.9G (total)
```

Pipeline B saves ~24% wall-clock FLOPs through parallel profiling.

---

## 5. Worked Numerical Example

At d=4, P=4 per domain (toy scale), N_d=2:

### 5.1 Single-Domain Profiling

Domain A model evaluated on D_A:
```
Capsule 0: f=0.6 (alive, frequent detector)
Capsule 1: f=0.0 (dead -- learned negative bias during training)
Capsule 2: f=0.3 (alive, moderate detector)
Capsule 3: f=0.0 (dead -- dying ReLU)
```

S_A^{own} = {1, 3}. Prune capsules 1, 3.
Pruned A model: 2 alive capsules.

Domain B model evaluated on D_B:
```
Capsule 0: f=0.0 (dead)
Capsule 1: f=0.5 (alive)
Capsule 2: f=0.0 (dead)
Capsule 3: f=0.4 (alive)
```

S_B^{own} = {0, 2}. Prune capsules 0, 2.
Pruned B model: 2 alive capsules.

### 5.2 Compose Pruned Models

```
Composed A_pruned:
  [a_{A,0}, a_{A,2}]   -- 2 capsules from domain A

  concatenated with

Composed B_pruned:
  [a_{B,1}, a_{B,3}]   -- 2 capsules from domain B

Total: 4 capsules (50% of original 8)
```

### 5.3 Compare with Pipeline A

If Pipeline A composed first, it would have 8 capsules, then profile
on joint data might find 5 dead (62.5%), leaving 3 alive.

Pipeline B found 4 dead across both domains (50%), leaving 4 alive.

The slight difference (4 vs 3 alive) comes from the ~6% capsules that
are alive in single-domain but dead in composed setting. After calibration,
both produce equivalent quality because these extra capsules contributed
noise (cross-domain activations), not signal.

---

## 6. Assumptions

1. **Dead capsule identity stability (Exp 16).** The pipeline relies on
   Jaccard=0.895 between single-domain and composed dead sets. At N_d >> 2,
   the additive perturbation from many domains could reduce this overlap.

2. **Calibration compensates for pruning differences.** The 6pp gap in
   pruning aggressiveness between Pipeline B and A is compensated by
   100 steps of calibration. If calibration budget is reduced, the gap
   might manifest as quality loss.

3. **ReLU activation function.** The zero-change pruning theorem requires
   truly zero activations. SiLU, GELU, etc. have no dead neurons, so this
   pipeline is ReLU-specific.

4. **Binary profiling at f=0.** Capsules with 0 < f < epsilon are kept.
   A more nuanced threshold could change the pruning/quality tradeoff.

5. **Two domains.** The experiment uses N_d=2. Scaling to N_d=5+ may change
   the pruning gap and require re-validation.

6. **Same calibration budget.** Both pipelines get identical calibration
   (100 steps, lr=3e-4). The experiment isolates the effect of pruning
   order, not calibration budget.

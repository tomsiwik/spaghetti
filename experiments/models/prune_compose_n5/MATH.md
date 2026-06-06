# Pre-Composition Pruning at N=5: Mathematical Foundations

## 1. Problem Statement

Validate that the pre-composition pruning pipeline (profile each domain
independently, prune, then compose) produces equivalent quality to
compose-then-prune at N=5 domains, where identity Jaccard has degraded
from 0.895 (N=2) to 0.792.

```
Pipeline A (baseline):  Compose -> Profile -> Prune -> Calibrate
Pipeline B (proposed):  Profile -> Prune -> Compose -> Calibrate
```

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
N_d       -- number of domains (5 for quintary split)
L         -- number of transformer layers (4)

A_k in R^{P x d}     -- detector matrix for domain k (rows are a_{k,i}^T)
B_k in R^{d x P}     -- expansion matrix for domain k (columns are b_{k,i})

D_k                   -- domain k's validation dataset
D_joint               -- joint validation dataset (union of all domains)

f_{k,i}^{own}        -- activation freq of capsule i in domain k model on D_k
f_{c,i}^{joint}      -- activation freq of capsule i in composed model on D_joint

S_k^{own}            -- dead set from own-domain profiling: {i : f_{k,i}^{own} = 0}
S_c                   -- dead set from composed-model profiling: {i : f_{c,i}^{joint} = 0}

J(A,B)               -- Jaccard similarity: |A & B| / |A | B|
O(A,B)               -- Overlap coefficient: |A & B| / min(|A|, |B|)
```

---

## 2. Scaling from N=2 to N=5

### 2.1 Perturbation Growth with N

The composed model's layer l forward pass is:

```
y_composed(x) = sum_{k=1}^{N_d} B_k @ ReLU(A_k @ x_l)
```

For capsule (k, i), the pre-activation is:

```
z_{k,i}(x) = a_{k,i}^T x_l
```

In the composed model, x_l includes residual contributions from all
domains' capsules in earlier layers. The perturbation delta_l from
composition is:

```
x_l^{composed} = x_l^{single} + delta_l
delta_l = sum_{j != k} sum_{m} b_{j,m} * ReLU(a_{j,m}^T x_{l-1}^{composed})
```

The number of perturbation terms scales with (N_d - 1). At N=2,
delta_l comes from 1 other domain. At N=5, delta_l comes from 4 other
domains. If the perturbation from each domain is approximately
independent and of similar magnitude epsilon:

```
||delta_l||_2 ~ sqrt(N_d - 1) * epsilon    (random walk scaling)
```

or, in the worst case:

```
||delta_l||_2 <= (N_d - 1) * epsilon        (adversarial alignment)
```

### 2.2 Jaccard Degradation Model

Exp n5_identity_scaling measured the trajectory:

```
N=2: J = 0.871 +/- 0.032
N=3: J = 0.853 +/- 0.045
N=4: J = 0.822 +/- 0.060
N=5: J = 0.792 +/- 0.054
```

The approximately linear degradation rate is:

```
dJ/dN ~ -0.026 per additional domain
```

This suggests the perturbation grows sub-linearly (closer to sqrt(N)
than N), which is consistent with random-walk scaling of uncorrelated
domain perturbations.

### 2.3 Impact on Pre-Composition Pruning

The capsules affected by Pipeline B vs Pipeline A fall into three
categories:

1. **Dead in both** (S_k^{own} & S_c^{k-half}): Correctly pruned
   by both pipelines. This is the majority (~79% at N=5).

2. **Dead only in A** (S_c^{k-half} \ S_k^{own}): Capsules that die
   under composition perturbation but are alive in single-domain.
   Pipeline B misses these. These are "composition-killed" capsules.
   At N=5, ~10.5% of capsules per domain fall here (up from ~6% at N=2).

3. **Dead only in B** (S_k^{own} \ S_c^{k-half}): Capsules that are
   dead in single-domain but revive under composition. Very rare (~1-2%).

### 2.4 Why Quality Equivalence Persists

For the capsules in category 2 (missed by Pipeline B):

```
contribution_{missed} = sum_{i in cat2} b_{k,i} * ReLU(a_{k,i}^T x)
```

These capsules are:
- Dead for their own domain data (f_{k,i}^{own} = 0)
- Alive only under composition perturbation (cross-domain inputs)
- NOT trained on the data that activates them
- Therefore producing noise, not signal

After calibration (100 steps on joint data), the router learns to
route around these noisy contributions. The quality bound from
MATH.md of prune_before_compose still applies:

```
|L_A - L_B| <= C * |S_B \ S_A| * epsilon_margin
```

At N=5, |S_B \ S_A| is larger (more missed capsules), but
epsilon_margin (activation magnitude of these cross-domain-only
capsules) remains small. The product stays well below the 3% threshold.

---

## 3. Pruning Ratio Analysis at N=5

### 3.1 Expected Pruning Rates

Single-domain death rates at N=5 (quintary split, 3-seed mean):

```
a-e: 51.4%
f-j: 46.2%
k-o: 52.0%
p-t: 48.4%
u-z: 44.1%
```

Pipeline A (compose-then-prune) profiles the composed N*P = 640
capsules per layer on joint data. Capsules see all 5 domains' data,
so some cross-domain activations prevent pruning of capsules that
are dead for their own domain.

Pipeline B (prune-before-compose) profiles each P=128 capsule pool
on only its own domain data. Capsules see only own-domain data,
producing more aggressive pruning.

### 3.2 Pruning Gap Direction

At N=2, Pipeline B pruned MORE (61.2% vs 55.2%, gap +6pp). This is
because own-domain profiling does not see cross-domain inputs that
cause incidental activations.

At N=5, the same logic applies: Pipeline B sees only 1/5 of the data
distribution, so it finds fewer cross-domain activations and prunes
more aggressively per domain.

However, the composed model at N=5 has more domains contributing
perturbation, potentially killing MORE capsules (expanding the
composed dead set). This can flip the gap direction depending on
the balance between:
- More own-domain pruning (B prunes more)
- More composition-killed capsules (A prunes more of these)

### 3.3 Size Advantage

Pipeline B's composed model is smaller:

```
Pipeline A model size (before pruning): N_d * P = 5 * 128 = 640 capsules/layer
Pipeline A after pruning: ~997/4 = 249 capsules/layer
Pipeline B composed size: ~1141/4 = 285 capsules/layer (already pruned)
```

At N=5, Pipeline B's pre-pruned models produce a composed model
that is smaller than unpruned but larger than Pipeline A's post-prune
result. The gap in final alive capsules is ~14% (1141 vs 997).

---

## 4. Computational Cost at N=5

### 4.1 Profiling Cost

Pipeline A (composed, joint data):
```
Cost_A = n_batches * B * T * L * (P * N_d) * d * 2
       = 20 * 32 * 32 * 4 * 640 * 64 * 2
       = 13.4G FLOPs
```

Pipeline B (per-domain, own data, parallelizable):
```
Cost_B_per_domain = n_batches * B * T * L * P * d * 2
                  = 20 * 32 * 32 * 4 * 128 * 64 * 2
                  = 2.7G FLOPs per domain

Cost_B_total = 5 * 2.7G = 13.4G FLOPs (sequential)
             = 2.7G FLOPs (parallel across 5 domains)
```

Pipeline B achieves 5x speedup when profiling is fully parallelized.
At N=5, this advantage is more significant than at N=2 (2x).

### 4.2 Calibration Cost

Both pipelines calibrate for 100 steps on joint data, but:
- Pipeline A operates on ~249 capsules/layer (post-prune)
- Pipeline B operates on ~285 capsules/layer (pre-pruned composition)

Pipeline B's calibration is ~14% more expensive per step due to
more capsules. Total calibration cost difference is small.

### 4.3 Total Pipeline Cost (Wall-Clock)

```
Pipeline A:
  Compose: <1ms
  Profile (composed): 13.4G
  Prune: <1ms
  Calibrate: 100 steps * ~16K FLOPs/token = ~1.6G
  Total: ~15.0G FLOPs

Pipeline B (parallel profiling):
  Profile (per-domain, parallel): 2.7G (wall-clock)
  Prune: <1ms per domain
  Compose: <1ms
  Calibrate: 100 steps * ~18K FLOPs/token = ~1.9G
  Total: ~4.6G FLOPs (wall-clock)
```

Pipeline B saves ~69% wall-clock FLOPs at N=5 through parallel
profiling. The savings grow linearly with N (at N=2 it was ~24%).

---

## 5. Worked Numerical Example

At d=4, P=4, N_d=5 (toy scale):

### 5.1 Single-Domain Profiling (5 domains)

```
Domain A: capsules [0,1,2,3] -> dead = {1,3} -> prune to [0,2]
Domain B: capsules [0,1,2,3] -> dead = {0,2} -> prune to [1,3]
Domain C: capsules [0,1,2,3] -> dead = {0,1} -> prune to [2,3]
Domain D: capsules [0,1,2,3] -> dead = {2,3} -> prune to [0,1]
Domain E: capsules [0,1,2,3] -> dead = {1}   -> prune to [0,2,3]
```

### 5.2 Pipeline B: Compose Pruned Models

```
Composed = [A_0, A_2, B_1, B_3, C_2, C_3, D_0, D_1, E_0, E_2, E_3]
Total: 11 capsules (55% of 20 original)
```

### 5.3 Pipeline A: Compose Then Prune

```
Composed (full) = [A_0..3, B_0..3, C_0..3, D_0..3, E_0..3]
Total: 20 capsules
After joint profiling: suppose 12 dead -> 8 alive (60% pruned)
```

Pipeline A finds 3 additional dead capsules that Pipeline B missed
(capsules that are alive for own domain but dead under composition).
After calibration, both reach the same quality.

---

## 6. Assumptions

1. **Identity overlap sufficient at N=5.** The pipeline relies on
   Jaccard=0.792 at N=5 (vs 0.895 at N=2). The experiment tests
   whether this degraded overlap still permits quality-equivalent
   pre-composition pruning.

2. **Calibration compensates for missed capsules.** The 5.6pp pruning
   gap (B prunes less than A at N=5) means Pipeline B retains ~144
   more capsules. These extra capsules are noise (cross-domain
   activations), and calibration learns to ignore them.

3. **ReLU activation function.** As with prune_before_compose, this
   pipeline is specific to ReLU. SiLU has no dead neurons.

4. **Binary profiling at f=0.** Same threshold as N=2 experiment.

5. **Five domains (quintary split).** The domains are character-level
   name subsets split by first letter (a-e, f-j, k-o, p-t, u-z).
   Domain sizes are unequal (2,359 to 10,479 names).

6. **Same calibration budget.** Both pipelines get identical
   calibration (100 steps, lr=3e-4). The experiment isolates the
   effect of pruning order at higher N, not calibration budget.

# Per-Capsule Revival Tracking: Mathematical Foundations

## 1. Problem Statement

Experiment 17 established that aggregate ReLU death rate follows a
"spike and slow decay" trajectory: 18.8% at initialization, spiking to
55.1% at S=50, then slowly declining to 47.3% by S=3200. The aggregate
decrease of ~8 percentage points was attributed to "neural revival" via
inter-layer coupling.

But aggregate rates mask per-capsule dynamics. The critical question:

**Q: Does the aggregate death decrease reflect the SAME capsules reviving,
or population turnover (different capsules dying and reviving)?**

Three possible dynamics:
- **Sticky death**: Once dead, capsules stay dead. The aggregate decrease
  comes from fewer NEW deaths, not revival of old dead capsules.
  Jaccard(dead_100, dead_3200) ~ 1.0.
- **True revival**: Dead capsules at S=100 become alive at later checkpoints
  through inter-layer coupling. Jaccard(dead_100, dead_3200) << 1.0, with
  the S=100 cohort showing substantial revival.
- **Population turnover**: Different capsules cycle through dead/alive
  states at each checkpoint. Jaccard is moderate (~0.7-0.8), but the
  S=100 dead cohort shows both revival AND new deaths replacing them.

---

## 2. Notation

All notation follows training_duration/MATH.md and pruning_controls/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
P_total   -- total capsules across layers = P * L = 512
S         -- number of fine-tuning steps

a_i(S) in R^d  -- detector vector for capsule i after S steps
f_i(S)         -- activation frequency of capsule i after S steps

D_S = {i : f_i(S) = 0}  -- set of dead capsule indices at step S
A_S = {i : f_i(S) > 0}  -- set of alive capsule indices at step S

|D_S| + |A_S| = P_total  for all S
```

---

## 3. Transition Matrix Formalism

### 3.1 Per-Interval Transitions

Between consecutive checkpoints S_1 and S_2, each capsule undergoes
one of four transitions:

```
         At S_2:
         Dead    Alive
At S_1:
Dead     n_DD    n_DA     (dead->dead, dead->alive)
Alive    n_AD    n_AA     (alive->dead, alive->alive)
```

Where:
- n_DD = |D_{S_1} & D_{S_2}|  (persistently dead)
- n_DA = |D_{S_1} & A_{S_2}|  (revived)
- n_AD = |A_{S_1} & D_{S_2}|  (newly dead)
- n_AA = |A_{S_1} & A_{S_2}|  (persistently alive)

Conservation: n_DD + n_DA + n_AD + n_AA = P_total

### 3.2 Revival and Death Rates

Per-interval revival rate:
```
r_revival(S_1, S_2) = n_DA / |D_{S_1}|
```

Per-interval new death rate:
```
r_death(S_1, S_2) = n_AD / |A_{S_1}|
```

The aggregate death rate change between S_1 and S_2 decomposes as:
```
delta(S_2) - delta(S_1) = (n_AD - n_DA) / P_total
```

If n_DA > n_AD, the aggregate death rate decreases (more revivals than
new deaths). This is what Exp 17 observed for S > 400.

### 3.3 Cumulative vs Per-Interval

A capsule may undergo multiple transitions across checkpoints:
dead at S=100, alive at S=200, dead again at S=400 (flickering).
Per-interval analysis captures this; cohort analysis does not.

---

## 4. Jaccard Similarity

### 4.1 Definition

```
J(D_{S_1}, D_{S_2}) = |D_{S_1} & D_{S_2}| / |D_{S_1} | D_{S_2}|
```

Properties:
- J = 1.0: identical dead sets
- J = 0.0: completely disjoint dead sets
- J > 0.85: death is "sticky" (little capsule identity change)
- J < 0.70: substantial turnover

### 4.2 Expected Values Under Different Hypotheses

**Hypothesis A (Sticky death, J > 0.85)**:
```
|D_{S_1} & D_{S_2}| ~ min(|D_{S_1}|, |D_{S_2}|) - small_turnover
```
The dead set is nearly a subset relation: D_3200 subset D_100 minus
a few revivals. The aggregate decrease comes from a small number of
capsules reviving.

**Hypothesis B (True revival, J ~ 0.5-0.7)**:
```
|D_{S_1} & D_{S_2}| << min(|D_{S_1}|, |D_{S_2}|)
```
Many capsules in D_100 are no longer in D_3200, and many capsules in
D_3200 were not in D_100. Substantial reshuffling.

**Hypothesis C (Random turnover)**:
If dead/alive status at each checkpoint were independent (null model):
```
E[J] = (delta_1 * delta_2) / (delta_1 + delta_2 - delta_1 * delta_2)
```
At delta_1 = 0.55, delta_2 = 0.47:
```
E[J_null] = (0.55 * 0.47) / (0.55 + 0.47 - 0.55 * 0.47) = 0.259 / 0.762 = 0.340
```

A measured Jaccard significantly above 0.340 indicates death is
stickier than random assignment, confirming some persistence.

---

## 5. Cohort Analysis

### 5.1 Anchor Cohort

Define the anchor cohort as C_100 = D_100 (all capsules dead at S=100).

For each later checkpoint S:
```
survived(S) = |C_100 & D_S| / |C_100|    (fraction still dead)
revived(S)  = |C_100 & A_S| / |C_100|    (fraction that revived)

survived(S) + revived(S) = 1.0 for all S >= 100
```

### 5.2 Survival Curve Interpretation

The survival curve survived(S) tells us directly: of the capsules
that were dead at the aggregate peak (S=100), how many remain dead?

- If survived(3200) ~ 1.0: sticky death, the cohort never revives
- If survived(3200) ~ delta(3200)/delta(100) ~ 0.85: proportional
  decrease, no special cohort effect
- If survived(3200) << 0.85: the S=100 cohort revives faster than
  the background rate (true targeted revival)

### 5.3 Contribution to Aggregate Decrease

The aggregate death decrease from S=100 to S=3200 is:
```
delta(100) - delta(3200) ~ 55.5% - 47.3% = 8.2 pp (from Exp 17)
```

This has two additive sources:
```
revival_contribution = (1 - survived(3200)) * |C_100| / P_total
new_death_avoided    = aggregate_decrease - revival_contribution
```

If revival_contribution >> new_death_avoided: true revival dominates.
If revival_contribution << new_death_avoided: the decrease is mainly
because later training produces fewer new deaths (gradient shrinkage).

---

## 6. Experimental Design

### 6.1 Checkpoints

Same step counts as Exp 17: S in {0, 50, 100, 200, 400, 800, 1600, 3200}

### 6.2 Profiling Protocol

Identical to Exp 17: profile_activations() with 20 batches of 32 on
domain validation data. The profiling dataset is constant across all
checkpoints. Only the model weights change.

### 6.3 Identity Tracking

At each checkpoint, record the per-capsule binary dead/alive mask.
Capsules are identified by (layer_index, capsule_index) which is
stable across checkpoints because the model architecture is fixed.

Key: all step counts share the same training seed, so S=50 is literally
the first 50 steps of the S=3200 trajectory. The "same capsule" at
different checkpoints has the same architectural position but different
weights (from different amounts of training).

### 6.4 Statistical Design

- 3 seeds (42, 123, 7) as in all prior experiments
- Report transition matrices, cohort survival, and Jaccard at each
  checkpoint pair
- Aggregate across seeds for all metrics

---

## 7. Kill Criteria

| Criterion | Threshold | What it means |
|-----------|-----------|---------------|
| Jaccard(dead_100, dead_3200) > 0.85 | >0.85 | Death is sticky; revival negligible |
| Max revival rate per interval < 5% | <5% | Inter-layer coupling too weak |
| Total turnover events < 10/seed | <10 | Dynamics too sparse at micro scale |

---

## 8. Worked Numerical Example

At d=4, P=4, L=1 (4 capsules total, single layer):

### S=100
```
Capsule 0: dead   (a_0^T x <= 0 for all x)
Capsule 1: dead
Capsule 2: alive  (fires on 40% of inputs)
Capsule 3: alive  (fires on 60% of inputs)
D_100 = {0, 1}, A_100 = {2, 3}, delta(100) = 50%
```

### S=200
```
Capsule 0: dead   (still dead, no gradient)
Capsule 1: alive  (REVIVED: earlier layers shifted x distribution)
Capsule 2: alive  (still alive)
Capsule 3: dead   (NEWLY DEAD: gradient update pushed past boundary)
D_200 = {0, 3}, A_200 = {1, 2}, delta(200) = 50%
```

### Transition matrix (S=100 -> S=200):
```
         Dead@200  Alive@200
Dead@100    1 (DD)    1 (DA)    <-- capsule 0 stays dead, capsule 1 revives
Alive@100   1 (AD)    1 (AA)    <-- capsule 3 dies, capsule 2 stays alive
```

Revival rate = 1/2 = 50% (of dead capsules at S=100, half revived)
New death rate = 1/2 = 50% (of alive capsules at S=100, half died)

Aggregate death rate unchanged (50% -> 50%), but capsule identity
completely reshuffled: Jaccard(D_100, D_200) = |{0}| / |{0,1,3}| = 0.33.

This demonstrates POPULATION TURNOVER: same aggregate rate, different
capsules.

### S=3200
```
Capsule 0: alive  (REVIVED at step 2000, stable since)
Capsule 1: alive  (revived at S=200, stayed alive)
Capsule 2: alive  (always alive)
Capsule 3: dead   (died at S=200, stayed dead)
D_3200 = {3}, A_3200 = {0, 1, 2}, delta(3200) = 25%
```

Cohort analysis: C_100 = {0, 1}
- survived(3200) = |{0,1} & {3}| / 2 = 0/2 = 0%
- revived(3200) = |{0,1} & {0,1,2}| / 2 = 2/2 = 100%

Jaccard(D_100, D_3200) = |{}| / |{0,1,3}| = 0.0

This extreme example shows complete cohort revival with new death
elsewhere. The aggregate decrease (50% -> 25%) comes entirely from
the original cohort reviving.

---

## 9. Assumptions

1. **Capsule identity is stable.** Each capsule is identified by
   (layer, index). The architecture does not change across checkpoints,
   so the same index always refers to the same row of A / column of B.

2. **Same training trajectory.** All checkpoints use the same seed,
   so S=100 is literally the first 100 steps of the S=3200 run. This
   means transitions are genuine temporal evolution, not confounded by
   different random trajectories.

3. **Binary dead/alive classification.** We use f_i = 0 as the death
   criterion (same as Exp 9/10/17). "Nearly dead" capsules (f_i < 0.01)
   are classified as alive. This is conservative: some "alive" capsules
   may be functionally irrelevant.

4. **Profiling dataset is representative.** The 20-batch, 32-sample
   profiling protocol from Exp 9/10/17 captures the relevant activation
   patterns. Small profiling set may miss rare activations, slightly
   overestimating death.

5. **Inter-layer coupling is the revival mechanism.** Exp 17 hypothesized
   that dead neuron revival occurs because weight updates in earlier layers
   shift the input distribution to later layers. This experiment tests
   whether revival actually happens at the per-capsule level; it does not
   directly test the mechanism (that would be Exp 20: layer freezing).

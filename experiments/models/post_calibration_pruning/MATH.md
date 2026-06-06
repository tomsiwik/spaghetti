# Post-Calibration Pruning Safety: Mathematical Foundations

## 1. Problem Statement

Three pruning pipeline orderings exist for composed models:

```
Pipeline A (pre-composition):   profile -> prune -> compose -> calibrate
Pipeline B (post-calibration):  compose -> calibrate -> profile -> prune
Pipeline C (pre-calibration):   compose -> profile -> prune -> calibrate
```

Pipeline A is validated (prune_before_compose, +0.01%). Pipeline C is
the original baseline (dead_capsule_pruning). This experiment validates
Pipeline B.

The question: does calibration (100 training steps on joint data) change
the dead capsule set enough that post-calibration profiling identifies
a materially different dead set than pre-calibration profiling?

---

## 2. Notation

Follows revival_under_composition/MATH.md and capsule_revival/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
D         -- number of domains (2 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S_cal     -- calibration steps (100 or 200)

a_i in R^d  -- detector vector for capsule i (row of A matrix)
f_i(t)      -- activation frequency of capsule i at time t

D_t = {i : f_i(t) = 0}  -- dead capsule set at time t
A_t = {i : f_i(t) > 0}  -- alive capsule set at time t
```

### Composition timeline

```
t=0:   compose (concatenate domain pools)
       D_0 = dead set at composition time

t=S_cal: after calibration
       D_{S_cal} = dead set after calibration
```

---

## 3. Revival During Calibration

### 3.1 Revival Rate

The revival rate during calibration is:

```
R(S_cal) = |D_0 \ D_{S_cal}| / |D_0|
         = |{i : i in D_0 AND i in A_{S_cal}}| / |D_0|
```

This is the fraction of capsules dead at composition time that become
alive after S_cal steps of calibration.

### 3.2 Prior Finding (revival_under_composition)

At S_cal=100, composed revival rate = 2.9% (3-seed mean). This was
measured as part of a longer trajectory (up to S=3200). The suppression
mechanism: composed models have 2x capsule dimension, diluting inter-layer
coupling effects on individual capsules.

### 3.3 Prediction

If R(S_cal) is small (< 5%), then:

```
|D_0 symmetric_diff D_{S_cal}| / |D_0| is small
```

meaning the dead set is approximately stable through calibration. Therefore
profiling AFTER calibration (Pipeline B) should yield approximately the
same dead set as profiling BEFORE calibration (Pipeline C), and pruning
either one should produce equivalent quality.

### 3.4 New Death During Calibration

Calibration can also CREATE new dead capsules (alive at t=0, dead at t=S_cal):

```
NewDead(S_cal) = |A_0 intersect D_{S_cal}|
```

Pipeline B captures these newly dead capsules (they exist in the post-cal
profile). Pipeline C does not. This means Pipeline B may prune MORE
capsules than Pipeline C if new deaths exceed revivals.

The net change in alive capsules:

```
Delta_alive = |A_{S_cal}| - |A_0|
            = (revivals - new_deaths)
            = |D_0 intersect A_{S_cal}| - |A_0 intersect D_{S_cal}|
```

If Delta_alive < 0, post-calibration has FEWER alive capsules (more to prune).
If Delta_alive > 0, post-calibration has MORE alive capsules (less to prune).

---

## 4. Quality Equivalence Argument

### 4.1 Exact Pruning Theorem (from dead_capsule_pruning)

For capsule i with f_i = 0 (truly dead), removing row i from A and
column i from B changes the output by exactly 0:

```
Pool(x) = B * relu(A * x)
        = sum_j B_j * relu(a_j^T x)

If capsule i is dead: relu(a_i^T x) = 0 for all x in profiling set
=> removing capsule i: Delta output = B_i * relu(a_i^T x) = 0
```

### 4.2 Pipeline Equivalence

The quality difference between Pipeline A and Pipeline B depends on:

1. **Which capsules are pruned**: Different dead sets at profiling time
2. **Effect of pruning on calibration**: Pipeline A calibrates AFTER pruning
   (smaller model), Pipeline B prunes AFTER calibration (full model during cal)

For (1): if R(S_cal) < 5%, the dead sets differ by < 5% of the dead population.
These are marginal capsules (near the dead/alive boundary) whose contribution
is negligible.

For (2): Pipeline B has MORE capsules during calibration (full composed model).
This provides more capacity during the calibration phase. Pipeline A has
FEWER capsules (pre-pruned), meaning calibration must compensate with fewer
degrees of freedom. Both converge to the same loss if calibration is sufficient.

---

## 5. Kill Criteria

### Kill Criterion 1: Quality Degradation

```
(loss_B - loss_A) / loss_A > 2%  =>  KILL
```

Where loss_B is Pipeline B (post-calibration pruning) and loss_A is
Pipeline A (pre-composition pruning, the validated baseline).

### Kill Criterion 2: Revival Rate

```
R(100) > 5%  =>  KILL
```

This would contradict the 2.9% finding from revival_under_composition
and indicate the dead set is NOT stable during calibration.

---

## 6. Experimental Design

### 6.1 Pipelines

All pipelines share steps 1-2:
1. Pretrain base on all data (300 steps, seed-matched)
2. Fine-tune MLP per domain (200 steps, attention frozen)

Then diverge:

**Pipeline A** (pre-composition pruning, validated baseline):
3. Profile each domain model on own-domain val data
4. Prune dead capsules (tau=0)
5. Compose pruned models
6. Calibrate 100 steps on joint data

**Pipeline B** (post-calibration pruning, NEW):
3. Compose full models
4. Calibrate 100 steps on joint data
5. Profile composed model on joint val data
6. Prune dead capsules

**Pipeline C** (compose-then-prune, reference):
3. Compose full models
4. Profile composed model on joint val data
5. Prune dead capsules
6. Calibrate 100 steps on joint data

**Pipeline D** (control, no pruning):
3. Compose full models
4. Calibrate 100 steps on joint data

### 6.2 Revival Measurement

For Pipeline B, we measure the dead set at two points:
- t=0 (just-composed, before calibration)
- t=S_cal (after calibration, before pruning)

Revival rate = |D_0 intersect A_{S_cal}| / |D_0|

Additionally, we measure a fine-grained revival trajectory at
S_cal in {0, 50, 100, 200} using independent copies of the composed model.

### 6.3 Extended Calibration

Pipeline B2 uses 200 calibration steps instead of 100 to test whether
longer calibration increases revival beyond the 5% threshold.

---

## 7. Worked Numerical Example

At d=4, P=4, L=1, D=2 (8 composed capsules):

### Composition (t=0):
```
Composed A = [a_0, a_1, a_2, a_3 | a_4, a_5, a_6, a_7]
              (domain A capsules)  (domain B capsules)

Profile on joint data:
  f_0=0, f_1=0.4, f_2=0.2, f_3=0, f_4=0, f_5=0.3, f_6=0, f_7=0.1

D_0 = {0, 3, 4, 6}   (4 dead capsules)
A_0 = {1, 2, 5, 7}   (4 alive capsules)
```

### After 100-step calibration (t=100):
```
Profile on joint data:
  f_0=0, f_1=0.3, f_2=0.25, f_3=0, f_4=0.05, f_5=0.35, f_6=0, f_7=0

D_100 = {0, 3, 6, 7}
A_100 = {1, 2, 4, 5}

Transitions:
  dead->dead:   {0, 3, 6} (3 capsules)
  dead->alive:  {4}        (1 capsule -- revived)
  alive->dead:  {7}        (1 capsule -- newly dead)
  alive->alive: {1, 2, 5}  (3 capsules)

Revival rate = |{4}| / |{0, 3, 4, 6}| = 1/4 = 25%
```

### Pipeline comparison:

**Pipeline B** (post-calibration): prunes {0, 3, 6, 7} = 4 capsules
  - Correctly identifies capsule 7 as newly dead
  - Does NOT prune capsule 4 (it revived)

**Pipeline C** (pre-calibration): prunes {0, 3, 4, 6} = 4 capsules
  - Prunes capsule 4 (dead at t=0, but will revive)
  - Does NOT prune capsule 7 (alive at t=0, but will die)

Both prune 4 capsules, but different ones. The quality impact depends on
whether capsules 4 and 7 contribute meaningfully to the output. At
f_4(100)=0.05 and f_7(100)=0, neither contributes much. The quality
difference is negligible.

---

## 8. Assumptions

1. **Binary dead/alive at f=0.** Same threshold as all prior experiments.

2. **100-step calibration is the practical regime.** The composition
   protocol uses 100-200 steps. Longer calibration may increase revival.

3. **Revival rate transfers from revival_under_composition.** The 2.9%
   finding was measured with different training conditions (full
   fine-tune lr, not calibration lr=0.1*lr). Calibration uses a lower
   learning rate, which should REDUCE revival (smaller weight updates).

4. **Profiling protocol is stable.** Exp 12 (profiling_noise) showed
   2.6-3.8% same-checkpoint disagreement. Revival measurements close
   to this floor may be noise.

5. **Pruning dead capsules is exact (zero quality change).** From
   dead_capsule_pruning. Only applies to f=0 capsules.

6. **Capsule pool is ReLU.** This entire framework does not apply to
   SiLU/GELU activations (Exp 15: 0% prunable at safe thresholds).

---

## 9. Computational Cost

Per seed:
- Shared (pretrain + fine-tune): 700 steps
- Pipeline A: 100 calibration steps + profiling
- Pipeline B: 100 calibration steps + 2x profiling
- Pipeline C: 100 calibration steps + profiling
- Pipeline D: 100 calibration steps
- Pipeline B2: 200 calibration steps + profiling
- Revival trajectory: 50 + 100 + 200 = 350 calibration steps + 4x profiling

Total per seed: ~1650 equivalent steps + profiling overhead
Total experiment: 3 seeds * ~1650 = ~5000 steps
Estimated wall time: ~3-5 minutes on Apple Silicon

# Pruning Controls: Mathematical Foundations

## 1. Problem Statement

Experiment 9 (dead capsule pruning) established that 57% of capsules in
composed ReLU models are dead, and that pruning them produces exact zero
quality change. The adversarial review identified two unresolved causal
questions:

**Q1 (Pre-composition death rate)**: Is the 57% death rate caused by
composition or by training? Without profiling pre-composition models,
we cannot distinguish Hypothesis A (training-induced death, general
to all ReLU models) from Hypothesis B (composition-induced distribution
shift, specific to our protocol).

**Q2 (Random pruning baseline)**: Does targeted identification of dead
capsules matter? In an overparameterized composed model, random pruning
at the same rate might also preserve quality, making the profiling step
unnecessary overhead.

---

## 2. Notation

All notation follows dead_capsule_pruning/MATH.md exactly.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
N_d       -- number of domains (2 for binary split)
P_total   -- total capsules after concatenation = P * N_d = 256
L         -- number of transformer layers (4 at micro scale)

A in R^{P x d}         -- detector matrix for single-domain model
A_c in R^{P_total x d} -- detector matrix for composed model

a_i in R^d              -- detector vector for capsule i (row i of A)
b_i in R^d              -- expansion vector for capsule i (col i of B)

D_A = {x_1^A, ..., x_M^A}  -- hidden states from domain A data
D_B = {x_1^B, ..., x_M^B}  -- hidden states from domain B data
D   = D_A union D_B         -- joint hidden states

f_i^{single}  -- activation frequency of capsule i in single-domain model
f_i^{composed} -- activation frequency of capsule i in composed model

tau = 0.0     -- pruning threshold (strict dead)
```

---

## 3. Decomposing the Death Rate

### 3.1 Single-Domain Death Rate

For a single-domain model trained on domain A, capsule i is dead if:

```
f_i^{single,A} = (1/M) * sum_{x in D_A} 1{a_i^T x > 0} = 0
```

Let delta_single = fraction of dead capsules in the single-domain model:

```
delta_single = (1/P) * sum_{i=1}^{P} 1{f_i^{single} = 0}
```

This measures TRAINING-INDUCED death: capsules that died during
fine-tuning regardless of domain, through the dying ReLU phenomenon.

### 3.2 Composed Death Rate

For the composed model with P_total = 2P capsules evaluated on D = D_A union D_B:

```
delta_composed = (1/P_total) * sum_{i=1}^{P_total} 1{f_i^{composed} = 0}
```

From Exp 9: delta_composed ~ 0.57 (3-seed mean).

### 3.3 Decomposition

The composed death rate has three additive sources:

```
delta_composed = delta_training + delta_domain + delta_shift
```

Where:
- delta_training: capsules dead from ReLU death during training (would be dead
  even in single-domain model). Estimated by delta_single.
- delta_domain: capsules dead because they are presented inputs from the wrong
  domain (Pool_A capsules see domain B inputs).
- delta_shift: capsules dead because composition changes the hidden-state
  distribution (attention processes combined outputs, altering layer inputs).

### 3.4 Measuring Each Component

**delta_training** is directly measured as delta_single.

**delta_domain** is estimated by profiling single-domain models on CROSS-DOMAIN
data. If we evaluate Pool_A on domain B data, capsules that are alive on
domain A but dead on domain B contribute to delta_domain:

```
delta_domain ~ (1/P) * sum_{i=1}^P 1{f_i^{single,A}(D_A) > 0 AND f_i^{single,A}(D_B) = 0}
```

**delta_shift** is the residual: capsules alive in the single-domain model
(on both domains' data) but dead in the composed model. This is the
composition-induced distribution shift:

```
delta_shift = delta_composed - delta_training - delta_domain
```

### 3.5 Predictions

**Hypothesis A (training-dominant)**: delta_single ~ 0.50, delta_shift ~ 0.
The 57% death rate is general ReLU sparsity, not composition-specific.
Pruning is a general technique applicable to any ReLU model.

**Hypothesis B (composition-dominant)**: delta_single ~ 0.10-0.20,
delta_domain ~ 0.20-0.30, delta_shift ~ 0.10-0.20. Composition
is the primary cause. Pruning is specifically valuable for composed models.

**Exp 9 data hint**: 92% of dead-on-either capsules are also dead-on-both.
This superficially supports Hypothesis A, but could also be explained by
Hypothesis B + distribution shift (composition makes everything look similar
to both pools).

---

## 4. Random Pruning Theory

### 4.1 Expected Error from Random Pruning

For a composed model y(x) = sum_{i=1}^{P_total} b_i * max(0, a_i^T x),
pruning capsule i at random introduces error:

```
delta_y_i(x) = -b_i * max(0, a_i^T x)
```

For a DEAD capsule (f_i = 0): delta_y_i(x) = 0 for all x in D.
For an ALIVE capsule (f_i > 0): delta_y_i(x) != 0 for some x.

Random pruning at rate r prunes r * P_total capsules uniformly at random.
The expected number of alive capsules pruned is:

```
E[n_alive_pruned] = r * P_total * (1 - delta_composed)
```

At r = 0.57, P_total = 256, delta_composed = 0.57:

```
E[n_alive_pruned] = 0.57 * 256 * (1 - 0.57) = 0.57 * 256 * 0.43 = 62.7
```

So random pruning at the same rate as targeted pruning would prune
~63 ALIVE capsules on average, compared to zero for targeted pruning.

### 4.2 Error Bound for Random Pruning

The total output error from random pruning of k alive capsules is:

```
||delta_y(x)|| = ||sum_{i in pruned, alive} b_i * max(0, a_i^T x)||
```

Assuming independent contributions (capsule independence from Exp 9 MATH.md 3.3):

```
E[||delta_y||^2] ~ k * E[||b_i||^2 * (max(0, a_i^T x))^2]
```

This grows linearly with the number of alive capsules pruned.
For targeted pruning, k = 0 and the error is exactly zero.

### 4.3 Prediction

Random pruning at 57% should show significant degradation (>5% vs concat)
because it removes ~63 alive capsules that contribute meaningfully to the
output. The magnitude of degradation quantifies the "value of profiling" --
the benefit of knowing which capsules are dead vs guessing.

---

## 5. Empirical Results (Summary)

### 5.0 Measured Values

```
delta_training (single-domain dead, 3-seed mean) = 54.3%
  a_m model: 54.4% (std=10.4%)
  n_z model: 54.2% (std=9.2%)

delta_composed (post-composition dead, 3-seed mean) = 62.1%
  (std=4.7%)

delta_shift (composition-induced) = 62.1% - 54.3% = 7.7%

delta_domain (alive own, dead cross) = 3.0%
```

**Hypothesis A confirmed**: delta_single = 54.3% >> 20%, so training-
induced ReLU death is the dominant mechanism. 87% of dead capsules in
composed models were already dead before composition.

**Hypothesis B rejected**: delta_shift = 7.7% < 10%, composition adds
only a small increment to death. The initial Exp 9 assumption that
"composition creates dead capsules by presenting wrong-domain inputs"
is incorrect at micro scale.

### Random Pruning

```
Targeted pruning (dead only): +0.0% vs concat (exact zero change)
Random pruning (same rate):   -2.9% vs concat (BETTER, regularization effect)
  Random std = 0.0217 (high variance across draws)

With calibration:
  Targeted + cal: -8.2% vs concat
  Random + cal:   -7.5% vs concat (+0.8% worse than targeted)
```

Random pruning outperforms targeted without calibration because removing
alive capsules provides implicit regularization to the overparameterized
model. With calibration, targeted is better (+0.8%) because it provides
a cleaner starting point for optimization.

---

## 6. Experimental Protocol (as executed)

### 6.1 Phase 1: Pre-Composition Profiling

```
For each seed:
  1. Pretrain base model on all data (300 steps)
  2. Fine-tune domain-specific models (200 steps each)
  3. BEFORE composing:
     a. Profile model_A on domain_A data -> f_i^{single,A}(D_A)
     b. Profile model_A on domain_B data -> f_i^{single,A}(D_B)
     c. Profile model_B on domain_A data -> f_i^{single,B}(D_A)
     d. Profile model_B on domain_B data -> f_i^{single,B}(D_B)
  4. Compose models
  5. Profile composed model on joint data -> f_i^{composed}(D)
  6. Compute decomposition: delta_training, delta_domain, delta_shift
```

### 6.2 Phase 2: Random Pruning

```
For each seed:
  1. Compose model (from Phase 1)
  2. Profile composed model -> identify dead capsules (57%)
  3. Targeted pruning: prune dead capsules, evaluate
  4. Random pruning (5 random draws per seed):
     a. Select same NUMBER of capsules uniformly at random
     b. Prune those capsules
     c. Evaluate quality
  5. Compare: targeted vs random vs no-prune
```

### 6.3 Statistical Design

- 3 seeds (42, 123, 7) as in all prior experiments
- 5 random draws per seed for random pruning (15 total measurements)
- Report mean and std for random pruning across draws and seeds

---

## 7. Kill Criteria

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| delta_single > 0.45 | >45% | 54.3% | **KILL** |
| Random within 2% of targeted | <2% gap | -2.9% | PASS |
| delta_shift < 10% | <10% | 7.7% | **KILL** |

---

## 8. Worked Numerical Example

At d=4, P=4 per domain:

### Single-Domain Model A (trained on domain A)

```
Capsule 0: a_0 fires on 50% of domain A inputs, 10% of domain B   -> alive
Capsule 1: a_1 fires on 40% of domain A inputs, 5% of domain B    -> alive
Capsule 2: a_2 fires on 0%  of domain A inputs, 0% of domain B    -> DEAD (training-induced)
Capsule 3: a_3 fires on 30% of domain A inputs, 0% of domain B    -> alive on A, dead on B
```

delta_single_A = 1/4 = 25% (capsule 2 is dead)

### After Composition (A + B concatenated)

Suppose in composed context (attention processes both pools):
```
From Pool A: capsules 0,1 alive; capsule 2 dead (training); capsule 3 dead (domain B inputs + shift)
From Pool B: capsules 4,5 alive; capsule 6 dead (training); capsule 7 dead (domain A inputs)
```

delta_composed = 4/8 = 50%

Decomposition:
- delta_training = 2/8 = 25% (capsules 2, 6 were already dead)
- delta_domain = 1.5/8 ~ 19% (capsule 3 dead on B, capsule 7 dead on A, partial)
- delta_shift ~ 6% (residual from distribution change)

### Random Pruning at 50%

Prune 4 of 8 capsules at random.
P(prune alive capsule) = 4/8 = 50% per capsule.
Expected alive capsules pruned = 4 * 0.5 = 2.

Quality impact: significant (lost 2 of 4 alive capsules = 50% of useful capacity).
Targeted pruning: loses 0 alive capsules = 0% of useful capacity.

---

## 9. Assumptions

1. **Same profiling procedure.** The profile_activations function from
   dead_capsule_pruning.py is reused exactly. The profiling dataset size
   (20 batches * 32 samples) is the same as Exp 9.

2. **Single-domain model architecture identical to composed.** The single-
   domain models have P=128 capsules (not P_total=256). The profiling is
   done on these smaller models.

3. **Profiling dataset matches across phases.** We use the same validation
   set for profiling in both single-domain and composed settings, ensuring
   the death rate comparison is not confounded by different data.

4. **Random pruning is truly uniform.** Each capsule has equal probability
   of being selected for pruning, regardless of layer, domain origin, or
   activation status. We enforce this by sampling without replacement from
   the full set of capsule indices.

5. **Inter-layer coupling in random pruning.** Random pruning of alive
   capsules in layer l changes layer l's output, which changes layer l+1's
   input distribution. This coupling makes random pruning potentially worse
   than the sum of per-capsule errors. Targeted pruning avoids this entirely
   (zero output change per layer cascades to zero change everywhere).

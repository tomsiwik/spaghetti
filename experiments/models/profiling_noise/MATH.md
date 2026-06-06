# Profiling Noise Quantification: Mathematical Foundations

## 1. Problem Statement

Experiment 18 reported that 28.1% of capsules dead at S=100 revive by
S=3200, with D->A transition rates accelerating from 5.8% to 15.9% per
interval. The adversarial review flagged a confound: with only 640 samples
per profiling run (20 batches x 32), borderline capsules near f=0 might
flicker between dead and alive classifications across different random
batches, inflating D->A transition counts.

**Q: How much of the observed D->A transition rate is genuine revival
vs sampling noise in the profiling procedure?**

---

## 2. Notation

All notation follows capsule_revival/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
P_total   -- total capsules across layers = P * L = 512
S         -- number of fine-tuning steps
N_prof    -- number of profiling samples = n_batches * batch_size = 640

a_i(S) in R^d  -- detector vector for capsule i after S steps
f_i(S)         -- activation frequency of capsule i after S steps

D_S = {i : f_i(S) = 0}  -- set of dead capsule indices at step S
A_S = {i : f_i(S) > 0}  -- set of alive capsule indices at step S
```

**New notation for dual profiling:**

```
f_i^A(S)  -- activation frequency from profiling run A (seed=1000)
f_i^B(S)  -- activation frequency from profiling run B (seed=2000)

D_S^A = {i : f_i^A(S) = 0}   -- dead set from run A
D_S^B = {i : f_i^B(S) = 0}   -- dead set from run B

D_S^C = D_S^A & D_S^B         -- consensus dead set (dead in BOTH runs)
A_S^C = complement of D_S^C   -- consensus alive set

F_S = (D_S^A symmetric_diff D_S^B)  -- flickering set (disagree between runs)
```

---

## 3. Noise Model

### 3.1 Source of Profiling Noise

A capsule i with true population activation probability p_i fires on
a sample x with probability p_i. Over N_prof samples, the observed
activation count n_i ~ Binomial(N_prof, p_i).

The observed frequency is f_i = n_i / N_prof. A capsule is classified
dead when f_i = 0, i.e., n_i = 0.

The probability of a false-dead classification (truly alive capsule
classified as dead) is:

```
P(f_i = 0 | p_i > 0) = (1 - p_i)^N_prof
```

For N_prof = 640:
- p_i = 0.001: P(false dead) = (0.999)^640 = 0.527 (coin flip)
- p_i = 0.005: P(false dead) = (0.995)^640 = 0.041
- p_i = 0.010: P(false dead) = (0.990)^640 = 0.0016
- p_i = 0.050: P(false dead) = (0.950)^640 < 1e-14

Critical insight: only capsules with p_i < ~0.005 have appreciable
false-dead probability. Capsules with p_i > 0.01 are classified
correctly with >99.8% reliability.

### 3.2 False Revival from Noise

A false D->A transition occurs when:
- At checkpoint S_1: f_i^run1 = 0 (classified dead), but p_i(S_1) > 0
- At checkpoint S_2: f_i^run2 > 0 (classified alive)

This happens when a borderline capsule (small p_i) happens to miss all
640 samples at S_1 but catches at least one at S_2. The probability is:

```
P(false D->A) = P(f_i(S_1) = 0) * P(f_i(S_2) > 0)
             = (1 - p_i(S_1))^N * (1 - (1 - p_i(S_2))^N)
```

### 3.3 Dual-Profiling Test

By profiling the same checkpoint twice with different random seeds,
we measure the noise floor directly:

```
|F_S| = |D_S^A symmetric_diff D_S^B|
```

Any element of F_S is a capsule whose classification changes with the
random seed alone (no weight change). This gives:

```
noise_rate(S) = |F_S| / P_total
```

If noise_rate is small (< 5%), then the profiling protocol is reliable
and D->A transitions between different checkpoints are dominated by
genuine weight changes, not sampling variance.

### 3.4 Consensus Correction

The consensus mask D_S^C = D_S^A & D_S^B is a conservative dead set:
a capsule is dead only if it is dead in both independent profiling runs.
This eliminates false-dead classifications at the cost of missing some
truly dead capsules with very small p_i.

The false-dead probability under consensus:

```
P(consensus false dead | p_i) = ((1 - p_i)^N_prof)^2 = (1 - p_i)^(2*N_prof)
```

This is equivalent to profiling with 2*N_prof = 1280 samples for the
death classification.

For p_i = 0.001: P = (0.999)^1280 = 0.278 (reduced from 0.527)
For p_i = 0.005: P = (0.995)^1280 = 0.0017 (reduced from 0.041)

---

## 4. Expected Disagreement Under Null Model

### 4.1 Binomial Model

Under the null model (no true death, all capsules have some p_i > 0),
the expected number of false-dead capsules per profiling run is:

```
E[|D_S^A|_false] = sum_i (1 - p_i)^N_prof
```

If K capsules have p_i in (0, 0.005) (the "borderline" population), the
expected false-dead count is approximately K * mean((1-p)^640) for p in
(0, 0.005).

The expected disagreement between two runs is approximately:

```
E[|F_S|] ~ 2 * K_borderline * mean_p_in_border * (1 - mean_p_in_border)^(N_prof-1)
```

This is hard to compute without knowing the distribution of p_i values.
The empirical measurement sidesteps this entirely.

### 4.2 Interpretation of Measured Disagreement

If |F_S| / P_total < 5%: most capsules have well-separated p_i from zero.
The borderline population is small, and the profiling protocol is reliable.

If |F_S| / P_total > 20%: a large fraction of capsules are borderline.
The profiling protocol is unreliable, and D->A transition counts may be
dominated by sampling noise.

---

## 5. Noise Attribution for D->A Transitions

### 5.1 Upper Bound on Noise-Driven D->A

Between checkpoints S_1 and S_2, the maximum number of noise-driven
D->A transitions is bounded by:

```
noise_DA(S_1, S_2) <= min(|F_{S_1}|, |F_{S_2}|)
```

Because a noise-driven D->A requires the capsule to be borderline at
the first checkpoint (could be classified either way).

### 5.2 Consensus-Based Noise Measurement

A more precise measure: compare transition counts from single-run
masks vs consensus masks. If noise inflates D->A transitions, then:

```
DA_single > DA_consensus
noise_fraction = (DA_single - DA_consensus) / DA_single
```

If DA_single <= DA_consensus, the noise contribution is zero or negative.
The "negative" case occurs because consensus reduces the dead set size
(fewer capsules classified as dead), which can increase the D->A
transition rate as a fraction of the dead population.

---

## 6. Experimental Design

### 6.1 Checkpoints

Same as Exp 18: S in {0, 50, 100, 200, 400, 800, 1600, 3200}

### 6.2 Dual Profiling Protocol

At each checkpoint, run profile_activations() twice:
- Run A: seed=1000 (20 batches x 32 = 640 samples)
- Run B: seed=2000 (20 batches x 32 = 640 samples)

The two runs use different random batches from the same validation
dataset but the same model weights. Any dead/alive disagreement is
definitionally noise.

### 6.3 Analysis

1. **Per-checkpoint disagreement**: |F_S| / P_total for each S
2. **Transition comparison**: D->A counts from single-run vs consensus
3. **Cohort analysis**: S=100 dead cohort tracking with both methods
4. **Jaccard comparison**: Dead-set similarity with both methods

### 6.4 Statistical Design

- 3 seeds (42, 123, 7) for training, matching Exp 18
- 2 profiling seeds (1000, 2000) at each checkpoint
- Report aggregate metrics across training seeds

---

## 7. Kill Criteria

| Criterion | Threshold | What it means |
|-----------|-----------|---------------|
| Same-checkpoint disagreement > 20% | >20% | Profiling protocol unreliable |
| Noise-attributable D->A > 50% total | >50% | Revival finding is artifactual |
| Noise-corrected revival rate < 5% | <5% | True revival too weak to matter |

---

## 8. Worked Numerical Example

At d=4, P=4, L=1, N_prof=10 (4 capsules, single layer, 10 profiling samples):

### Checkpoint S=100

```
Capsule 0: p_0 = 0.0    (truly dead, a_0^T x <= 0 for all x)
Capsule 1: p_1 = 0.002  (borderline: fires on 0.2% of inputs)
Capsule 2: p_2 = 0.3    (clearly alive)
Capsule 3: p_3 = 0.6    (clearly alive)
```

Run A (10 samples): capsule 1 fires on 0 samples.
  f_1^A = 0.0, classified DEAD. D_100^A = {0, 1}

Run B (10 samples): capsule 1 fires on 1 sample.
  f_1^B = 0.1, classified ALIVE. D_100^B = {0}

Disagreement: F_100 = {1}. |F_100| = 1, disagreement = 25%.
Consensus: D_100^C = {0} (only capsule 0 is dead in both runs).

### Checkpoint S=200

Both runs agree: capsule 0 still dead, capsule 1 fires (p_1 grew to 0.05).
D_200^A = D_200^B = {0}. Disagreement = 0.

### Transition S=100 -> S=200

Single-run A: capsule 1 goes from "dead" (false) to alive. D->A = 1.
Consensus: capsule 1 was alive at S=100 (in consensus). D->A = 0.

This illustrates how consensus eliminates the false D->A transition
for the borderline capsule. The genuine state change (p_1: 0.002 -> 0.05)
IS real, but it was already alive at S=100 -- just barely.

---

## 9. Assumptions

1. **Independence of profiling runs.** Runs A and B use non-overlapping
   random batches from the same validation dataset. The dataset is
   large enough that 640 samples represent <50% of available data,
   so the two runs are approximately independent.

2. **Profiling protocol matches Exp 18.** We use identical
   profile_activations() parameters (20 batches, 32 samples each).
   The only difference is the random seed for batch selection.

3. **Same training trajectories.** Training uses the same seeds as
   Exp 18. Only the profiling step differs (dual profiling vs single).

4. **Binary classification at f=0.** We use the same death threshold
   as all prior experiments. The borderline population (0 < f < 0.05)
   is measured but not used for reclassification.

5. **Consensus is conservative, not definitive.** The consensus mask
   (dead in both runs) has lower false-dead rate but higher false-alive
   rate. It provides a lower bound on the dead set, not the true dead set.

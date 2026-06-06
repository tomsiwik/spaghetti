# Training Duration vs Death Rate: Mathematical Foundations

## 1. Problem Statement

Experiment 10 (pruning controls) established that 54.3% of capsules in
single-domain ReLU models are dead after 200 steps of fine-tuning. This
number is the foundation of the pruning story: approximately half of all
ReLU neurons can be removed at zero cost.

But 200 steps is short. At macro scale, fine-tuning runs are 10K-100K+
steps. The critical open question:

**Q: Does the death rate at 200 steps predict the death rate at convergence?**

Three possible trajectories:
- **Monotonic increase**: Death accumulates with training. Longer training
  kills more neurons. The 54% is a lower bound.
- **Stable equilibrium**: Death stabilizes early. The 54% is representative
  of all training durations.
- **Recovery**: Early-dead neurons revive as the optimizer finds better
  solutions. The 54% is an upper bound that decreases with training.

---

## 2. Notation

All notation follows pruning_controls/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S         -- number of fine-tuning steps (varied: 50 to 3200)

a_i(S) in R^d  -- detector vector for capsule i after S fine-tuning steps
b_i(S) in R^d  -- expansion vector for capsule i after S fine-tuning steps

D = {x_1, ..., x_M}  -- profiling hidden states (from validation data)

f_i(S) = (1/M) * sum_{x in D} 1{a_i(S)^T x > 0}
       -- activation frequency of capsule i after S steps

delta(S) = (1/P) * sum_{i=1}^{P} 1{f_i(S) = 0}
         -- death rate after S steps (fraction of capsules that never fire)
```

---

## 3. Theory of ReLU Neuron Death During Training

### 3.1 Why Neurons Die

A ReLU neuron i is dead when a_i^T x <= 0 for all x in the data
distribution. This happens when:

1. **Initialization**: a_i starts in a half-space that contains no data.
   For symmetric initialization (e.g., Xavier/Kaiming), P(a_i^T x <= 0
   for all x) increases with the ratio ||a_i|| / spread(x).

2. **Gradient update**: A negative gradient step pushes a_i into a
   dead region. Once dead, grad_a f_i = 0 (ReLU gradient is zero for
   negative inputs), so the neuron cannot recover.

3. **Large learning rate**: Bigger steps increase the probability of
   crossing the hyperplane into the dead half-space.

### 3.2 The Irreversibility Argument

For a dead neuron i (f_i = 0):

```
d L / d a_i = sum_x (d L / d h_i) * 1{a_i^T x > 0} * x = 0
```

since 1{a_i^T x > 0} = 0 for all x when f_i = 0. The gradient is
exactly zero, so the optimizer cannot update a_i. The neuron stays
dead forever (under standard gradient descent).

**Exception**: Optimizer momentum. Adam/SGD with momentum accumulate
gradient history. If a_i received non-zero gradients before dying,
the momentum term can still push a_i. However, Adam's first moment
decays exponentially (beta_1 = 0.9 typically), so after ~10 steps
without gradients, momentum effectively vanishes.

### 3.3 Prediction: Monotonic Increase

Given irreversibility:

```
delta(S_2) >= delta(S_1)  for S_2 > S_1
```

Proof sketch: If capsule i is dead at step S_1 (f_i(S_1) = 0),
then a_i(S_1) = a_i(S_2) (no gradient updates) and f_i(S_2) = 0.
Dead neurons remain dead. New neurons can only die (transition
alive -> dead), not revive (dead -> alive). Therefore the death
rate is monotonically non-decreasing.

**Caveat**: This assumes zero gradient for dead neurons. In practice:
1. Adam momentum provides a non-zero update for ~10 steps after death
2. Weight decay (if used) can shrink dead neurons further or shift them
3. Batch normalization (if used) can shift the distribution

Our architecture uses RMSNorm (not BatchNorm) and Adam without weight
decay, so the monotonic prediction should hold.

### 3.4 Rate of Death Accumulation

At each step, the probability of killing a currently-alive neuron
depends on:

```
P(alive -> dead in one step) ~ lr * ||g_i|| / ||a_i|| * phi(margin_i)
```

where:
- lr is the learning rate
- g_i is the gradient for capsule i
- margin_i = min_x (a_i^T x) for x in data support (how close a_i is
  to the death boundary)
- phi is a function of the margin (smaller margin = easier to kill)

As training progresses:
- ||g_i|| decreases (loss converges, smaller gradients)
- margin_i may change (some neurons get pushed closer, others further)

This suggests the death rate curve is:
- **Steep initially** (large gradients, many marginal neurons)
- **Flattening over time** (smaller gradients, surviving neurons have
  larger margins)

Expected shape: **saturating exponential**

```
delta(S) ~ delta_inf * (1 - exp(-S / tau_death))
```

where delta_inf is the asymptotic death rate and tau_death is the
characteristic timescale.

---

## 4. Experimental Design

### 4.1 Step Sweep

Fine-tuning steps: S in {50, 100, 200, 400, 800, 1600, 3200}

Rationale:
- 50: Very early (half of Exp 10's measurement)
- 100: One third of Exp 10
- 200: Exp 10's measurement (should replicate 54.3%)
- 400: 2x Exp 10
- 800: 4x Exp 10 (approaching convergence at micro scale)
- 1600: 8x Exp 10 (likely well past convergence)
- 3200: 16x Exp 10 (extreme over-training)

The geometric spacing (each point is 2x the previous) uniformly samples
the log-time axis, which is natural for exponential/power-law processes.

### 4.2 Profiling Protocol

For each step count S:
1. Start from the SAME pretrained base model (300 steps on all data)
2. Fine-tune MLP weights only for S steps (attention frozen)
3. Profile activation frequencies on the validation set
4. Compute death rate delta(S)
5. Also record: val loss at S steps (to correlate death with quality)

### 4.3 Controls

- **At initialization (S=0)**: Profile the pretrained base model before
  any fine-tuning. This is the "natural" death rate from pretraining.
- **Exp 10 replication**: The S=200 point should match 54.3%.
- **Val loss convergence**: Track val loss to determine when the model
  has converged (diminishing returns from more training).

### 4.4 Per-Layer Analysis

From Exp 10 we know death is layer-dependent:
- Layer 0: ~0% dead (processes raw embeddings)
- Layers 1-3: 65-80% dead (attention-refined representations)

We track per-layer death rates at each step count to determine whether
the layer dependence emerges early or develops gradually.

---

## 5. Curve Fitting

### 5.1 Model: Saturating Exponential

```
delta(S) = delta_inf * (1 - exp(-S / tau)) + delta_0
```

Parameters:
- delta_0: death rate at initialization (before fine-tuning)
- delta_inf: additional death from fine-tuning (asymptotic contribution)
- tau: characteristic timescale (steps for ~63% of deaths to occur)

Total asymptotic death rate: delta_0 + delta_inf

### 5.2 Model: Power Law (Alternative)

```
delta(S) = alpha * S^beta + delta_0
```

If beta < 1, death accumulates sub-linearly (diminishing returns).
If beta > 1, death accelerates (unlikely given gradient shrinkage).

### 5.3 Fit Procedure

Least squares fit to the observed (S, delta(S)) pairs across all seeds.
Compare AIC/BIC for exponential vs power law model selection.

---

## 6. Kill Criteria

| Criterion | Threshold | What it means |
|-----------|-----------|---------------|
| Death decreases from 200 to 3200 steps | delta(3200) < delta(200) - 5pp | Early death is transient |
| Death at 3200 < 30% | delta(3200) < 30% | Pruning weakens at macro |
| Death unstable across seeds at any S | std(delta(S)) > 20pp | Measurement unreliable |

---

## 7. Worked Numerical Example

At d=4, P=4:

### S=0 (after pretraining, before fine-tuning)
```
Capsule 0: a_0 fires 60% -> alive
Capsule 1: a_1 fires 45% -> alive
Capsule 2: a_2 fires  0% -> DEAD (died during pretraining)
Capsule 3: a_3 fires 30% -> alive
delta(0) = 1/4 = 25%
```

### S=50 (early fine-tuning)
```
Capsule 0: fires 55% -> alive (slightly shifted)
Capsule 1: fires  0% -> DEAD (large gradient killed it)
Capsule 2: fires  0% -> DEAD (still dead, no gradient)
Capsule 3: fires 35% -> alive
delta(50) = 2/4 = 50%
```

### S=200 (standard fine-tuning)
```
Capsule 0: fires 50% -> alive
Capsule 1: fires  0% -> DEAD (still dead)
Capsule 2: fires  0% -> DEAD (still dead)
Capsule 3: fires  0% -> DEAD (died at step 180)
delta(200) = 3/4 = 75%
```

### S=3200 (over-training)
```
Capsule 0: fires 48% -> alive (stable)
Capsule 1: fires  0% -> DEAD
Capsule 2: fires  0% -> DEAD
Capsule 3: fires  0% -> DEAD
delta(3200) = 3/4 = 75%  (no new deaths, plateau reached)
```

This example shows: rapid early death (25% -> 50% in 50 steps),
continued death (50% -> 75% by step 200), then stabilization
(75% at step 3200).

---

## 8. Empirical Results vs Theory

### 8.1 Monotonicity Prediction: WRONG

The theoretical prediction (Section 3.3) that delta(S) is monotonically
non-decreasing was falsified. Empirically:

```
delta(0)    = 18.8%   (pretrained base)
delta(50)   = 55.1%   (rapid spike)
delta(100)  = 55.5%   (peak)
delta(200)  = 52.9%   (beginning of decay)
delta(3200) = 47.3%   (continued decay)
```

The aggregate death rate DECREASES after S=100 at all three seeds.

### 8.2 Why the Theory Failed

The irreversibility argument (Section 3.2) assumes dead neurons receive
exactly zero gradient forever. This is correct for an individual neuron
in isolation, but ignores inter-layer coupling:

In a multi-layer network where all MLP layers train simultaneously:

```
x_{l+1} = x_l + B_l @ ReLU(A_l @ norm(x_l))
```

When A_l and B_l change (through their OWN gradients from alive capsules),
the distribution of x_{l+1} shifts. This changes the input to layer l+1,
potentially pushing a_i^T x_{l+1} above zero for previously-dead capsule
i in layer l+1.

The corrected analysis:

```
d a_i / d step = 0                    (direct gradient, if dead)
d (a_i^T x) / d step != 0 in general (input distribution changes)
```

A dead neuron i in layer l can revive if:

```
exists x in D: a_i^T (x_l + delta_x_l) > 0
where delta_x_l arises from weight updates in layers 0..l-1
```

This does NOT require gradient flow through the dead neuron. It requires
earlier layers to shift the input distribution enough to move some data
points across neuron i's decision boundary.

### 8.3 Revised Trajectory Model

The data suggests a "spike and slow decay" model:

```
delta(S) = delta_spike * exp(-S/tau_rise) * (1 + decay_rate * log(1 + S/S_0))
         ~ rising exponential convolved with slow logarithmic decay
```

More practically, the data is well-described by:
- S < 50: steep rise (18.8% -> 55.1%)
- S = 50-400: plateau with fluctuation (52-56%)
- S > 400: slow decline (~2.5pp per 10x training)

The saturating exponential from Section 5.1 captures the rise well
(R^2 = 0.91) but systematically underpredicts the decline phase.

---

## 9. Assumptions

1. **Same profiling protocol as Exp 10.** We use profile_activations()
   with 20 batches of 32 samples on validation data. The profiling
   dataset is constant across all step counts.

2. **Same base model.** All step-count variants start from the SAME
   pretrained base model (300 steps on all data). The only variable
   is the number of fine-tuning steps.

3. **Frozen attention during fine-tuning.** Consistent with the Exp 10
   protocol. Only MLP (capsule pool A and B) weights are updated during
   fine-tuning.

4. **Death is irreversible.** FALSIFIED. Dead capsules CAN revive
   through inter-layer coupling: weight updates in earlier layers shift
   the input distribution to later layers, potentially pushing dead
   neurons' inputs above zero. This was the key theoretical prediction
   that the experiment disproved (see Section 8.2).

5. **Val loss converges before 3200 steps.** At micro scale (d=64, 200K
   params, ~25K training names), 3200 steps is ~10x the convergence point.
   This ensures we measure death in both converging and converged regimes.

6. **Training-domain data is sufficient for profiling.** We profile on
   the domain's own validation data. Cross-domain profiling is not needed
   for this experiment (Exp 10 already characterized cross-domain death).

# Activation Frequency Pruning: Mathematical Foundations

## 1. Problem Statement

Given the finding that mean gate-product magnitude is an ANTI-signal for pruning
at macro scale (8.9x worse than random), determine whether activation FREQUENCY
(firing rate) provides an independent and better pruning signal.

### 1.1 Notation

```
L         -- number of transformer layers (24 in Qwen2.5-0.5B)
d         -- embedding dimension (896)
d_ff      -- intermediate MLP dimension (4864)
N         -- total neurons = L * d_ff (116,736)

W_gate^l in R^{d_ff x d}  -- gate projection at layer l
W_up^l   in R^{d_ff x d}  -- up projection at layer l
W_down^l in R^{d x d_ff}  -- down projection at layer l

h_j^l(x) = SiLU(e_j^T W_gate^l x) * e_j^T W_up^l x   -- gate product

D = {x_1, ..., x_M}  -- calibration set (M = 16,384 positions)

mu_j^l     = (1/M) sum_m |h_j^l(x_m)|                  -- mean magnitude
f_j^l(eps) = (1/M) sum_m 1[|h_j^l(x_m)| > eps]         -- firing frequency
eps        -- firing threshold (hyperparameter)
```

### 1.2 Key Definitions

**Firing frequency** f_j^l(eps): the fraction of calibration positions where
neuron j at layer l produces a gate product with absolute value exceeding eps.

- f_j = 1.0 means the neuron fires on EVERY position (always-on)
- f_j = 0.0 means the neuron NEVER fires above eps (dead or near-dead)
- f_j ~ 0.01 means the neuron fires on ~1% of positions (specialist)

---

## 2. Why Frequency Differs from Magnitude

### 2.1 The Specialist Neuron Problem (from parent experiment)

A specialist neuron has:
- Low mean: mu_j = (1/M) sum |h_j(x_m)| is small
- High max: max_m |h_j(x_m)| >> mu_j
- Low frequency: f_j(eps) << 1 (fires rarely but strongly)

Parent experiment showed: pruning by ascending mu_j (lowest mean first) is 8.9x
WORSE than random, because low mu_j selects specialists.

### 2.2 The Always-On Neuron Hypothesis

An "always-on" neuron has:
- Moderate-to-high mean: mu_j is moderate
- Low max/mean ratio: max_j / mu_j is close to 1
- High frequency: f_j(eps) ~ 1.0 (fires on nearly all inputs)

**Hypothesis**: Always-on neurons encode redundant, context-independent
information. Their constant activation acts like a bias term. Removing them
distributes error uniformly across all inputs (graceful degradation) rather
than catastrophically on specific inputs.

### 2.3 When Frequency and Magnitude Diverge

Two neurons A and B with equal mean mu_A = mu_B = 0.05:

```
Neuron A (specialist):
  Activations: [0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0]  -- fires once strongly
  mu_A = 0.05, f_A(0.01) = 0.1

Neuron B (always-on):
  Activations: [0.05, 0.04, 0.06, 0.05, 0.05, ...]  -- fires on everything
  mu_B = 0.05, f_B(0.01) = 1.0
```

Mean magnitude cannot distinguish these. Frequency can.

The Spearman correlation between f_j and mu_j measures this divergence:
- rho ~ 1.0: frequency is monotonically related to magnitude (redundant)
- rho << 1.0: frequency provides independent information

**Kill criterion 2**: |rho| > 0.8 means frequency is redundant.

---

## 3. Pruning Protocol

### 3.1 Frequency-Based Neuron Ranking

For a chosen epsilon, rank all N = L * d_ff neurons by f_j^l(eps).

**High-first pruning** (our hypothesis): prune the neurons with highest
frequency first. These are the "always-on" neurons hypothesized to be redundant.

**Low-first pruning** (control): prune lowest-frequency neurons first.
These are specialists. Expected to be catastrophic (confirming parent finding).

### 3.2 Pruning Mechanism

Same as parent experiment: zero out rows in gate_proj and up_proj for pruned
neurons. This is zero-shot structured pruning (no recovery training).

For neuron j at layer l:
```
W_gate^l[j, :] = 0
W_up^l[j, :]   = 0
```

This completely silences the neuron's contribution to the residual stream.

### 3.3 Evaluation

Perplexity on WikiText-2 validation split (held-out from calibration).
Compare against:
- Baseline (no pruning): ppl_0
- Random pruning (3 seeds): ppl_rand
- Mean magnitude pruning (parent signal): ppl_mag
- Frequency pruning high-first: ppl_freq_high
- Frequency pruning low-first: ppl_freq_low

### 3.4 Kill Criterion 1 Formalization

At 5% pruning fraction (5,837 neurons):

```
delta_freq = ppl_freq_high - ppl_0
delta_rand = ppl_rand - ppl_0

ratio = delta_rand / delta_freq
```

PASS if ratio > 2.0 (frequency causes less than half the damage of random).
KILL if ratio <= 2.0.

---

## 4. Choice of Epsilon

The firing threshold eps is a hyperparameter. Its choice affects what "firing" means:

- eps too low (e.g., 0.0001): nearly all neurons fire on all positions, f_j ~ 1.0
  for all j, no discrimination
- eps too high (e.g., 1.0): nearly no neurons fire, f_j ~ 0.0, no discrimination

Optimal eps should maximize the variance of f_j across neurons, creating a clear
separation between always-on and specialist neurons.

From parent experiment, the gate product distribution has:
- Median: 0.078
- P10: 0.044
- P90: 0.159

We test eps in {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}. The range 0.01-0.05
should provide the most informative frequency signal (below the median, above noise).

---

## 5. Computational Cost

Profiling frequency requires one forward pass through the model on the calibration
set, identical to profiling mean magnitude. The only addition is counting firings
per epsilon (a binary comparison per position per neuron), which is negligible.

```
FLOPs(profiling) = M * (standard forward pass FLOPs)
Memory(profiling) = O(L * d_ff * |epsilons|) for frequency counters
                  = 24 * 4864 * 7 = ~816K floats (negligible)
```

---

## 6. Assumptions

| # | Assumption | Justification | Kill if violated |
|---|-----------|---------------|-----------------|
| 1 | Always-on neurons are redundant | They encode context-independent information (like a bias) | Pruning them catastrophic (KC1 fails) |
| 2 | Frequency is not redundant with magnitude | Different statistics of the activation distribution | KC2: Spearman |rho| > 0.8 |
| 3 | eps=0.01 is informative | Below median (0.078), above noise | Test multiple eps values |
| 4 | Specialist neurons fire rarely | Low frequency + high max/mean ratio | Distribution analysis |
| 5 | Zero-shot pruning is feasible for some signal | At least one signal beats random | All signals fail -> pruning method is wrong |

---

## 7. Worked Example (Micro Scale)

With d_ff=8, M=10 positions, eps=0.01:

```
Neuron 0: activations = [0.002, 0.001, 0.003, 0.002, 0.001, 0.002, 0.001, 0.003, 0.002, 0.001]
  mu_0 = 0.0018, f_0(0.01) = 0.0 (never fires)
  -> dead/near-dead neuron

Neuron 1: activations = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  mu_1 = 0.05, f_1(0.01) = 0.1 (specialist -- fires on 1/10 positions)
  -> pruning this catastrophically hurts the 1 input it serves

Neuron 2: activations = [0.05, 0.04, 0.06, 0.05, 0.05, 0.04, 0.06, 0.05, 0.05, 0.04]
  mu_2 = 0.049, f_2(0.01) = 1.0 (always-on)
  -> pruning this hurts all inputs equally but mildly (0.049 per position)

Neuron 3: activations = [0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.7, 0.8, 0.7]
  mu_3 = 0.78, f_3(0.01) = 1.0 (always-on, high magnitude)
  -> pruning this is catastrophic regardless of frequency

Ranking by frequency (high-first): [2, 3, 1, 0]
- Neuron 2 pruned first: moderate, uniform damage
- Neuron 3 pruned second: high damage (despite high frequency)

Key insight: frequency alone is not sufficient. High-frequency + low-magnitude
neurons are the safest targets. But as a first-order signal, high-frequency
may still beat low-magnitude (which selects specialists).
```

# Gumbel-Sigmoid Routing Ablation: Mathematical Foundations

## Router Architecture

A 2-layer MLP maps mean-pooled hidden states to per-adapter gate logits:

Given input h in R^d (mean-pooled transformer hidden state):
- z = GELU(W_1 h + b_1), where W_1 in R^{h x d}, b_1 in R^h (h = hidden_dim)
- l = W_2 z + b_2, where W_2 in R^{N x h}, b_2 in R^N (N = number of adapters)

## Gate Mechanisms

### Non-competing (Sigmoid / Gumbel-Sigmoid)

Each adapter has an independent Bernoulli gate:

g_i = sigma((l_i + G_i) / tau)

where G_i = -log(-log(U_i)), U_i ~ Uniform(0,1) (Gumbel noise), tau > 0 is temperature.

At inference: g_i = 1[l_i > 0] (hard threshold on raw logits).

Properties:
- Gates are independent: activating expert i does not suppress expert j
- Multiple experts can be simultaneously active with high probability
- Binary cross-entropy loss per gate: L = -sum_i [t_i log(g_i) + (1-t_i) log(1-g_i)]

### Competing (Softmax / Gumbel-Softmax)

Gates are normalized to sum to 1:

g = softmax((l + G) / tau)

Properties:
- Zero-sum: activating expert i necessarily suppresses others
- Natural probability distribution over experts
- Cross-entropy loss: L = -log(g_{target})

## Temperature Parameter

Controls sharpness of the gate distribution.

For sigmoid: sigma(x/tau) approaches step function as tau -> 0, uniform 0.5 as tau -> inf.
For softmax: softmax(x/tau) approaches argmax as tau -> 0, uniform 1/N as tau -> inf.

### Annealing Schedule

Linear interpolation: tau(t) = tau_start + (tau_end - tau_start) * (t / T)

High initial tau (exploration) -> low final tau (exploitation).

## Gate Activation Penalty (L1 Regularization)

An auxiliary L1 penalty on total gate activation prevents expert collapse:

L_aux = alpha * sum_i mean_batch(g_i)

where:
- g_i in [0,1] is the sigmoid gate activation for expert i
- mean_batch averages over the training batch
- sum_i sums over all N experts
- alpha controls regularization strength

This is NOT the Switch Transformer load-balancing loss (which uses f_i * p_i products).
It is simpler: an L1 penalty on total gate mass.

**Why it works**: The primary BCE loss pushes the target gate UP toward 1.
The L1 penalty pushes ALL gates DOWN toward 0. The net effect is:
- Target gate: BCE push-up wins (strong gradient from cross-entropy)
- Non-target gates: L1 push-down wins (no opposing BCE gradient)
- Result: sharper routing with suppressed off-target activations

This prevents expert collapse where a few dominant experts absorb routing
probability from similar-looking domains (e.g., chemistry vs science_qa).

At alpha = 0: no regularization, collapse-prone domains get 0% accuracy.
At alpha ~ 0.1: effective suppression without harming primary routing.
At alpha >= 0.5: over-regularization destroys routing (52% top-2 at alpha=0.5).

## Straight-Through Gumbel Estimator

Forward pass uses hard (discrete) gates, backward pass uses soft (continuous) gradients:

g_hard = 1[g_soft > 0.5]
g_ST = g_soft + stop_gradient(g_hard - g_soft)

This preserves gradient flow while making forward-pass decisions discrete.

## Why Zero-Accuracy Domains Occur

### Hidden State Confusion

For domain i with centroid c_i = E[h | domain=i], routing fails when:

cos(c_i, c_j) approx 1 for some j != i

Measured confusions:
- chemistry-science_qa: cos = 0.992
- wikitext-history: cos = 0.996
- debate-legal: cos = 0.961

### Intra-Domain Variance

Domain i is hard to route when Var[h | domain=i] is high relative to
the inter-domain distance. Measured:
- dialogue: variance = 4.375 (13x higher than typical)
- Low-variance domains (reasoning: 0.332, cooking: 0.350) route perfectly

### The Asymmetry

More training (3000 -> 6000 steps) fixes most collapse cases (chemistry 0% -> 80%,
debate 0% -> 70%). L1 gate regularization additionally recovers wikitext (0% -> 40%)
and pushes chemistry to 100%. But high-variance domains (dialogue) remain unroutable
because the centroid is a poor representative -- no linear classifier on mean-pooled
states can separate dialogue from its heterogeneous overlap with many domains.

## Parameter Counts

| Component | Parameters |
|-----------|-----------|
| Router (h=256) | d*256 + 256 + 256*N + N = 2560*256 + 256 + 256*49 + 49 = 668,977 |
| Router (h=128) | d*128 + 128 + 128*N + N = 2560*128 + 128 + 128*49 + 49 = 334,513 |
| Per adapter (rank-16) | ~1,900 (ternary) |
| Total system (49 adapters + router) | ~762K |

## Worked Example

d=2560, N=49, h=256, tau=2.0->0.5, 3000 steps:

Step 0: Sample "chemistry" text, extract h in R^2560, tau=2.0
- l = W_2 * GELU(W_1 * h) in R^49 (raw logits)
- G ~ Gumbel(0,1)^49
- g = sigma((l + G) / 2.0) in [0,1]^49 (soft, high temp = exploration)
- Target: t = [0,...,0,1,0,...,0] (1 at chemistry index 8)
- Loss: BCE over 49 gates

Step 2999: tau=0.5
- g = sigma((l + G) / 0.5) (sharper, near-binary)
- Router has learned to output high l_8 for chemistry-like inputs

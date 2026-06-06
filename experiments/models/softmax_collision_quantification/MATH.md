# Softmax Collision Quantification: Mathematical Foundations

## Variables and Notation

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| N | scalar | Number of expert groups |
| k | scalar | Top-k experts selected per token |
| d | scalar | Embedding dimension |
| x | (B, T, d) | Input token representations |
| W_r | (d, N) | Router weight matrix |
| s | (B, T, N) | Raw router logits: s = x W_r |
| p | (B, T, N) | Softmax probabilities: p_i = exp(s_i/T) / sum_j exp(s_j/T) |
| T | scalar | Temperature parameter (T > 0) |
| epsilon | scalar | Collision threshold |
| g | (B, T) | Top-1 vs top-2 gap: g = p_(1) - p_(2) |
| C(epsilon) | scalar | Collision rate: fraction of tokens with g < epsilon |
| m | scalar | Target margin for margin loss |
| L_margin | scalar | Margin loss: mean(max(0, m - g)) |

## Definition of Collision

A routing **collision** occurs when the softmax router assigns near-equal
probabilities to its top-2 expert choices. Formally, for a token with sorted
probabilities p_(1) >= p_(2) >= ... >= p_(N):

    collision(x, epsilon) = 1[p_(1)(x) - p_(2)(x) < epsilon]

The collision rate over a dataset D is:

    C(epsilon) = (1 / |D|) * sum_{x in D} collision(x, epsilon)

### Why Collisions Matter

When g < epsilon, the router is effectively random between two experts.
This has two consequences:
1. **Gradient instability**: Small perturbations flip the expert assignment,
   creating noisy gradients for calibration.
2. **Wasted compute**: Both selected experts produce nearly identical
   contributions (weighted by near-equal probabilities).

## Empirical Scaling Law

We observe that collision rate follows a power law in N:

    C(epsilon) = a * N^b

The parameters (a, b) depend on epsilon and are fit via log-log linear
regression on N = {8, 16, 32, 64}:

| Epsilon | a | b | r^2 |
|---------|------|-------|------|
| 0.01 | 0.064 | 0.614 | 0.959 |
| 0.05 | 0.471 | 0.139 | 0.935 |
| 0.10 | 0.634 | 0.073 | 0.999 |

**Caveat**: This fit uses 4 data points for 2 parameters. The r^2 values
are high mechanically but the confidence intervals on the exponent b are
wide. The exponent b ~ 0.6 at tight epsilon should be treated as an
approximate characterization, not a precise prediction.

### Why Collision Rate Grows with N (Qualitative)

Two effects compound:

1. **Probability compression**: With N experts, the maximum softmax
   probability for a uniform distribution is 1/N. As N grows, the
   probability mass per expert shrinks, compressing the gap between
   any two experts.

2. **Order statistics spacing**: The gap between the largest and
   second-largest of N i.i.d. random variables shrinks as N grows.
   For Gaussian logits, the expected gap between consecutive order
   statistics near the maximum scales as O(1/log(N)).

These effects are well-documented in the softmax bottleneck literature
(Yang et al. 2018) and the MoE routing instability literature (Fedus
et al. 2021, ReMoE 2024). We contribute an empirical quantification
of the scaling rate, not the directional observation itself.

## Temperature Scaling

Replace p = softmax(s) with p = softmax(s / T):

    p_i = exp(s_i / T) / sum_j exp(s_j / T)

**Effect on gap**: For T < 1 (sharpening), the probability gap between
any two experts with logit difference g grows as approximately
exp(g/T) / exp(g), which is superlinear in 1/T. This resolves collisions
by amplifying small logit differences into large probability differences.

**Dual-temperature decomposition**: When measuring collision rates, we
can separate two effects:
- **Training effect**: T < 1 during training produces models with
  inherently sharper logit distributions (measured at T=1.0)
- **Inference effect**: Applying T < 1 at measurement time further
  sharpens any logit distribution

Our experiment (Section: Empirical Results) shows these contribute
roughly 1:2 for T=0.5 (training:inference = 0.117:0.227 collision
rate reduction).

## Margin-Based Auxiliary Loss

Add a hinge loss encouraging the top-1/top-2 gap to exceed target margin m:

    L_margin = (1 / BT) * sum_{b,t} max(0, m - g_{b,t})

where g_{b,t} = p_(1)_{b,t} - p_(2)_{b,t}.

**Gradient**: For tokens where g < m (collision zone):

    dL_margin / ds_i = (1/BT) * (dp_(1)/ds_i - dp_(2)/ds_i)

This pushes the router to increase the score of the current winner and
decrease the runner-up score, directly resolving the collision.

## Computational Cost Analysis

| Method | Extra Params | Extra FLOPs/token | Extra Memory |
|--------|-------------|-------------------|--------------|
| Baseline softmax | 0 | 0 | 0 |
| Temperature | 0 | N (division) | 0 |
| Margin loss | 0 | 2N (sort + hinge) | O(BT) |
| Cuckoo dual-hash | dN | 2dN | 2x router |

Temperature scaling is essentially free. Margin loss is O(N log N) for sorting
but negligible compared to expert compute O(d^2).

## Worked Example (Micro Scale)

d = 64, N = 32, k = 2, T = 1.0

Measured collision rate C(0.01) at N=32: 0.606 (61% of tokens have
top-1 vs top-2 gap below 0.01).

With T=0.5 at inference: C(0.01) drops to 0.245 (59% reduction).
Decomposed: 0.117 from learned logit sharpening + 0.227 from
inference-time temperature.

Extrapolation using empirical fit: at N=128, C(0.01) ~ 0.96.
At N=256+, effectively all tokens collide at this threshold.

## Assumptions

1. **Independence**: Router logits for different tokens are independent.
   Violated in practice (sequential correlation), but holds approximately
   across batches.

2. **Post-training measurement**: Collision rates are measured after
   500 training steps. After training, logits develop multi-modal
   structure with peaks at preferred experts. The scaling law
   characterizes this trained regime, not random initialization.

3. **Top-k stability**: We assume the collision only matters for the top-2
   (selection boundary). Collisions deeper in the ranking (3rd vs 4th) are
   irrelevant for top-2 routing.

4. **Fixed epsilon**: The "meaningful" epsilon may change with N because
   the probability scale changes. We sweep multiple epsilon values to
   address this.

5. **N-capsules confound**: N and capsules_per_group co-vary in Phase 1
   (total capsules held at ~256). The collision scaling may partly reflect
   reduced per-group expressivity at high N, not purely softmax compression.
   See Limitations in PAPER.md.

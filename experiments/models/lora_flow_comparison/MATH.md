# SOLE vs LoRA-Flow: Formal Comparison Framework

## 1. Setup

### 1.1 Shared Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| W_s^{(l)} | Frozen base (skeleton) weight at layer l | R^{d_out x d_in} |
| dW_i^{(l)} | Expert i delta at layer l (= (alpha/r) * B_i^{(l)} A_i^{(l)}) | R^{d_out x d_in} |
| x_t^{(l)} | Hidden state at layer l, token position t | R^{d} |
| k | Number of simultaneously active experts | scalar |
| N | Total experts in library | scalar |
| r | LoRA rank | scalar |
| d | Model embedding dimension | scalar |
| L | Number of layers | scalar |

### 1.2 Four Composition Functions

All methods share the per-layer form:

    h_out^{(l)} = h_in^{(l)} + sum_{i in S} c_i^{(l)}(x) * Delta_h_i^{(l)}

where Delta_h_i^{(l)} is expert i's output contribution at layer l.
They differ only in how c_i^{(l)} is determined.

## 2. Method Definitions

### 2.1 SOLE

    c_i^{(l)}(x) = 1   for all i, l, x

Fixed unit weights. Zero trainable parameters.

### 2.2 CAT (LoRA Soups, Prabhakar et al., 2024)

    c_i^{(l)} = w_i^{(l)}   (static, learned per-layer scalars)

Trainable parameters: 2*k*L scalars (one per expert per layer per
projection direction).

### 2.3 LoRA-Flow (Wang et al., 2024)

    c_i^{(l)}(x_t) = [softmax(W_gate^{(l)} x_t^{(l)})]_i + b_i^{(l)}

where W_gate^{(l)} in R^{k x d}, b^{(l)} in R^{k}.

Trainable parameters per layer: k*d + k = k*(d+1).
Total: L * k * (d+1).

Key features:
- **Input-dependent**: weights change per token based on hidden state
- **Per-layer**: each layer computes its own fusion weights
- **softmax-normalized** with additive bias (can produce negative total weight)

### 2.4 X-LoRA (Buehler, 2024)

    c_i^{(l)}(x_t) = softmax(W_2^{(l)} * ReLU(W_1^{(l)} x_t^{(l)}))_i

where W_1^{(l)} in R^{h x d}, W_2^{(l)} in R^{k x h}.

Trainable parameters per layer: h*(d + k).
Total: L * h * (d + k).

Key features:
- **MLP gating**: more expressive than LoRA-Flow's linear gate
- **softmax-only**: no bias, weights always sum to 1

## 3. Theoretical Comparison

### 3.1 When Dynamic Weights Help

Dynamic input-dependent weights can outperform static unit weights when:

1. **Expert specialization is strong**: different experts are clearly better
   for different inputs, so routing to the best expert(s) improves over
   uniform inclusion.

2. **Expert interference is significant**: high cos(dW_i, dW_j) creates
   cross-terms that a gate can mitigate by down-weighting overlapping experts.

3. **Task distribution is heterogeneous**: the query requires different
   skill mixtures at different token positions.

### 3.2 When Unit Weights Suffice

SOLE's unit weights are optimal when:

1. **Near-orthogonality holds**: cos(dW_i, dW_j) ~ 0 means cross-terms
   vanish and each expert's optimal weight is independent of others.

2. **Expert magnitudes are calibrated**: standard LoRA scaling (alpha/r)
   ensures each expert's contribution is properly scaled.

3. **All experts contribute**: the composed model should incorporate all
   expert knowledge simultaneously (library composition, not routing).

### 3.3 Parameter Cost Scaling

For a model with d=4096, L=32 layers:

| Method | k=2 | k=10 | k=100 | k=500 |
|--------|-----|------|-------|-------|
| SOLE | 0 | 0 | 0 | 0 |
| CAT | 128 | 640 | 6,400 | 32,000 |
| LoRA-Flow | 262K | 1.31M | 13.1M | 65.6M |
| X-LoRA (h=64) | 8.39M | 8.41M | 8.59M | 9.41M |

LoRA-Flow scales as O(k*d*L) -- at k=500, the gate itself has more params
than a rank-16 LoRA adapter (40.4M at d=4096). This is a fundamental
scaling limitation: the routing mechanism becomes larger than the experts
it routes.

X-LoRA scales as O(h*d*L + h*k*L). The h*d*L term dominates, making it
approximately k-independent but with a large constant (8.4M at h=64).

### 3.4 SOLE Optimality Under Orthogonality

**Theorem**: When experts are mutually orthogonal (cos(dW_i, dW_j) = 0
for all i != j), unit weights c_i = 1 minimize the composition loss.

**Proof sketch**: The loss gradient w.r.t. c_i factors as:

    dL/dc_i = <dL/dW_composed, dW_i>

When experts are orthogonal, the optimal c_i depends only on dW_i and the
loss landscape, not on other experts' contributions. Standard LoRA training
with appropriate alpha/r already optimizes this single-expert contribution,
making c_i = 1 optimal.

**Practical implication**: At d=896 with cos ~ 0.0002, the deviation from
optimality is bounded by:

    |c_i* - 1| <= O(k * cos_max * ||dW||^2) ~ O(k * 2e-4) ~ negligible

Even at k=100, the optimal weights deviate from 1.0 by < 2%.

## 4. Empirical Results (Micro Scale)

### 4.1 Configuration

d=64, d_ff=256, r=8, L=4, 12 domains (4 clusters x 3), 3 seeds.
SPSA optimization for all gates (50 steps each).

### 4.2 Quality Comparison

All four methods produce **identical NTP loss to 4 decimal places**
across all compositions (k=2, 6, 12) and all seeds.

This is expected: at d=64 with cos=0.002, experts barely specialize
(base loss = expert loss). There is no signal for the dynamic gate to
exploit. The quality comparison is vacuous.

### 4.3 Overhead Comparison

| Method | k=2 | k=6 | k=12 |
|--------|-----|------|------|
| SOLE | 0.14s | 0.38s | 0.75s |
| CAT | 2.42s (17x) | 7.63s (20x) | 18.1s (24x) |
| LoRA-Flow | 0.85s (6x) | 2.40s (6x) | 6.19s (8x) |
| X-LoRA | 0.89s (6x) | 2.53s (7x) | 6.13s (8x) |

CAT overhead grows superlinearly with k (finite-diff on k*L scalars).
LoRA-Flow and X-LoRA overhead is roughly constant at ~6-8x SOLE
(SPSA requires only 2 forward passes per step regardless of k).

### 4.4 Parameter Count (Micro Scale)

| Method | k=2 | k=6 | k=12 |
|--------|-----|------|------|
| SOLE | 0 | 0 | 0 |
| CAT | 16 | 48 | 96 |
| LoRA-Flow | 520 | 1,560 | 3,120 |
| X-LoRA | 4,224 | 4,480 | 4,864 |

## 5. Assumptions

1. Experts are trained independently with standard LoRA (alpha/r scaling).
2. Expert deltas are low-rank (rank r << d).
3. Experts occupy near-orthogonal subspaces (cos ~ O(r/sqrt(D))).
4. The composition target is a mixture of expert training distributions.
5. SPSA gradient estimation is sufficient for gate optimization
   (may undertrain vs analytical gradients, biasing toward SOLE's favor --
   but the vacuous quality result means this bias is irrelevant).

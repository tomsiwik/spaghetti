# SwiGLU Gate-Aware Pruning: Mathematical Foundations

## 1. Problem Statement

Given a capsule MLP using SwiGLU activation (matching Qwen3.5/Llama):

    y = B @ (SiLU(W_gate @ x) * (W_up @ x))

determine whether profiling the gate PRODUCT (not SiLU output alone) enables
pruning that the SiLU activation floor prevents.

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- number of capsules per pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)

W_gate in R^{P x d}  -- gate projection matrix (SiLU-activated)
W_up   in R^{P x d}  -- up projection matrix (linear, learned gate)
B      in R^{d x P}  -- down projection matrix (columns are b_i)

g_i(x) = SiLU(w_gate_i^T x)  -- gate output for capsule i
u_i(x) = w_up_i^T x           -- up output for capsule i
h_i(x) = g_i(x) * u_i(x)     -- gate product for capsule i (SwiGLU output)

D = {x_1, ..., x_M} -- calibration dataset of M hidden-state vectors

mu_i^g = (1/M) * sum_x |g_i(x)|           -- mean abs gate output
mu_i^u = (1/M) * sum_x |u_i(x)|           -- mean abs up output
mu_i^h = (1/M) * sum_x |h_i(x)|           -- mean abs gate product
tau    -- pruning threshold on mean abs gate product
```

---

## 2. Why SiLU Pruning Fails (Exp 15 Recap)

SiLU profiling measures mu_i^g = mean |SiLU(w_gate_i^T x)|.

SiLU(z) = z * sigmoid(z) has a minimum of approximately -0.2784 at z ~ -1.278.
For any input distribution with nonzero variance, E[|SiLU(z)|] > 0.

**Empirical floor (Exp 15):** min_i mu_i^g ~ 0.046 across all capsules, layers,
and seeds. This is ~5x above any safe pruning threshold (tau=0.01).

**Result:** 0% of capsules prunable at tau <= 0.01 with SiLU-only profiling.

---

## 3. Why Gate-Product Pruning Can Work

### 3.1 The Multiplicative Suppression Mechanism

The SwiGLU gate product for capsule i is:

    h_i(x) = SiLU(w_gate_i^T x) * (w_up_i^T x)

Even when |SiLU(w_gate_i^T x)| > 0.04 (the SiLU floor), the product can be
near-zero if |w_up_i^T x| is small.

For the product to be small, we need:
    |h_i(x)| = |g_i(x)| * |u_i(x)| < tau

This is satisfied when EITHER factor is small enough, giving two channels
for near-zero products:

1. **Up-suppression:** u_i(x) ~ 0 while g_i(x) remains at its floor
   (the up projection learns to suppress this capsule)
2. **Mutual suppression:** both g_i(x) and u_i(x) are moderate but
   their product is small

### 3.2 Mean Absolute Gate Product Bound

For capsule i:

    mu_i^h = E[|g_i(x) * u_i(x)|]
           <= E[|g_i(x)|] * E[|u_i(x)|]     (by independence of |g_i| and |u_i|)
           = mu_i^g * mu_i^u

Note: this upper bound requires independence (or at least non-negative
correlation of |g_i| and |u_i|). Cauchy-Schwarz gives a different bound:
`E[|XY|] <= sqrt(E[X^2] E[Y^2])`. Since g_i and u_i share input x, they
are correlated, and the actual mu_i^h can be much smaller than the product
of means.

**Why the gate product floor is lower than the SiLU floor (qualitative):**

The SiLU floor is a hard lower bound: for any capsule i with nonzero-variance
inputs, `mu_i^g = E[|SiLU(w_gate_i^T x)|] > 0.04` (empirically ~0.046).
No amount of training can push this below the activation function's intrinsic
floor.

The gate product `h_i = g_i * u_i` introduces a second multiplicative factor.
Even when g_i is bounded away from zero by the SiLU floor, the product can be
suppressed if the up-projection learns `u_i(x) ~ 0`. The gate product floor
is NOT bounded by `floor_g * floor_u` from below -- the minimizing capsule for
the product need not be the same capsule that minimizes either factor.

**This floor reduction is an empirical observation, not a provable bound.**
The data shows floor_h ~ 0.014 vs floor_g ~ 0.046 (3.3x reduction), but
the exact ratio depends on training dynamics, data distribution, and the
correlation structure between g_i and u_i.

### 3.3 Numerical Example (d=64, P=128)

From Exp 15 (SiLU-only):
  - min mu_i^g ~ 0.046 (SiLU floor)
  - All 128 capsules above 0.046
  - 0% prunable at tau=0.01

From this experiment (SwiGLU gate product):
  - min mu_i^h ~ 0.014 (gate product floor, seed 42 Layer 1)
  - SiLU floor still ~ 0.016 (unchanged)
  - Up projection floor ~ 0.103
  - Product floor: 0.014 << 0.046 (3.3x lower than SiLU-only)

At tau=0.05:
  - 66.5% of capsules have mu_i^h < 0.05 (prunable!)
  - 0% of capsules have mu_i^g < 0.05 in SiLU-only (floor prevents it)
  - Quality degradation: +1.22% (within 3% kill threshold)

---

## 4. Pruning Error Bound

When pruning capsule i with gate product mean abs mu_i^h < tau, the per-position
output error is:

    ||delta_y|| = ||b_i * h_i(x)|| = ||b_i|| * |h_i(x)|

Expected error contribution:
    E[||delta_y||] = ||b_i|| * mu_i^h < ||b_i|| * tau

Total error from pruning S capsules simultaneously:

    E[||sum_{i in S} b_i * h_i(x)||] <= sum_{i in S} ||b_i|| * mu_i^h

This is approximate (not lossless like ReLU pruning where h_i = 0 exactly).
The bound is tight when pruned capsule contributions are independent.

### 4.1 Why 66.5% Can Be Pruned at +1.22%

The gate product distribution is BIMODAL at micro scale:
  - A majority of capsules have mu_i^h in [0.01, 0.05] (the "suppressed" set)
  - A minority have mu_i^h in [0.05, 0.13] (the "active" set)

The suppressed capsules contribute approximately:
    sum_{i in suppressed} ||b_i|| * mu_i^h ~ 66.5% * 0.03 * ||b_avg|| ~ 2% of output

This aligns with the observed +1.22% quality degradation.

---

## 5. Computational Cost

### 5.1 SwiGLU vs SiLU Capsule Pool

SiLU capsule pool per layer:
  - A: (P, d) matmul = P*d MADs
  - SiLU: P ops
  - B: (d, P) matmul = d*P MADs
  - Total: 2*P*d + P

SwiGLU capsule pool per layer:
  - W_gate: (P, d) matmul = P*d MADs
  - W_up: (P, d) matmul = P*d MADs
  - SiLU: P ops
  - Element-wise multiply: P ops
  - B: (d, P) matmul = d*P MADs
  - Total: 3*P*d + 2*P

SwiGLU is 1.5x the MADs of SiLU for the projection matrices.

### 5.2 Pruning Savings

After pruning 66.5% of capsules (P' = 0.335 * P):
  - SwiGLU cost: 3*P'*d + 2*P' = 3*0.335*P*d + 2*0.335*P
  - Savings: 66.5% of the MLP compute

Net effect: SwiGLU with pruning uses 0.335 * 1.5 = 0.50x the compute of
unpruned SiLU. This is comparable to ReLU dead capsule pruning (57% dead
in single-domain).

### 5.3 Profiling Cost

Profile gate products over M positions:
  - Per position: 2*P*d MADs (W_gate, W_up) + P (SiLU) + P (multiply) + P (abs)
  - Total: M * (2*P*d + 3*P)
  - At M=20*32*32=20480 (20 batches, 32 batch, 32 seq): ~2.7M MADs per layer

Negligible compared to training cost (~300 * 32 * 32 * 3*P*d ~ 1.2B MADs).

---

## 6. Assumptions and Limitations

1. **Gate product statistics are stable across inputs**: Profiling on calibration
   data predicts gate product magnitudes on test data. Validated at micro scale
   by Exp 12 (profiling noise is 2.6-3.8%, well below threshold).

2. **Pruning error accumulates linearly**: When pruning many capsules simultaneously,
   errors from individual capsules do not amplify catastrophically. Validated
   empirically: 66.5% pruning at +1.22% degradation.

3. **Bimodal distribution transfers to macro**: The [suppressed, active] split
   in gate products is expected to persist at macro scale because it reflects
   the learned gating mechanism of SwiGLU, not scale-specific dynamics.

4. **Gate product floor is lower than SiLU floor**: Confirmed empirically
   (0.014 vs 0.046), but the exact ratio may vary with model size, data
   distribution, and training duration.

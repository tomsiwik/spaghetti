# Entropy-Adaptive Routing: Mathematical Foundations (Revised)

## Variables and Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| x | (B, T, d) | Input hidden states, batch B, sequence T, dim d |
| G | scalar | Number of expert groups |
| W_r | (G, d) | Router weight matrix |
| s | (B, T, G) | Raw routing scores, s = x @ W_r^T |
| p | (B, T, G) | Routing probabilities, p = softmax(s) |
| H | (B, T) | Per-token routing entropy |
| H_max | scalar | Maximum entropy = log(G) |
| tau | scalar | Entropy threshold (learned or fixed) |
| alpha | (B, T) | Soft k-selection variable, in [0, 1] |
| k(t) | (B, T) | Per-token number of active experts |
| lambda | scalar | Sparsity coefficient for compute-efficiency loss |

## Core Mechanism

### Step 1: Routing Distribution

Standard softmax routing:

    s_g = x^T w_g,  for g = 1, ..., G
    p_g = exp(s_g) / sum_j exp(s_j)

### Step 2: Per-Token Entropy

Shannon entropy of the routing distribution:

    H(p) = -sum_{g=1}^{G} p_g log(p_g)

Properties:
- H = 0 when one p_g = 1 (perfect confidence, one expert dominates)
- H = log(G) when p_g = 1/G for all g (maximum uncertainty)
- H is differentiable with respect to routing scores s

### Step 3: Adaptive k Selection

Soft threshold using sigmoid for differentiability:

    alpha = sigmoid((H - tau) / T_temp)

where T_temp is a temperature parameter controlling sharpness (we use T_temp = 0.1).

- alpha near 0: high confidence, use k=1
- alpha near 1: low confidence, use k=2

The hard threshold equivalent:

    k(t) = 1  if H(p_t) < tau
    k(t) = 2  if H(p_t) >= tau

**Revision note (Fix 5):** The soft-to-hard gap is negligible (<0.1% quality
delta across 3 seeds). At T_temp=0.1, the sigmoid is sharp enough that soft
and hard produce functionally identical outputs.

### Step 4: Soft Mask Interpolation

Let M_1 be the top-1 mask and M_2 be the top-2 mask:

    M_1[g] = 1 if g = argmax(s), else 0
    M_2[g] = 1 if s_g >= s_{(2)}, else 0

where s_{(2)} is the 2nd-largest score.

The interpolated mask:

    M = (1 - alpha) * M_1 + alpha * M_2

Renormalized weights:

    w_g = p_g * M_g / sum_j (p_j * M_j)

Output:

    y = sum_{g=1}^{G} w_g * Expert_g(x)

**Critical observation:** Even with alpha near 0, w_g is nonzero for all groups
because the softmax probabilities p_g are always positive and the mask
interpolation never produces exact zeros. At G=8, the minimum weight for any
expert is bounded below by p_min * M_1_min > 0. This means conditional expert
execution (skipping experts with w < epsilon) produces zero savings in practice.

### Step 5: Threshold Learning

The threshold tau is parametrized via sigmoid to stay in [0, H_max]:

    tau = sigmoid(raw_tau) * H_max

raw_tau is a learnable scalar optimized alongside routing weights during
calibration.

**Revision note (Fix 2):** In the original implementation, raw_tau was not
properly unfrozen during calibration. After calling model.freeze(), raw_tau
(a bare mx.array, not inside an nn.Module) was not reached by
layer.pool.router.unfreeze(). The fix uses layer.pool.unfreeze(keys=["raw_tau"])
explicitly. With proper unfreezing, tau values range from 0.55 to 1.27
(vs. original 0.33-0.43), and the per-layer variation pattern changes.

## Auxiliary Losses

### Balance Loss (standard)

    L_bal = G * sum_g (mean_over_tokens(p_g))^2

Same as CapsuleMoE. Prevents routing collapse.

### Sparsity Loss (new)

    L_sparse = lambda * mean(alpha)

Encourages lower k by penalizing the soft k-selection variable.
lambda = 0 recovers standard routing. Higher lambda pushes toward k=1.

### Total Loss

    L = L_NTP + 0.01 * (L_bal + L_sparse)

## Computational Cost Analysis (Revised)

### Per-Token Routing Cost

Standard fixed-k:
- Router: G * d MADs (compute scores)
- Top-k: O(G log G) (sort or partial sort)
- Experts: k * C_expert FLOPs

Entropy-adaptive adds:
- Entropy: G multiply-adds + G log operations (from p * log(p))
- Sigmoid: 1 exp + 1 div
- TWO topk operations: 2 * O(G log G)
- Mask interpolation: 2G multiply-adds
- Renormalization: G multiply-adds + 1 div
- Per-expert max check (conditional execution): G comparisons

Total overhead: ~4G + 2*O(G log G) + 3 operations per token.

At G=8, d=64:
- Standard routing: 512 MADs
- Entropy overhead: ~35 operations (not 26 as claimed in V1)
- Overhead is ~7% of routing cost, ~0.08% of total layer compute

**However, the dominant cost at micro scale is not FLOPs but Python-level
loop overhead.** The per-expert max-weight check (Fix 1) adds a Python .item()
call and comparison for each expert in each layer, which is far more expensive
than the mathematical operations. Measured wall-clock: EA is 3.5-4.5x slower
than fixed k=2 (0.32-0.41s vs 0.09s for 20 eval batches).

### Compute Savings (Revised)

**Theoretical savings (unchanged):**

    savings = (k_fixed - avg_k) / k_fixed

At avg_k = 1.83 (best case, sc=0.3): savings = 8.5% of expert compute.

**Actual savings at micro scale: ZERO.**

Reasons:
1. Soft mask interpolation produces nonzero weights for all G experts
2. Even with epsilon-based skipping (eps=0.001), 0 of 8 experts are skipped
   in practice because min(p_g) > 0 for softmax outputs
3. The entropy/threshold/masking overhead exceeds any hypothetical savings
4. Wall-clock is 3.5-4.5x worse than fixed k=2

**Projected macro scenario (speculative, not validated):**

At G=256 with ReLU routing (not softmax), many experts have exactly zero
activation. In this regime, conditional execution becomes meaningful.
The entropy-based approach may have value if combined with ReLU routing
(entropy identifies how many experts to activate, ReLU identifies which ones).

## Worked Example (d=64, G=4)

Given hidden state x with routing scores s = [2.1, 0.5, 0.3, -0.1]:

    exp(s) = [8.17, 1.65, 1.35, 0.90]
    Z = 12.07
    p = [0.677, 0.137, 0.112, 0.075]

    H = -(0.677*log(0.677) + 0.137*log(0.137) + 0.112*log(0.112) + 0.075*log(0.075))
    H = -(0.677*(-0.390) + 0.137*(-1.988) + 0.112*(-2.187) + 0.075*(-2.590))
    H = -(-0.264 - 0.272 - 0.245 - 0.194)
    H = 0.975

    H_max = log(4) = 1.386

    With tau = 0.5 * 1.386 = 0.693:
    H = 0.975 > tau -> alpha near 1 -> use k=2

    With tau = 1.1:
    H = 0.975 < tau -> alpha near 0 -> use k=1

Note: even in the k=1 case (alpha near 0), the mask interpolation gives:
    M = (1-alpha)*M_1 + alpha*M_2

With alpha = sigmoid((0.975 - 1.1) / 0.1) = sigmoid(-1.25) = 0.223:
    M = 0.777 * [1,0,0,0] + 0.223 * [1,1,0,0] = [1.0, 0.223, 0, 0]
    weighted: w = p * M / sum(p*M) = [0.677, 0.031, 0, 0] / 0.707 = [0.957, 0.043, 0, 0]

The 2nd expert still gets 4.3% weight -- nonzero, cannot be skipped.

## Random-k Control Analysis (New, Fix 3)

The random-k baseline assigns k=1 with probability p_k1 (matched to observed
EA fraction) and k=2 otherwise. Quality matches EA within noise:

    E[val_loss | random_k] <= E[val_loss | entropy_adaptive]  (observed, 3 seeds)

This implies that the entropy criterion does not identify a meaningful subset
of tokens. The composition task is simply robust to randomly reducing a small
fraction (5-22%) of tokens to k=1. The information-theoretic justification
(routing entropy as a confidence measure) is correct in principle but irrelevant
in practice at G=8 where the softmax distribution is never sparse enough to
create clear confident/uncertain separation.

## Assumptions (Revised)

1. **Entropy correlates with routing quality**: NOT VALIDATED. Random k-selection
   matches entropy-based selection, suggesting entropy is not a useful signal
   for determining per-token compute needs at G=8.

2. **Expert redundancy at low entropy**: PARTIALLY CORRECT but irrelevant.
   When one expert dominates (p_1 >> p_2), the 2nd expert's contribution is
   indeed small. But dropping it randomly (independent of entropy) works equally
   well, so the entropy signal is not needed to identify these tokens.

3. **Threshold is stable**: QUESTIONABLE. With proper unfreezing, tau values
   vary substantially across seeds (0.55-1.27 range, high per-layer std).
   The "optimal" tau depends heavily on random initialization.

4. **Differentiable approximation is adequate**: CONFIRMED. The soft-to-hard
   gap is <0.1%. This is the one assumption that held up under revision.

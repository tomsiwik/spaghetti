# AIMD Expert Load Balancing: Mathematical Foundations

## Problem Setup

Consider a Mixture-of-Experts (MoE) layer with G expert groups. For each input
token x in R^d, a router produces scores s in R^G and selects top-k groups via
softmax routing:

    p_i = softmax(s)_i          for i = 1..G
    selected = top_k(s, k)
    w_i = p_i * I[i in selected] / sum_{j in selected} p_j

The **load fraction** for expert i over a batch of B*T tokens is:

    f_i = (1/BT) sum_{b,t} w_i(x_{b,t})

Fair allocation: f_i = 1/G for all i.

## Load Imbalance Metric

    Imbalance(f) = max_i(f_i) - min_i(f_i)

Perfect balance: Imbalance = 0. Worst case (one expert gets all): Imbalance = 1.

## Three Balancing Strategies

### 1. Auxiliary Loss (Switch Transformer, Fedus et al. 2022)

Add to training loss:

    L_bal = alpha_bal * G * sum_i (f_i * bar{p}_i)

where bar{p}_i = (1/BT) sum_{b,t} p_i(x_{b,t}) is the mean routing probability.

This is minimized when f_i = bar{p}_i = 1/G (uniform).

- **Gradient-based**: The balance loss contributes gradients to the router weights.
- **Coupled**: Balance competes with task loss in the optimizer.
- alpha_bal = 0.01 (standard coefficient).

### 2. AIMD Bias Feedback (This Work)

Add a non-learned bias b in R^G to routing logits:

    s_i = router(x)_i + b_i

Update b after each forward pass (NOT through gradients):

    For each expert i:
        excess = f_i - 1/G
        if excess > epsilon:    b_i <- beta * b_i - alpha * (excess / (1/G))
        if excess < -epsilon:   b_i <- b_i + alpha

Parameters:
- alpha = 0.05: additive increase step
- beta = 0.5: multiplicative decrease factor
- epsilon = 0.02: dead zone around target (prevents oscillation)

**Connection to TCP (Jacobson 1988, Chiu-Jain 1989):**
In TCP AIMD congestion control, each sender's congestion window w evolves:
- Additive Increase: w <- w + alpha/w (gentle linear growth)
- Multiplicative Decrease: w <- beta * w (aggressive halving on congestion)

The Chiu-Jain theorem proves this converges to the unique fair allocation
where all senders get equal bandwidth, for any 0 < beta < 1 and alpha > 0.

Our expert routing bias mirrors this: experts are "senders", tokens are
"bandwidth". The multiplicative decrease on overloaded experts provides
stronger correction than additive decrease (DeepSeek-V3 style).

**Connection to DeepSeek-V3 (auxiliary-loss-free):**
DeepSeek-V3 uses additive-only bias updates:
    b_i <- b_i - gamma * sign(f_i - 1/G)
This is symmetric (AI/AD = Additive Increase / Additive Decrease), which
converges to fair allocation but more slowly than AIMD.

### 3. No Balance (Control)

Pure softmax routing with no balancing mechanism. Expectation: router
exploits a few experts heavily, leaving others idle.

## Convergence Analysis

### Auxiliary loss convergence
The balance loss gradient with respect to router weights W_r is:

    dL_bal/dW_r = alpha_bal * G * sum_i (df_i/dW_r * bar{p}_i + f_i * dbar{p}_i/dW_r)

This provides continuous gradient signal at every step, modulated by the
optimizer's learning rate and momentum.

### AIMD convergence
The bias update is a discrete dynamical system:

    b(t+1) = F(b(t), f(t))

The fixed point is b* such that f_i(b*) = 1/G for all i. The Chiu-Jain
theorem guarantees convergence in the TCP setting where loads are monotonic
in bias. However, in neural routing:

1. The router weights W_r evolve simultaneously via gradient descent
2. The bias b modifies a moving target (router logits change each step)
3. The load function f(b) is non-monotonic due to softmax saturation

These violations of TCP assumptions mean AIMD convergence is NOT guaranteed
in the neural routing setting. The experiment tests whether it works in practice.

## Computational Cost

| Method | Training overhead | Memory | Params |
|--------|------------------|--------|--------|
| No balance | 0 | 0 | 0 |
| Aux loss | +1 gradient term per layer | O(G) | 0 (uses existing router) |
| AIMD | +G comparisons per layer per step | O(G) bias vector | 0 (bias is not learned) |

All three have identical model parameter counts: 203,136 at micro scale
(d=64, G=4, 64 capsules/group, 4 layers).

## Worked Example (d=64, G=4)

After a batch with loads f = [0.40, 0.10, 0.25, 0.25]:
- Target = 0.25, epsilon = 0.02
- Expert 0: excess = 0.15 > 0.02 -> b_0 = 0.5 * b_0 - 0.05 * (0.15/0.25) = 0.5*0 - 0.03 = -0.03
- Expert 1: excess = -0.15 < -0.02 -> b_1 = 0.0 + 0.05 = 0.05
- Expert 2: excess = 0.0 (in dead zone) -> no change
- Expert 3: excess = 0.0 (in dead zone) -> no change

New bias: [-0.03, 0.05, 0.0, 0.0]

After next batch (assume same router weights), the softmax shifts:
- Expert 0's logit decreases by 0.03 -> fewer tokens
- Expert 1's logit increases by 0.05 -> more tokens

The asymmetry is key: overloaded expert gets both multiplicative decay AND
an additive penalty proportional to how far it exceeds target.

## Assumptions

1. Load fractions f_i are meaningful estimates of expert utilization
   (requires sufficiently large batch)
2. Bias b affects routing scores additively (pre-softmax)
3. The dead zone epsilon prevents oscillation around equilibrium
4. Beta < 1 ensures multiplicative decrease is more aggressive than
   additive increase (the TCP "fairness asymmetry")
5. Router weights and AIMD bias updates do not destructively interfere
   (THIS IS THE KEY TESTABLE ASSUMPTION)

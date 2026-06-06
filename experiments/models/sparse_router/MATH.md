# Sparse Routing: Mathematical Foundations

## 1. Problem Statement

The capsule MoE composition protocol (capsule_moe MATH.md §6) uses top-k_g=2
group selection from G=8 composed groups (4 per domain). This activates 25% of
MLP parameters per token (before ReLU sparsity).

**Question:** Can top-1 group selection match top-2 quality within 5%, halving
the active compute? And does the learned softmax router outperform uniform
routing at this sparsity level?

This is VISION.md Exp 2 — the first test where learned routing faces a
meaningful challenge (G=8 composed vs G=4 single-domain where uniform ≈ learned).

---

## 2. Variable Definitions

All variables with dimensions/shapes, consistent with capsule_moe MATH.md:

```
d       = 64       -- model hidden dimension
G_d     = 4        -- capsule groups per domain (from capsule_moe)
D       = 2        -- number of domains (a-m, n-z)
G       = D * G_d  -- total composed groups (= 8)
P_g     = 64       -- capsules per group (P/G in capsule_moe notation)
P       = G * P_g  -- total capsules after composition (= 512)
k       in {1, 2, 4, 8}  -- top-k groups selected per token (sweep variable)
T       = 32       -- block size (sequence length)
n_layer = 4        -- transformer layers
V       = 27       -- vocabulary size (26 letters + BOS)

x       in R^d           -- token hidden state (input to capsule pool)
W_r     in R^{G x d}     -- softmax router weight matrix
s       in R^G            -- router logits: s = x @ W_r^T
p       in R^G            -- router probabilities: p = softmax(s)
m       in {0,1}^G        -- top-k selection mask
w       in R^G            -- masked, renormalized routing weights

A_g     in R^{P_g x d}   -- group g detector matrix (rows = capsule a_i^T)
B_g     in R^{d x P_g}   -- group g expansion matrix (cols = capsule b_i)
h_g     in R^{P_g}        -- capsule activations: h_g = ReLU(A_g @ x)
```

---

## 3. Top-k Group Routing Mechanism

### 3.1 Router Scoring

The softmax router computes group probabilities (capsule_moe MATH.md §3.1):

```
s = x @ W_r^T          s in R^G         (linear projection)
p = softmax(s)          p in R^G         (probability distribution)
```

### 3.2 Top-k Mask and Weight Computation

Given top-k parameter k:

```
threshold = k-th largest value in s
m_g = 1  if s_g >= threshold,  else 0       for g = 1..G
w_g = (p_g * m_g) / (sum_{g'} p_{g'} * m_{g'} + eps)
```

The mask m selects exactly k groups. The weights w are renormalized probabilities
over the selected groups, summing to 1.

### 3.3 Forward Pass

```
CapsulePool_k(x) = sum_{g=1}^{G} w_g * B_g @ ReLU(A_g @ x)
```

Since w_g = 0 for non-selected groups, only k terms contribute:

```
CapsulePool_k(x) = sum_{g in TopK(s)} w_g * B_g @ ReLU(A_g @ x)
```

### 3.4 The k=1 Special Case

At k=1, only the highest-scoring group fires:

```
g* = argmax_g s_g
w_{g*} = 1.0              (only group selected, renormalization gives 1)
CapsulePool_1(x) = B_{g*} @ ReLU(A_{g*} @ x)
```

The routing weight is always 1.0 — the router decides WHICH group, but doesn't
modulate strength. This is the hardest routing setting: one shot, no hedging.

### 3.5 The k=G Case (Dense)

At k=G (=8), all groups are selected:

```
w_g = p_g    for all g     (no masking, original softmax probs)
CapsulePool_G(x) = sum_{g=1}^{G} p_g * B_g @ ReLU(A_g @ x)
```

This is equivalent to a soft mixture of all groups — no sparsity.

---

## 4. Active Compute Analysis

### 4.1 Active Parameter Fraction

At each k, the active MLP parameters per token are:

```
active_params(k) = k * params_per_group = k * 2 * d * P_g
total_params     = G * 2 * d * P_g
active_ratio(k)  = k / G
```

| k | Active groups | Active ratio | Active MLP params |
|---|--------------|-------------|-------------------|
| 1 | 1 of 8       | 12.5%       | 8,192             |
| 2 | 2 of 8       | 25.0%       | 16,384            |
| 4 | 4 of 8       | 50.0%       | 32,768            |
| 8 | 8 of 8       | 100%        | 65,536            |

For reference, the dense GPT MLP has 8d^2 = 32,768 params per layer. So:
- k=1 uses 25% of dense MLP capacity
- k=2 uses 50% (current capsule_moe default)
- k=4 uses 100% (parameter-parity with dense MLP)
- k=8 uses 200% (over-parameterized, soft ensemble)

### 4.2 FLOPs Per Token Per Layer

Router FLOPs (constant for all k):

```
F_router = 2 * d * G + G         (matmul + softmax)
         = 2 * 64 * 8 + 8
         = 1,032
```

Per selected group (capsule computation):

```
F_group = 2 * d * P_g + 2 * d * P_g    (A_g @ x + B_g @ h_g)
        = 4 * d * P_g
        = 4 * 64 * 64
        = 16,384
```

Total (assuming conditional execution):

```
F_total(k) = F_router + k * F_group
```

| k | Router | Groups | Total | vs Dense MLP | vs k=2 |
|---|--------|--------|-------|-------------|--------|
| 1 | 1,032  | 16,384  | 17,416  | 27%        | 52%    |
| 2 | 1,032  | 32,768  | 33,800  | 52%        | 100%   |
| 4 | 1,032  | 65,536  | 66,568  | 102%       | 197%   |
| 8 | 1,032  | 131,072 | 132,104 | 202%       | 391%   |

Dense MLP: F_MLP = 16 * d^2 = 65,536.

**Key insight: top-1 halves FLOPs vs top-2 (theoretical).** At micro scale, the
implementation runs all groups and zeros non-selected (capsule_moe MATH.md §4.3),
so actual FLOPs are constant. The experiment measures QUALITY at different k,
with the understanding that FLOP savings only materialize with conditional
execution at larger G.

### 4.3 Effective Sparsity (Including ReLU)

After ReLU gating, empirically ~50% of capsules within a selected group produce
zero output (Li et al., "The Lazy Neuron Phenomenon"). The effective active
fraction is:

```
effective_ratio(k) = (k / G) * 0.5
```

| k | L1 ratio | Effective (with ReLU) |
|---|----------|----------------------|
| 1 | 12.5%    | ~6.25%               |
| 2 | 25.0%    | ~12.5%               |
| 4 | 50.0%    | ~25%                 |
| 8 | 100%     | ~50%                 |

At k=1, only ~6% of total capsule capacity is active per token. This is extreme
sparsity — the question is whether the router can concentrate useful computation
into that 6%.

---

## 5. Information-Theoretic Router Analysis

### 5.1 Router Entropy

The Shannon entropy of the router probability distribution p measures routing
confidence:

```
H(p) = -sum_{g=1}^{G} p_g * log(p_g)
```

Maximum entropy (uniform distribution):

```
H_max = log(G) = log(8) = 2.079 nats
```

### 5.2 Entropy-Sparsity Relationship

For top-k to work well, the router must concentrate probability mass on k groups.
The ideal distribution for top-k routing has:

```
p_g = 1/k    for g in TopK
p_g = 0      for g not in TopK
```

This gives entropy H = log(k):

| k | Ideal H | H / H_max | Description |
|---|---------|-----------|-------------|
| 1 | 0.000   | 0%        | Fully peaked — single group dominates |
| 2 | 0.693   | 33%       | Current capsule_moe target |
| 4 | 1.386   | 67%       | Moderate spread |
| 8 | 2.079   | 100%      | Uniform — routing is useless |

**Kill condition for router quality:** If the router entropy at k=1 is
H > 0.9 * H_max = 1.871 nats, the router is near-uniform and provides no
useful routing signal. This directly measures whether increasing G from 4 to 8
enables meaningful learned routing.

### 5.3 Routing Confidence Score

Define the top-k concentration ratio:

```
C_k = sum_{g in TopK(s)} p_g
```

This is the total probability mass captured by the selected groups.

| k | Ideal C_k | Meaning |
|---|-----------|---------|
| 1 | → 1.0     | Router is very confident in one group |
| 2 | → 1.0     | Top-2 captures nearly all probability |
| 8 | = 1.0     | Always 1.0 (all groups selected) |

If C_1 ≈ 1/8 = 0.125 (chance level), the router cannot differentiate groups.
If C_1 > 0.5, the router has a clear preference — top-1 captures meaningful signal.

---

## 6. Quality Degradation Model

### 6.1 Information Loss Framework

When reducing from k=G (dense) to k=1 (sparse), information is lost from the
unselected groups. Model the output of the full dense computation as:

```
y_dense = sum_{g=1}^{G} p_g * out_g        where out_g = B_g @ ReLU(A_g @ x)
```

The top-k approximation is:

```
y_k = sum_{g in TopK} w_g * out_g
```

The approximation error is:

```
err_k = ||y_dense - y_k||
      = ||sum_{g not in TopK} p_g * out_g - sum_{g in TopK} (w_g - p_g) * out_g||
```

After renormalization:

```
w_g = p_g / C_k       for g in TopK
w_g - p_g = p_g * (1/C_k - 1) = p_g * (1 - C_k) / C_k
```

So the error depends on both the probability mass of dropped groups (1 - C_k)
and the norm of their outputs.

### 6.2 When Top-1 Should Succeed

Top-1 selection preserves quality when:

1. **Group specialization**: Different groups learn different functions, and for
   any given token, one group's output dominates. This means ||out_{g*}|| >> ||out_g||
   for g ≠ g*, making the dropped terms small.

2. **Router confidence**: C_1 is high, meaning the router assigns most probability
   mass to one group. Then w_{g*} ≈ p_{g*} / C_1 ≈ 1 ≈ p_{g*} + residual,
   and the renormalization distortion is small.

3. **Redundancy across groups**: If groups within the same domain compute
   similar functions, dropping 3 of 4 intra-domain groups loses little. The
   inter-domain routing (selecting domain-a vs domain-b groups) matters more
   than intra-domain selection.

### 6.3 When Top-1 Should Fail

Top-1 fails when:
1. Information is distributed uniformly across groups (no specialization)
2. Multiple groups contribute non-redundantly for the same token
3. The router cannot concentrate probability mass (H ≈ H_max)

At micro scale, the risk is (1): with d=64 and simple character-level data,
groups may not specialize enough for one group to carry sufficient information.

---

## 7. Composition Protocol Extension

### 7.1 Training Protocol (Inherited)

The composition protocol is unchanged from capsule_moe MATH.md §6.3:

```
1. Train M_base on all data (300 steps, all params)
2. For each domain d:
   a. Initialize M_d from M_base
   b. Freeze attention + embeddings
   c. Fine-tune capsule groups only (300 steps)
3. Compose:
   a. Share W_attn from M_base
   b. Concatenate capsule groups: G = D * G_d = 8 groups
   c. Create router with output dim G=8
```

### 7.2 Router Calibration at Each k

The only change: calibrate the router at each top_k value independently.

```
For each k in {1, 2, 4, 8}:
   d. Initialize router W_r in R^{8 x 64} fresh
   e. Calibrate: 100 steps on mixed-domain data
      - Forward pass with top_k = k
      - Loss = L_CE + 0.01 * L_balance
      - Only W_r is trainable (capsule groups frozen)
   f. Evaluate on held-out validation set
```

**Why fresh router per k:** The optimal router weights differ across k. At k=1,
the router must make sharper decisions (low entropy). At k=8, the router only
modulates relative contributions (any entropy is fine). Training a single router
and sweeping k at eval would undercount k=1's potential.

### 7.3 Uniform Routing Baseline

For each k, also evaluate with uniform weights:

```
w_g = 1/k    for g in TopK (where TopK is chosen by score)
```

Wait — this is the same as the learned router with renormalization. For a true
uniform baseline:

```
Uniform top-k: select k groups randomly (or round-robin), weight 1/k each.
Uniform dense: all groups with weight 1/G = 1/8.
```

The comparison tests whether learned routing adds value beyond mere group selection.

---

## 8. Worked Numerical Example

### Setup

```
d = 64, G = 8, P_g = 64, T = 32
Composition: 4 groups from domain-a (a-m), 4 from domain-b (n-z)
```

### 8.1 Router Scores

Token x from domain-a (name "alice"):

```
s = x @ W_r^T = [2.1, 1.8, 3.5, 0.7, -0.3, -1.1, 0.2, -0.8]
                 |---- domain-a ----|  |---- domain-b ----|

p = softmax(s) = [0.17, 0.13, 0.69, 0.04, 0.02, 0.01, 0.03, 0.01]
                  sum ≈ 1.0
```

Router clearly prefers group 3 (domain-a, score 3.5).

### 8.2 Top-k Selection

**k=1:**
```
TopK = {3}
w_3 = 1.0
y_1 = out_3
C_1 = 0.69       (captures 69% of probability mass)
```

**k=2:**
```
TopK = {1, 3}
w_1 = 0.17 / (0.17 + 0.69) = 0.20
w_3 = 0.69 / (0.17 + 0.69) = 0.80
y_2 = 0.20 * out_1 + 0.80 * out_3
C_2 = 0.86       (captures 86%)
```

**k=4:**
```
TopK = {1, 2, 3, 7}
C_4 = 0.17 + 0.13 + 0.69 + 0.03 = 1.02 ≈ 1.0  (wait, need top by score)
Actually top-4 by score: s = [2.1, 1.8, 3.5, 0.7, -0.3, -1.1, 0.2, -0.8]
Top-4: groups {1,2,3,4} (scores 2.1, 1.8, 3.5, 0.7)
C_4 = 0.17 + 0.13 + 0.69 + 0.04 = 1.03 ≈ 1.0 (captures ~all mass)
```

**k=8 (dense):**
```
w_g = p_g for all g
y_8 = sum p_g * out_g
```

### 8.3 Quality Estimate

If out_3 dominates (group 3 specializes in "alice"-type tokens):

```
||y_1 - y_2|| ≈ 0.20 * ||out_1||    (small if out_1 ≈ out_3 direction)
||y_1 - y_8|| ≈ 0.31 * ||residual||  (domain-b groups contribute ~4%)
```

If group 3 carries most information for this token, top-1 loses very little.

### 8.4 Router Entropy

```
H = -sum p_g * log(p_g)
  = -(0.17*(-1.77) + 0.13*(-2.04) + 0.69*(-0.37) + 0.04*(-3.22)
      + 0.02*(-3.91) + 0.01*(-4.61) + 0.03*(-3.51) + 0.01*(-4.61))
  = -(−0.30 − 0.27 − 0.26 − 0.13 − 0.08 − 0.05 − 0.11 − 0.05)
  = -(-1.25)
  = 1.25 nats

H / H_max = 1.25 / 2.079 = 0.60
```

This is moderate entropy — the router has a clear preference (group 3 at 69%)
but isn't fully peaked. Good for top-1 (C_1 = 0.69 > 0.5).

### 8.5 Uniform Routing Comparison

With uniform weights at k=1 (random group selection):

```
E[y_uniform_1] = (1/8) * sum out_g     (expected value = average of one group)
y_learned_1   = out_{g*}               (best group selected)
```

If group quality varies, learned selection outperforms random by choosing the
highest-quality group. The margin depends on how much groups specialize.

---

## 9. Assumptions

1. **The softmax router differentiates composed groups (G=8).** At G=4
   (single domain), learned routing ≈ uniform routing. At G=8 (two domains
   composed), the router must distinguish 8 groups — 4 per domain. The router
   should assign domain-a tokens to domain-a groups and vice versa, with
   additional intra-domain specialization.

   *Justification:* The capsule_moe composition validates that the calibrated
   softmax router achieves -0.3% vs joint at k=2, G=8. It successfully routes
   at this scale. The question is whether it concentrates enough for k=1.

2. **Group specialization exists in the composed model.** Fine-tuning capsule
   groups on different domains (attention frozen) creates groups that respond
   differently to different inputs. Groups trained on domain-a should activate
   more strongly for domain-a tokens.

   *Justification:* Composition quality (-0.3% vs joint) with only 100 steps
   of router calibration implies the router finds and exploits group differences.
   If groups were identical, routing would be meaningless and composition would
   fail.

3. **Top-k selection with renormalization is a valid approximation.** The
   renormalized weights w_g = p_g / C_k inflate the contribution of selected
   groups. At k=1, w_{g*} = 1.0 regardless of p_{g*}. This could amplify noise
   if the router is uncertain.

   *Risk:* At k=1, a token where p_{g*} = 0.15 (barely leading) still gets
   w_{g*} = 1.0. The hard selection may amplify routing errors that soft
   weighting (k=2+) would smooth over.

4. **100 steps of router calibration suffices for each k value.** The
   capsule_moe composition uses 100 steps at k=2. Lower k may need more
   calibration (sharper routing requires more signal) or less (fewer parameters
   to effectively tune).

   *Mitigation:* Monitor validation loss convergence during calibration at
   each k. If k=1 hasn't converged at 100 steps, extend to 200.

5. **Load balancing loss behavior changes with k.** At k=1, only 1 of 8
   groups fires per token. The balance loss L_bal = G * sum(f_g^2) penalizes
   uneven group utilization. At k=1, extreme imbalance is more likely (one
   popular group could dominate). The coefficient alpha=0.01 may need adjustment.

   *Mitigation:* Log per-group activation frequency and balance loss at each k.
   If one group captures >50% of tokens at k=1, consider increasing alpha.

6. **Micro-scale limitations don't prevent meaningful group specialization.**
   At d=64 with character-level data, the capacity per group (P_g=64 capsules)
   is small. Groups within the same domain may learn nearly identical functions,
   making intra-domain routing useless.

   *Key insight:* Even if intra-domain routing is uninformative, INTER-domain
   routing (selecting domain-a vs domain-b groups) should work. The router
   only needs to correctly assign tokens to their domain's groups. Within a
   domain, any group suffices. This gives a floor of 50% useful routing
   decisions (correct domain) even without intra-domain specialization.

---

## 10. Falsification Criteria

The hypothesis is killed if ANY of:

1. **Top-1 degrades >10% vs top-2 in composition quality.** Measured as
   relative increase in validation loss:

   ```
   degradation = (val_loss_k1 - val_loss_k2) / val_loss_k2 * 100%
   ```

   Success: <5%. Kill: >10%.

2. **Learned top-1 loses to or ties uniform top-1.** If random group
   selection at k=1 matches learned selection, the softmax router provides
   no value at this sparsity level. Measured on 3 seeds; learned must win
   on mean val loss by >1 standard error.

3. **Router entropy at k=1 exceeds 0.9 * H_max.** If H > 1.871 nats at
   k=1 after calibration, the router is near-uniform and cannot make
   meaningful routing decisions. This directly tests whether G=8 enables
   differentiation.

4. **Top-1 degrades >15% vs joint training.** Even if top-1 is close to
   top-2, if both are far from joint, the issue is composition — not sparsity.
   Comparison to joint as a sanity check.

---

## 11. Measurements and Metrics

### 11.1 Primary Metrics (per seed, per k)

| Metric | Description |
|--------|-------------|
| val_loss | Validation loss on mixed-domain data |
| val_loss_a | Validation loss on domain-a only |
| val_loss_b | Validation loss on domain-b only |
| degradation_vs_k2 | (val_loss_k - val_loss_k2) / val_loss_k2 |
| degradation_vs_joint | (val_loss_k - val_loss_joint) / val_loss_joint |

### 11.2 Router Analysis Metrics (per layer, per k)

| Metric | Description |
|--------|-------------|
| router_entropy | H(p) averaged over validation tokens |
| concentration_k | C_k = sum of top-k probabilities |
| group_frequency | Per-group activation fraction over validation set |
| domain_alignment | Fraction of tokens where top-1 group matches token domain |
| balance_loss | L_bal value at end of calibration |

### 11.3 Baselines

| Baseline | Description |
|----------|-------------|
| joint | Trained on mixed data from scratch (no composition) |
| top-2 (capsule_moe) | Current validated composition protocol |
| uniform_k1 | Uniform weights, top-1 random group selection |
| uniform_dense | All groups, weight 1/8 each |

---

## 12. Computational Budget

### 12.1 Training (Per Seed)

```
Base pretraining:            300 steps (inherited, ~30s)
Domain-a fine-tuning:        300 steps (inherited, ~30s)
Domain-b fine-tuning:        300 steps (inherited, ~30s)
Joint baseline:              300 steps (inherited, ~30s)
Router calibration x4 (k=1,2,4,8): 4 * 100 steps = 400 steps (~40s)
Uniform baselines:           evaluation only (negligible)
```

Total per seed: ~1200 training steps + 400 calibration steps = ~1600 steps.
At ~3ms/step: ~5 seconds. With evaluation overhead: <1 minute per seed.

### 12.2 Total (3 Seeds)

```
3 * ~1 minute = ~3 minutes wall clock
```

Well within the 1-hour constraint.

### 12.3 Parameter Budget

No new parameters beyond capsule_moe composition. The router W_r is already
part of the composition protocol (G * d = 8 * 64 = 512 params per layer).
The only change is the top_k hyperparameter.

Total model: 203,136 params (same as capsule_moe at V=27).
Additional params: 0 (only a hyperparameter sweep).

---

## 13. Expected Outcomes

### 13.1 Optimistic Scenario (top-1 works)

```
k=1: val_loss within 3% of k=2
     router entropy < 1.0 nats (peaked)
     learned top-1 beats uniform top-1 by >5%
     domain_alignment > 80%
```

This would validate: (a) learned routing matters at G=8, (b) sparse computation
is viable, (c) the softmax router concentrates information sufficiently.

### 13.2 Pessimistic Scenario (top-1 fails)

```
k=1: val_loss degrades >10% vs k=2
     router entropy > 1.5 nats (near-uniform)
     learned ≈ uniform at k=1
     domain_alignment ≈ 50% (chance)
```

This would mean: information is distributed across groups at micro scale, and
no single group captures enough. The fix would be: (a) larger capacity per group,
(b) stronger domain diversity, or (c) auxiliary specialization losses.

### 13.3 Intermediate Scenario (gradual tradeoff)

```
k=1: 5-10% degradation vs k=2
k=4: < 1% degradation vs k=2
monotonic quality-compute curve with diminishing returns
```

This would mean: sparsity has a cost, but it's smooth and predictable. The
Pareto-optimal k depends on the quality-compute tradeoff. Most informative
outcome for the research program — it maps the frontier.

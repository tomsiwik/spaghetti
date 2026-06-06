# Contrastive Routing Keys: Mathematical Foundations

## 1. Problem Statement

The capsule MoE composition protocol (MATH.md, capsule_moe) requires ~100 steps
of mixed-domain router calibration. The router learns to discriminate domains
through reconstruction loss — an indirect signal. Meanwhile, LoRA A-matrices
fail entirely as routing keys (~50% accuracy, VISION.md Exp 1).

**Question:** Can we train dedicated routing keys with a contrastive loss that
explicitly optimizes for discrimination, achieving >85% routing accuracy with
fewer samples (~50/domain) and fewer steps (~50)?

---

## 2. Variable Definitions

All variables with dimensions/shapes:

```
d       = 64       -- model hidden dimension
d_key   = 8        -- routing key dimension (compression factor d/d_key = 8)
G       = 4        -- capsule groups per domain (from capsule_moe)
D       = 2        -- number of domains (a-m, n-z)
N       = D * G    -- total groups after composition (= 8)
k_g     = 2        -- top-k groups selected per token (per domain)
P       = 256      -- total capsules per domain (P/G = 64 per group)
T       = 32       -- block size (sequence length)
S       = 50       -- calibration samples per domain
L       = 50       -- calibration optimization steps
tau     = 0.1      -- InfoNCE temperature (scalar)

x       in R^d         -- token hidden state (input to routing)
K_i     in R^{d x d_key}  -- routing key matrix for group i, i = 1..N
q       in R^{d_key}       -- query vector derived from x (see Section 3)
k_i     in R^{d_key}       -- key vector for group i (see Section 3)
```

---

## 3. Contrastive Routing Keys

### 3.1 Key Architecture

Each composed group `i` (i = 1..N) has a routing key matrix `K_i`. Given a
token hidden state `x in R^d`, the routing score for group `i` is:

```
z_i = x @ K_i          -- z_i in R^{d_key}
s_i = ||z_i||^2        -- scalar routing score
```

This is a projection-then-norm scoring: `K_i` projects `x` into a d_key-dimensional
routing space, and the squared norm measures how strongly `x` activates key `i`.

**Why squared norm, not dot product?** A single shared query would assume
domains differ along a single direction. Squared norm measures activation
magnitude in the key's subspace — each key defines its own d_key-dimensional
"receptive field" and tokens that strongly project into that subspace score
high. This naturally handles multi-dimensional domain boundaries.

### 3.2 Comparison to Capsule MoE Router

The capsule MoE group router (Section 3.1 of capsule_moe MATH.md) uses:

```
s = x @ W_r^T,     W_r in R^{G x d},   s in R^G
p = softmax(s)
```

This is a linear classifier over groups trained end-to-end with reconstruction
loss. It has `G * d` parameters per layer and learns routing implicitly.

The contrastive key replaces this with `N` key matrices, each `d x d_key`:

```
s_i = ||x @ K_i||^2,    K_i in R^{d x d_key},    i = 1..N
```

with routing scores trained explicitly for discrimination via InfoNCE loss.

---

## 4. InfoNCE Training Loss

### 4.1 Data Setup

From the training data, collect `S` labeled samples per domain:

```
X_1 = {x_1^{(1)}, ..., x_1^{(S)}}    -- hidden states from domain 1 (a-m)
X_2 = {x_2^{(1)}, ..., x_2^{(S)}}    -- hidden states from domain 2 (n-z)
```

Hidden states are extracted from the frozen base model's transformer blocks
(after layer norm, before the MLP). Each sample is one token embedding.

### 4.2 InfoNCE Loss Derivation

For a token `x` from domain `d` (d in {1, ..., D}), the positive groups are
those trained on domain `d` (indices `G*(d-1)+1` through `G*d`), and the
negative groups are all others.

**Simplification for D=2:** Domain `d` has a set of G group indices. We define
the routing target as the domain-level assignment (not individual group selection
within a domain):

```
score_d(x) = max_{i in groups(d)} s_i(x)
           = max_{i in groups(d)} ||x @ K_i||^2
```

Taking the max over groups within a domain: the domain is "selected" if any of
its groups scores highest. This avoids forcing individual group specialization
during routing key training — the keys only need to discriminate between domains.

The per-sample InfoNCE loss for a token `x` from domain `d`:

```
L_InfoNCE(x, d) = -log( exp(score_d(x) / tau) / sum_{d'=1}^{D} exp(score_{d'}(x) / tau) )
```

Expanding:

```
score_d(x) = max_{i in groups(d)} ||x @ K_i||^2

L_InfoNCE(x, d) = -score_d(x)/tau + log( sum_{d'=1}^{D} exp(score_{d'}(x) / tau) )
```

### 4.3 Batch Loss

Over a batch of `B = D * S` labeled samples:

```
L_batch = (1/B) * sum_{d=1}^{D} sum_{j=1}^{S} L_InfoNCE(x_d^{(j)}, d)
```

The gradient flows only through the `K_i` parameters. All other model weights
(attention, embeddings, capsule groups) remain frozen.

### 4.4 Temperature `tau`

The temperature controls the sharpness of the contrastive distribution:

- `tau -> 0`: Hard assignment. Gradient concentrates on the highest-scoring
  negative. Can cause training instability.
- `tau -> inf`: Uniform distribution. No learning signal.
- `tau = 0.1` (chosen): Moderate sharpness. Standard in contrastive learning
  (SimCLR, CLIP). At d_key=8, typical score magnitudes are O(1) to O(10),
  so tau=0.1 gives a dynamic range of ~exp(100) between well-separated domains.

**Sensitivity analysis needed:** If domains are very similar (a-m vs n-z), the
score gap may be small, and tau=0.1 may be too aggressive. The experiment
should sweep tau in {0.05, 0.1, 0.5, 1.0}.

---

## 5. Routing at Inference

### 5.1 Domain-Level Routing

After training keys, routing a token `x` at inference:

```
1. Compute scores:   s_i = ||x @ K_i||^2      for i = 1..N
2. Domain scores:    score_d = max_{i in groups(d)} s_i    for d = 1..D
3. Select domain:    d* = argmax_d score_d
4. Within domain:    apply capsule_moe routing (softmax over groups(d*))
```

Step 4 uses the existing capsule MoE group router, but restricted to the
selected domain's groups. This is a two-stage routing: contrastive keys
select the domain, then the standard softmax router selects groups within it.

### 5.2 Soft Routing Alternative

Instead of hard domain selection, use soft weighting over all N groups:

```
1. Compute scores:   s_i = ||x @ K_i||^2      for i = 1..N
2. Probabilities:    p_i = softmax(s / tau)    (temperature-scaled)
3. Top-k selection:  select top-2*k_g groups from all N
4. Apply:            standard capsule MoE weighted combination
```

This is simpler (no two-stage routing) and gracefully handles ambiguous tokens.
It reduces to the capsule MoE router mechanism, but with scores computed via
contrastive keys instead of a linear classifier.

**Recommendation:** Start with soft routing (5.2) as it directly replaces the
capsule MoE router. Hard routing (5.1) is a refinement for sparse computation.

---

## 6. Routing Accuracy Metric

### 6.1 Definition

For a held-out set of labeled tokens, routing accuracy is:

```
accuracy = (1/M) * sum_{j=1}^{M} 1[d*(x_j) == label(x_j)]
```

where `d*(x_j) = argmax_d score_d(x_j)` and `label(x_j)` is the true domain.

### 6.2 Baselines

| Method | Expected Accuracy | Source |
|--------|-------------------|--------|
| Random routing | 50% (D=2) | Chance |
| A-matrix self-routing | ~50% | Exp 1 (coin flip) |
| Softmax router (calibrated) | ~95%* | capsule_moe composition |
| **Contrastive keys (target)** | **>85%** | This experiment |

*Estimated from the -0.3% composition quality of calibrated router.

### 6.3 Success Criteria

1. **Primary:** Routing accuracy >85% on held-out tokens
2. **Sample efficiency:** Achieve this with S <= 50 samples/domain
3. **Step efficiency:** Achieve this in L <= 50 optimization steps
4. **Composition quality:** Val loss within 5% of joint training when using
   contrastive keys for routing in the composed model

---

## 7. Parameter and Compute Analysis

### 7.1 Parameter Count

Routing keys per layer:

```
params_keys = N * d * d_key
            = 8 * 64 * 8
            = 4,096
```

Compare to capsule MoE router per layer:

```
params_router = G * d = 4 * 64 = 256     (single domain)
params_router_composed = N * d = 8 * 64 = 512   (composed, D=2)
```

**Key overhead: 4,096 vs 512 = 8x more parameters for routing.**

But in absolute terms: 4,096 per layer * 4 layers = 16,384 total key params.
This is 8% of the capsule pool (32,768/layer * 4 = 131,072) and 8% of total
model params (~203K). This is within the <200K additional params constraint.

Total model with contrastive keys:

```
P_contrastive = P_capsule_moe + n_layer * N * d * d_key
              = 203,136 + 4 * 4,096
              = 203,136 + 16,384
              = 219,520
```

**219,520 total params. 16,384 additional params (8.1% overhead).**

### 7.2 Calibration Cost

Training data: `D * S = 2 * 50 = 100` labeled hidden states per layer.

Per step, the forward pass through keys:

```
FLOPS_keys = B * N * (2 * d * d_key + d_key)
           = 100 * 8 * (2 * 64 * 8 + 8)
           = 100 * 8 * 1,032
           = 825,600
```

Over L=50 steps (forward + backward ~= 3x forward):

```
FLOPS_calibration = L * 3 * FLOPS_keys
                  = 50 * 3 * 825,600
                  = 123,840,000   (~124M FLOPs)
```

This is negligible — a single forward pass of the full model on one batch is
~4M FLOPs per layer * 4 layers = ~16M FLOPs. Calibration is ~8 full forward passes.

### 7.3 Inference Cost

Per token at inference:

```
FLOPS_routing = N * (2 * d * d_key + d_key)    -- project + norm
              = 8 * (1,024 + 8)
              = 8,256
```

Compare to capsule MoE router:

```
FLOPS_router = 2 * N * d    -- linear projection
             = 2 * 8 * 64
             = 1,024
```

**Contrastive routing is 8x more expensive than linear routing.** But both are
negligible vs. the capsule computation (33,280 FLOPs for k_g=2 groups). The
routing overhead goes from 0.5K to 8.3K — from 1.5% to 25% of per-token FLOPs.

At d_key=4 (halved), routing drops to ~4K FLOPs (12% overhead). The experiment
should test d_key in {4, 8, 16} to find the accuracy/cost Pareto frontier.

---

## 8. Worked Numerical Example

### Setup

```
d = 64, d_key = 8, G = 4, D = 2, N = 8, tau = 0.1
```

Domain 1 (a-m) owns groups 1-4. Domain 2 (n-z) owns groups 5-8.

### Step 1: Extract hidden states

From the frozen base model, collect 50 tokens per domain. Each token produces
`x in R^64` at each layer. For simplicity, work with one layer.

```
X_1 = {x_1^(1), ..., x_1^(50)}    each in R^64    (a-m tokens)
X_2 = {x_2^(1), ..., x_2^(50)}    each in R^64    (n-z tokens)
```

### Step 2: Initialize keys

```
K_i ~ N(0, 1/sqrt(d)) for i = 1..8
Each K_i in R^{64 x 8}
```

### Step 3: Compute scores for one a-m token

```
x = x_1^(1) in R^64

z_1 = x @ K_1 in R^8,    s_1 = ||z_1||^2 = 3.7
z_2 = x @ K_2 in R^8,    s_2 = ||z_2||^2 = 2.1
z_3 = x @ K_3 in R^8,    s_3 = ||z_3||^2 = 4.2
z_4 = x @ K_4 in R^8,    s_4 = ||z_4||^2 = 1.8
z_5 = x @ K_5 in R^8,    s_5 = ||z_5||^2 = 2.9
z_6 = x @ K_6 in R^8,    s_6 = ||z_6||^2 = 3.1
z_7 = x @ K_7 in R^8,    s_7 = ||z_7||^2 = 1.5
z_8 = x @ K_8 in R^8,    s_8 = ||z_8||^2 = 2.4
```

Before training: scores are random. Domain 1 max = max(3.7, 2.1, 4.2, 1.8) = 4.2.
Domain 2 max = max(2.9, 3.1, 1.5, 2.4) = 3.1. Domain 1 wins by luck.

### Step 4: Compute InfoNCE loss

```
score_1 = max(s_1, s_2, s_3, s_4) = 4.2    (domain 1, correct)
score_2 = max(s_5, s_6, s_7, s_8) = 3.1    (domain 2, incorrect)

L = -score_1/tau + log(exp(score_1/tau) + exp(score_2/tau))
  = -4.2/0.1 + log(exp(42) + exp(31))
  = -42 + log(exp(42) * (1 + exp(-11)))
  = -42 + 42 + log(1 + exp(-11))
  = log(1 + 1.67e-5)
  = 1.67e-5
```

When the correct domain scores much higher, loss is near zero. When scores are
close or wrong domain leads, loss is large — proportional to the score gap.

### Step 5: After 50 steps of training

Keys converge such that domain-specific groups score high for their own tokens:

```
a-m token:  domain_1_score = 8.5,  domain_2_score = 1.2  -> correct routing
n-z token:  domain_1_score = 0.9,  domain_2_score = 7.8  -> correct routing
```

Routing accuracy on held-out tokens: target >85%.

---

## 9. Assumptions

1. **Domain-discriminative signal exists in hidden states.** The base model's
   hidden representations carry enough domain information for a linear probe
   (K_i is a linear projection) to distinguish a-m from n-z tokens. If the
   representation is domain-agnostic at the hidden state level, no key can
   discriminate.

   *Justification:* The softmax router in capsule_moe composition learns to
   discriminate domains in ~100 steps, suggesting the signal exists. A linear
   probe on hidden states should find it.

2. **50 samples suffice for d_key=8 dimensions.** With 8 free parameters per
   key column and 50 samples, we have a 50/8 ≈ 6x sample-to-parameter ratio
   per dimension. Total trainable params = N * d * d_key = 4,096. With 100
   samples and 50 steps, each sample is seen 50 times — possible overfitting.

   *Mitigation:* Regularize key norms. Monitor train vs. held-out accuracy.

3. **InfoNCE with max-pooled group scores is a valid contrastive objective.**
   The max over groups within a domain is not differentiable everywhere (has
   kinks), but works in practice with straight-through or soft-max relaxation.

   *Implementation:* Use `torch.max` which provides gradients to the argmax
   element (equivalent to straight-through for the max operation).

4. **Temperature tau=0.1 is appropriate for these score magnitudes.** Scores
   `s_i = ||x @ K_i||^2` scale with d_key and input norm. If scores are
   typically O(1-10), tau=0.1 gives good gradient signal. If scores are
   O(100+), tau should be larger.

   *Mitigation:* Normalize scores by d_key before applying temperature, or
   sweep tau.

5. **Routing keys can be trained independently per layer.** Each layer's
   hidden states are different, so each layer needs its own set of keys.
   The calibration extracts hidden states layer by layer and trains keys
   independently.

   *Justification:* The capsule MoE router is already per-layer. Same
   approach.

6. **At micro scale (d=64, a-m vs n-z), domains are distinguishable.**
   This is the main risk. a-m and n-z name distributions overlap in character
   frequencies. The discriminative signal may be weak.

   *Kill criterion:* If routing accuracy < 70% even with favorable
   hyperparameters (tau, d_key sweeps), the mechanism fails at this scale.

---

## 10. Falsification Criteria

The hypothesis is killed if ANY of:

1. **Routing accuracy < 70%** on held-out tokens after 50 steps of InfoNCE
   training with 50 samples/domain, even after sweeping tau and d_key.
   (85% is success; 70-85% is "promising but needs work"; <70% is dead.)

2. **Composition quality degrades by >10%** vs joint training when contrastive
   keys replace the calibrated softmax router. (5% is the target; 10% is the
   kill threshold.)

3. **Contrastive keys require more data/steps than the softmax router** they
   replace. If >100 samples or >100 steps are needed, the approach offers no
   advantage over the existing calibration protocol.

4. **Routing accuracy matches but is not better than a simple linear probe**
   (i.e., a single-layer classifier on hidden states). If a linear probe with
   the same data achieves the same accuracy in the same steps, the contrastive
   loss adds nothing — just use a classifier.

---

## 11. Computational Budget

### Training Phase (inherited from capsule_moe)

```
Base pretraining:     300 steps on all data
Domain fine-tuning:   300 steps/domain (capsule groups only, attention frozen)
```

This is the same as capsule_moe composition. No additional cost.

### Calibration Phase (NEW — this experiment)

```
Sample collection:    1 forward pass on 50 tokens/domain (negligible)
Key training:         50 steps on 100 samples (< 1 second at micro scale)
Total:                < 1 minute wall clock
```

### Comparison to Softmax Router Calibration

```
Softmax calibration:  ~100 steps on mixed-domain batches
                      Requires running full forward pass (attention + capsules)
                      per step to compute reconstruction loss

Contrastive keys:     ~50 steps on 100 pre-extracted hidden states
                      Only trains key matrices (no full forward pass needed)
                      ~10x cheaper per step (keys only vs full model)
```

**Total experiment time (including base training + fine-tuning): well under 1 hour.**

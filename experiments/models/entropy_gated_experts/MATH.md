# Entropy-Gated Expert Selection: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| x_t | (V,) | Base model logit vector at token position t |
| p_t | (V,) | softmax(x_t), predicted distribution over vocabulary V |
| H(p_t) | scalar | Shannon entropy of p_t |
| tau | scalar | Entropy threshold for gating |
| N | int | Number of domain experts |
| PPL_base | scalar | Token-weighted perplexity of base model (no adapters) |
| PPL_comp | scalar | Token-weighted perplexity with always-compose (all N adapters, 1/N scale) |
| PPL_gate | scalar | Token-weighted perplexity of entropy-gated system |
| f_skip | [0,1] | Fraction of tokens where composition is skipped |
| CV | scalar | Coefficient of variation (std/mean) of entropy distribution |
| eta | [0,1] | Otsu between-class variance ratio |

## Core Mechanism

### Token-Level Entropy

For each token position t, the base model produces logits x_t in R^V.
The predicted distribution is:

    p_t = softmax(x_t)

The Shannon entropy (in nats) is:

    H(p_t) = -sum_v p_t[v] * log(p_t[v])

**Bounds:** 0 <= H(p_t) <= log(V). For BitNet-2B-4T with V=32000:
- Minimum: H = 0 (model perfectly confident, one token has probability 1)
- Maximum: H = log(32000) ~ 10.37 nats (uniform distribution)

### Entropy Gating Decision

For each token t:

    if H(p_t) < tau:
        output_t = base_model(input_t)        # skip composition
    else:
        output_t = composed_model(input_t)     # apply N adapters

### Why Entropy, Not Softmax Max?

Softmax max (max_v p_t[v]) is a poor confidence measure because:
1. It ignores the shape of the distribution (peaked vs flat)
2. Calibration: high max != correct prediction (softmax miscalibration)
3. Entropy captures the full distributional uncertainty

However, entropy has its own calibration issues (noted in prior art).

## Kill Criterion K1: Sufficient Spread for Thresholding

The distribution of {H(p_t) : t in corpus} must have sufficient spread
for Otsu's method to find a meaningful threshold. We test two conditions:

### Condition 1: Coefficient of Variation

    CV = std({H(p_t)}) / mean({H(p_t)})

**Decision:** If CV < 0.5, the entropy values are too clustered relative
to their mean for thresholding to be meaningful -> KILL.

**Rationale:** CV measures relative spread. A CV of 0.5 means the standard
deviation is half the mean. Below this, most tokens have similar entropy
and any threshold would arbitrarily split a homogeneous population.

### Condition 2: Otsu Between-Class Variance Ratio

Given Otsu threshold tau*, compute:

    w_0 = |{t : H(p_t) < tau*}| / T    (weight of "confident" class)
    w_1 = |{t : H(p_t) >= tau*}| / T   (weight of "uncertain" class)
    mu_0 = mean({H(p_t) : H(p_t) < tau*})
    mu_1 = mean({H(p_t) : H(p_t) >= tau*})
    sigma_B^2 = w_0 * w_1 * (mu_0 - mu_1)^2    (between-class variance)
    sigma_T^2 = var({H(p_t)})                    (total variance)
    eta = sigma_B^2 / sigma_T^2

**Decision:** If eta < 0.15, the threshold does not explain enough
variance to be meaningful -> KILL.

**Rationale:** eta is always in [0, 1]. It measures what fraction of
total variance is explained by the two-class split. Values near 1
mean the threshold perfectly separates two distinct groups. The 0.15
threshold is conservative -- even 15% of variance explained is enough
for the gate to be useful.

**Note on bimodality (v1 correction):** We do NOT require or test
bimodality. Otsu's method works on any distribution with two groups
having different means. The v1 paper's dip test was incorrectly
implemented (KS-distance from uniform, not Hartigan's dip statistic)
and has been removed.

### Otsu's Method for Threshold Selection

Find tau by minimizing intra-class variance (Otsu, 1979):

    tau* = argmin_tau [ w_0(tau) * var_low(tau) + w_1(tau) * var_high(tau) ]

Equivalently, maximize between-class variance (sigma_B^2).

## Kill Criterion K2: PPL Comparison (Token-Weighted)

Both PPL_gate and PPL_comp must be computed as token-weighted averages:

    PPL = exp( (1/T) * sum_t CE(t) )

where CE(t) is the cross-entropy loss at token t and T is total tokens.

**Decision:** If PPL_gate > PPL_comp * 1.05 (more than 5% degradation),
gating loses too much quality -> KILL.

**Rationale (v2 correction):** The v1 paper compared token-weighted
gated PPL against arithmetic mean of per-domain PPLs, which inflated
the compose baseline due to domain imbalance (legal domain PPL ~20 on
3470 tokens vs python PPL ~2.5 on 2030 tokens). Token-weighted compose
PPL is the correct baseline.

The 5% tolerance accounts for the fact that gating is an approximation.
The value proposition is "skip 63% of composition work for only X%
quality loss." Even 1-2% degradation is acceptable if it enables
significant serving efficiency.

## Kill Criterion K3: Skip Fraction

    f_skip = |{t : H(p_t) < tau}| / |{all tokens}|

Must be >= 0.10 (10%) for the approach to matter at all.
Success criterion S1 requires f_skip >= 0.30 with < 1% PPL degradation.

## Computational Cost Analysis

### Always-Compose (Baseline)
For each token: one forward pass through base + all N adapters pre-merged.
Cost: C_fwd (one pass, adapters merged into weights beforehand).

### Entropy-Gated (Two-Pass, Current Implementation)
For each token:
1. Forward pass through base: C_fwd
2. Compute entropy from logits: O(V) (negligible vs C_fwd)
3. If H > tau: re-run with composed model: C_fwd (second pass)

**Net cost per token:**
- Confident token (H < tau): C_fwd (same as base-only)
- Uncertain token (H >= tau): 2 * C_fwd (base + composed)

**Expected total cost:**
    E[cost] = f_skip * C_fwd + (1 - f_skip) * 2 * C_fwd
            = (2 - f_skip) * C_fwd

For f_skip = 0.63: E[cost] = 1.37 * C_fwd -- SLOWER than always-compose.

### Entropy-Gated (Correct Architecture: Pre-Filter for Routing)

The real value is not in the two-pass design but in using entropy as a
pre-filter before the routing heads:

For each token:
1. Forward pass through base: C_fwd (always needed)
2. Compute entropy: O(V) (negligible)
3. If H < tau: use base output, skip routing heads entirely
4. If H >= tau: run routing heads (cost C_route) + compose top-k

    E[cost] = C_fwd + (1 - f_skip) * (C_route + C_compose_topk)

For f_skip = 0.63 and C_route << C_fwd:
    E[cost] ~ C_fwd + 0.37 * C_route

This saves the routing computation for 63% of tokens.

## Worked Example (Micro Scale)

d = 2560, N = 5, V = 32000, r = 16

Evaluate 13,652 tokens:
- 8,616 tokens have H(p_t) < 2.10 (confident, f_skip = 0.631)
- 5,036 tokens have H(p_t) >= 2.10 (uncertain)

PPL_comp (token-weighted) = 7.00
PPL_gate (Otsu) = 7.08, degradation = +1.13%

Per-domain degradation:
- python: +1.35% (89% skip)
- math: +1.86% (76% skip)
- medical: +0.82% (59% skip)
- legal: +0.53% (37% skip)
- creative: +1.26% (67% skip)

The mechanism trades 1.13% quality for 63% fewer composition operations.

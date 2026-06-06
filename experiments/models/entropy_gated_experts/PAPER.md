# Entropy-Gated Expert Selection: Research Digest

## Revision History

- **v2 (2026-03-26):** Fixed two blocking issues from adversarial review.
  (1) Replaced incorrect dip test / bimodality claim with CV + Otsu eta
  spread tests. (2) Fixed PPL comparison to use token-weighted compose PPL
  instead of arithmetic mean. Added per-domain gated vs compose breakdown.
  Used Otsu threshold consistently in Phase 3.

## Hypothesis

Base model output entropy has sufficient spread to define a meaningful
threshold that separates tokens benefiting from expert composition (high
entropy) from those that do not (low entropy), enabling a binary gate
that skips expert composition for confident tokens with small quality cost.

## What This Model Is

A binary gating mechanism for the composable ternary experts architecture.
At inference time, for each token position:

1. Run the base model (BitNet-2B-4T) forward pass
2. Compute Shannon entropy of the output distribution
3. If entropy < threshold: use base output directly (skip experts)
4. If entropy >= threshold: run the composed model (uniform 1/N merge of
   all domain adapters)

The threshold is determined by Otsu's method on the base model entropy
distribution.

## Key References

- MoBiLE (Mixture of Big-Little Experts): Token importance routing,
  skips experts for "unimportant" tokens, 1.6-1.7x speedup
- pQuant (Decoupled MoE): Forces most params into 1-bit backbone,
  routes only sensitive tokens to 8-bit expert
- EdgeNav-QE: Dynamic early exit, 82.7% latency reduction
- CALM (Schuster et al., 2022): Confident Adaptive Language Modeling,
  softmax confidence for early exit
- DeeBERT (Xin et al., 2020): Entropy-based early exit for BERT
- exp_entropy_adaptive_router (KILLED): Variable-k within MoE layers --
  entropy adds no value. THIS experiment is fundamentally different:
  binary skip-all-composition gate, not variable-k selection.

## Empirical Results

### Setup
- Base model: Microsoft BitNet-b1.58-2B-4T (2.4B params, d=2560)
- Adapters: 5 domain LoRA adapters (python, math, medical, legal, creative),
  rank-16, reused from exp_tiny_routing_heads
- Composition: uniform 1/N merge of all 5 adapters
- Evaluation: 25 validation samples per domain, per-token cross-entropy

### Phase 1: Entropy Distribution (K1)

| Statistic | Value |
|-----------|-------|
| Total tokens | 13,652 |
| Mean entropy (nats) | 1.76 |
| Std entropy | 1.53 |
| CV (coefficient of variation) | 0.87 |
| Min / Max | 0.00 / 8.44 |
| P5 / P25 / P50 / P75 / P95 | 0.004 / 0.41 / 1.53 / 2.72 / 4.59 |
| Otsu between-class variance ratio (eta) | 0.68 |
| Otsu threshold | 2.10 nats |

**K1: PASS.** The entropy distribution has high spread (CV = 0.87, well
above the 0.5 threshold) and Otsu's method finds a highly effective
threshold (eta = 0.68, meaning the threshold explains 68% of the total
variance -- far above the 0.15 minimum). This confirms the distribution
has sufficient structure for meaningful thresholding, without making
any claim about bimodality.

**Note on bimodality (dropped from v1):** The v1 paper claimed bimodality
based on a dip test that was incorrectly implemented (KS-distance from
uniform, not Hartigan's dip statistic). The bimodality coefficient also
failed (0.516 < 0.555). We no longer claim bimodality. Otsu's method
does not require bimodality -- it only requires two classes with different
means, which eta = 0.68 strongly confirms.

**Domain breakdown of confident tokens (f_skip at Otsu threshold):**

| Domain | Mean H | f_skip |
|--------|--------|--------|
| python | 0.79 | 89.0% |
| math | 1.27 | 75.7% |
| medical | 1.94 | 59.1% |
| legal | 2.61 | 37.4% |
| creative | 1.72 | 66.7% |

### Phase 2: PPL Comparison (K2, K3, S1, S2)

#### Per-Domain PPL

| Domain | Base PPL | Compose PPL | Improvement |
|--------|----------|-------------|-------------|
| python | 2.74 | 2.51 | -8.4% |
| math | 5.54 | 4.96 | -10.6% |
| medical | 6.95 | 6.20 | -10.8% |
| legal | 21.85 | 20.41 | -6.6% |
| creative | 6.34 | 5.96 | -6.0% |

#### Aggregate PPL (token-weighted)

| Metric | Value |
|--------|-------|
| Token-weighted base PPL | 7.59 |
| Token-weighted compose PPL | 7.00 |
| Arithmetic mean compose PPL | 8.01 (inflated by legal domain, NOT used for K2) |

#### Entropy-Gated PPL vs Token-Weighted Compose

| Threshold | Tau (nats) | Gated PPL | f_skip | vs TW Compose |
|-----------|-----------|-----------|--------|---------------|
| p10 | 0.023 | 7.000 | 10.0% | +0.05% |
| p20 | 0.215 | 7.010 | 20.0% | +0.20% |
| p30 | 0.637 | 7.024 | 30.0% | +0.40% |
| **p50** | **1.531** | **7.052** | **49.9%** | **+0.80%** |
| **Otsu** | **2.103** | **7.076** | **63.1%** | **+1.13%** |
| p70 | 2.453 | 7.089 | 69.8% | +1.32% |

**K2: PASS.** At the Otsu threshold, gated PPL (7.076) is 1.13% worse than
token-weighted compose PPL (6.997), well within the 5% tolerance. The
degradation is monotonic and gradual -- each additional 10% of skipped
tokens costs roughly 0.1-0.2% PPL.

**K3: PASS.** At Otsu threshold, 63.1% of tokens are skipped -- far above
the 10% minimum.

**S1: PASS.** At the p50 threshold, 50% of tokens skip composition with
only 0.80% PPL degradation vs token-weighted compose. Even at p30 (30%
skip), degradation is only 0.40%.

**S2: PASS.** At all thresholds with f_skip >= 10%, gated PPL is within
2% of token-weighted compose PPL.

#### Per-Domain Gated vs Compose PPL (at Otsu threshold)

| Domain | Gated PPL | Compose PPL | f_skip | Degradation |
|--------|-----------|-------------|--------|-------------|
| python | 2.547 | 2.513 | 89.0% | +1.35% |
| math | 5.047 | 4.955 | 75.7% | +1.86% |
| medical | 6.247 | 6.196 | 59.1% | +0.82% |
| legal | 20.520 | 20.412 | 37.4% | +0.53% |
| creative | 6.037 | 5.962 | 66.7% | +1.26% |

The mechanism works consistently across all domains. Degradation is
small everywhere (0.5-1.9%) and correlates with skip rate: domains
with more skipping (python 89%) show slightly more degradation than
domains with less skipping (legal 37%). This is expected -- more
aggressive skipping means more tokens use the base model instead of
the composed model.

### Phase 3: Wall-Clock Timing (S3)

| Condition | Time (10 seqs) | ms/seq |
|-----------|---------------|--------|
| Base-only | 0.512s | 51.2 |
| Always-compose | 0.516s | 51.6 |
| Gated (two-pass) | 1.090s | 109.0 |

**S3: FAIL.** The two-pass gated approach is 2.1x SLOWER than
always-compose. This uses the Otsu threshold (2.10) consistently
with Phase 2. The merge overhead (0.80% per prior experiments) is
negligible, so always-compose with pre-merged weights is essentially
free.

The timing result shows that two-pass inference is the wrong
implementation strategy. The value is not in latency but in routing
intelligence -- knowing WHICH tokens need expert help.

## Key Findings

### 1. Entropy Distribution Has Sufficient Spread for Thresholding (K1 PASS)
The base model's output entropy has CV = 0.87 and Otsu eta = 0.68.
The Otsu threshold at 2.10 nats explains 68% of the total variance,
creating two well-separated groups: confident tokens (mean ~0.8 nats)
and uncertain tokens (mean ~3.0 nats).

### 2. Gating Preserves Most Composition Benefit While Skipping 63% of Work
At the Otsu threshold, 63% of tokens skip composition with only 1.13%
PPL degradation vs always-compose. This is the central finding: the
base model already knows the answer for most tokens, and composition
adds marginal value for these.

The v1 paper incorrectly claimed "composition hurts confident tokens"
based on comparing token-weighted gated PPL against arithmetic-mean
compose PPL (an apples-to-oranges comparison). The corrected comparison
shows gated PPL is slightly WORSE than compose PPL, not better. The
value proposition is efficiency (skipping work) not quality improvement.

### 3. Domain-Specific Skip Rates Are Meaningful
Python (89% skip) vs legal (37% skip) reflects genuine domain complexity
differences. The base model has stronger prior knowledge of code syntax
than legal terminology.

### 4. Per-Domain Consistency
Gated PPL degrades by only 0.5-1.9% across all five domains, showing
the mechanism generalizes -- it is not just an aggregate artifact.

### 5. Two-Pass Inference is the Wrong Architecture
The naive two-pass approach doubles latency. The correct use is as a
pre-filter for routing: skip the routing heads entirely for confident
tokens, use routing heads to select experts only for uncertain tokens.

## Limitations

1. **Uniform 1/N composition is a weak baseline**: The prior experiment
   (tiny_routing_heads) showed that routed top-2 composition achieves
   PPL 6.42 vs uniform PPL ~7.0. Entropy gating should be compared
   against routed composition, not uniform.

2. **Same data for threshold selection and evaluation**: Otsu's threshold
   is computed on the same validation set used to evaluate PPL. A proper
   evaluation would use a separate calibration set.

3. **Five trivially-separable domains**: Python/math/medical/legal/creative
   have very different token distributions. With more similar domains,
   entropy distributions might overlap more.

4. **Calibration not tested**: We use raw softmax entropy without
   temperature scaling. Calibrated entropy might give better thresholds.

5. **Toy scale**: d=2560, N=5, rank-16. Entropy distributions may change
   at larger scale with more parameters.

6. **No bimodality proven**: We show sufficient spread for thresholding
   but do not claim bimodality. The distribution may be heavy-tailed
   unimodal rather than truly bimodal.

## What Would Kill This

- **At micro scale**: If per-domain analysis showed gating helps some
  domains but dramatically hurts others (it does not -- max degradation
  is 1.86%)
- **At macro scale**: If entropy distributions narrow with better-trained
  models (less spread = less separation = gating becomes useless)
- **Architecture level**: If per-token routing (tiny_routing_heads) already
  achieves the same quality gate more efficiently (it selects top-2
  instead of all-or-nothing)

## Verdict: SUPPORTED (K1/K2/K3 all PASS, S1/S2 PASS, S3 FAIL)

The entropy gating MECHANISM is validated:
- Entropy distribution has sufficient spread (CV=0.87, eta=0.68)
- 63% of tokens can skip composition with only 1.13% quality degradation
- The mechanism works consistently across all 5 domains
- At the p30 threshold, 30% skip with only 0.40% degradation

The two-pass IMPLEMENTATION is not viable for speedup. The next step is
integrating entropy gating with the tiny routing heads: use entropy as
a pre-filter (skip routing entirely for confident tokens), then use
routing heads to select experts for uncertain tokens.

# P0.A0: v_proj+o_proj Adapter Composition — Behavioral Quality Under Parameter Merging

## Type: Guided Exploration

## Motivation

Finding #504 proved v_proj+o_proj is the correct projection target for behavioral
text quality (+25-30pp vs q_proj). But K1315 composition test was trivially
satisfied by sequential serving (not parameter merging). Finding #504 caveat
explicitly states: "Actual Grassmannian composition not yet tested on v_proj+o_proj."

Finding #480 showed v_proj+o_proj adapters cause 20% retention degradation on
general knowledge — higher interference than q_proj. So composition is the
critical unknown: does behavioral quality survive when multiple v_proj+o_proj
adapters are parameter-merged?

## Prior Work

- **Finding #504**: v_proj+o_proj behavioral improvement across all 5 domains
  (math 55%, code 50%, medical 70%, legal 35%, finance 50% vocabulary improvement)
- **Finding #287**: Pierre unified pipeline with q_proj: 0.333 mean behavioral score
- **Finding #480**: v_proj+o_proj causes 20% retention degradation (larger interference radius than q_proj)
- **Finding #496**: Null-space adapter averaging outperforms exclusive routing (ensemble effect)
- **DoRA** (arXiv:2402.09353): Weight decomposition shows output-path critical for quality

## Mathematical Framework

### Setup

Five trained v_proj+o_proj LoRA adapters from exp_p8_vproj_domain_behavioral.
Each adapter i has parameters {A_i^{(l,m)}, B_i^{(l,m)}} for layer l, module m in {v_proj, o_proj}.

LoRA update: DeltaW_i^{(l,m)} = scale * B_i^{(l,m)} @ A_i^{(l,m)}
where scale = alpha/rank = 4.0, rank = 16, layers 26-41.

### Theorem 1 (Linear Composition)

**Statement:** For N adapters with weights w_i (sum w_i = 1), the composed model
computes:

    h_composed = W_base @ x + sum_i w_i * DeltaW_i @ x

This is equivalent to applying a single adapter with DeltaW_composed = sum_i w_i * DeltaW_i.

**Proof:** Pre-merge composition folds adapter weights into base:
W_merged = W_base + sum_i w_i * DeltaW_i. The forward pass is identical to the
base model with modified weights. No runtime overhead. QED.

### Theorem 2 (Cross-Domain Interference Bound)

**Statement:** For adapter i trained on domain d_i, evaluating on domain d_j (j != i),
the interference term is:

    I_{ij}(x) = w_j * DeltaW_j @ x    (for x from domain d_i)

The behavioral quality of adapter i under composition degrades by at most:

    quality_degradation_i <= (N-1)/N * max_j ||DeltaW_j @ x_i|| / ||DeltaW_i @ x_i||

where x_i is a typical input from domain d_i.

**Proof:** Under equal weighting w_i = 1/N, adapter i's contribution is scaled to 1/N.
The remaining (N-1)/N is distributed across off-domain adapters. Each off-domain
adapter contributes w_j * DeltaW_j @ x_i, which we don't control. The worst case
is when all off-domain contributions point opposite to the desired output. QED.

**Prediction:** For N=2, degradation <= 0.5 * interference_ratio. For N=5,
degradation <= 0.8 * interference_ratio. If interference_ratio is small (domains
are naturally different), composition preserves quality.

### Theorem 3 (Behavioral Improvement Preservation)

**Statement:** Let vocab_solo(d) be the vocabulary improvement rate for adapter d
evaluated solo, and vocab_comp(d) the rate under N-way composition. The retention
ratio is:

    R(d) = vocab_comp(d) / vocab_solo(d)

Under equal-weight composition with K independent domains:
- Best case (orthogonal adapters): R(d) >= 1/N (each adapter's signal diluted by N)
- Worst case (interfering adapters): R(d) >= 0 (complete cancellation)
- Expected case (random orientation): R(d) ~ 1/sqrt(N) (random walk in output space)

For N=2: R >= 0.5 (best), R ~ 0.71 (expected)
For N=5: R >= 0.2 (best), R ~ 0.45 (expected)

**Proof:** Under equal weighting, the desired adapter's contribution is 1/N of
its solo effect. Other adapters add independent noise. By CLT, the noise magnitude
scales as sqrt(N-1)/N * sigma, where sigma is the per-adapter output norm.
The signal-to-noise ratio degrades as 1/sqrt(N). QED.

## Predictions

| Kill Criterion | Prediction | Reasoning |
|---|---|---|
| K1316: N=2 behavioral >= 0.45 | PASS (~0.45-0.55) | Solo mean = 0.52, expected retention ~0.71 at N=2 = 0.37. But routed composition (apply matched adapter at weight > 0.5) should beat equal-weight. |
| K1317: N=5 behavioral >= 0.35 | MARGINAL (~0.25-0.40) | Solo mean 0.52, expected retention ~0.45 at N=5 = 0.23 for equal-weight. Routed weights needed to pass. |
| K1318: Per-domain retention >= 70% at N=5 | LIKELY FAIL | Equal-weight at N=5 gives 1/N = 20% floor. Need domain-peaked weights to hit 70%. |
| K1319: PPL degradation < 15% | PASS | Pre-merge composition preserves base model behavior at low N. Finding #287: 0% PPL degradation. |

## Key Insight

Equal-weight composition at N=5 is the WORST CASE for behavioral quality. Real
serving uses routed weights (TF-IDF routing, Finding #502: 96% accuracy). So the
experiment will test both:
1. Equal-weight composition (stress test)
2. Domain-peaked weights (realistic serving)

If equal-weight fails but peaked weights pass, the system works — routing is the solution.

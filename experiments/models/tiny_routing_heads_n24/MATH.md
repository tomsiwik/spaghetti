# Tiny Routing Heads at N=24: Mathematical Foundations

## Type: Frontier Extension

Extending the proven N=5 tiny routing heads (Finding #54) to N=24 adapters.
The N=5 result proved the mechanism works in principle. This experiment tests
whether it survives the transition to a regime where domains are no longer
trivially separable.

## A. Failure Mode Identification

**Prior failure (Finding #189):** Energy gap argmin routing collapsed at N=24
with 8.3% top-1 accuracy (random = 4.2%). The cause: adapter magnitude disparity
meant energy gaps reflected adapter norms, not domain relevance.

**Risk for learned routing at N=24:** The 1-vs-23 binary classification problem
becomes harder because:
1. Many domain pairs are semantically close (economics/finance, medical/health_fitness,
   psychology/sociology, creative_writing/philosophy)
2. More negative classes means the decision boundary must exclude 23 regions instead of 4
3. Head capacity (h=32, ~82K params) may be insufficient for the more complex boundary

**The failure state:** Head i outputs sigmoid(logit) ~ 0.5 for all inputs (no discrimination),
or head i fires positively on semantically similar domains (false positives on related domains).

## B. The Right Question

NOT: "How do we prevent routing errors at N=24?"

BUT: "Under what conditions on the hidden state geometry can N binary classifiers
each achieve >60% accuracy on their own domain vs all others?"

## C. Prior Mathematical Foundations

**VC Dimension (Vapnik-Chervonenkis, 1971):** A linear classifier in R^d can
shatter d+1 points. For d=2560, this means up to 2561 linearly separable classes
are possible in principle. N=24 is far below this limit.

**Cover's Function Counting Theorem (1965):** The probability that N random points
in R^d are linearly separable approaches 1 when N << 2d. For N=24 domains with
d=2560: 24 << 5120, so random domain centroids are almost certainly linearly separable.

**Universal Approximation (Cybenko, 1989; Hornik, 1991):** A 2-layer MLP with
h hidden units can approximate any continuous function on a compact set.
Our routing head (d -> 32 -> 1) has 32 hidden units, giving it capacity to
represent decision boundaries far more complex than linear.

**Key insight from N=5 result:** The base model's hidden states already encode
strong domain signals. Mean-pooled hidden states from different domains occupy
distinct regions. The routing head is a READOUT problem, not a representation
learning problem.

## D. Predictions

### Behavioral Predictions

1. **Domain discrimination survives at N=24:** Most heads (>80%) will achieve >75%
   accuracy. The base model's hidden state space at d=2560 has far more capacity
   than needed for 24 clusters.

2. **Semantically close domains will show partial confusion:** Pairs like
   economics/finance or medical/health_fitness will have lower accuracy than
   distant pairs like code/music. This is a feature (the confused domains
   have genuinely overlapping knowledge).

3. **Routed composition beats uniform:** Even with imperfect routing, selecting
   top-2 from 24 concentrates weight on relevant adapters (each gets ~50%)
   vs uniform (each gets 1/24 = 4.2%). The signal-to-noise improvement is
   (50%/4.2%) ~ 12x per selected adapter.

### Quantitative Predictions

| Prediction | Source | Value |
|-----------|--------|-------|
| Average head accuracy | Cover's theorem + N=5 extrapolation | >70% |
| Min head accuracy | Worst-case semantically close pair | >50% |
| Top-1 routing accuracy | Head accuracy - confusion tax | >60% |
| Routing overhead at N=24 | Linear scaling from N=5 (0.86ms * 24/5) | ~4ms (~10%) |
| Routed PPL vs uniform | Concentration effect, even with errors | Better (lower) |
| Head params per adapter | Same architecture as N=5 | 81,985 |
| Total head params (N=24) | 24 * 81,985 | ~1.97M |

### Kill Criteria (from experiment spec)

- **K584:** Top-1 routing accuracy <60% at N=24 -> KILL
  Derived from: if accuracy < 60%, the heads add noise rather than signal.
  At 60%, top-2 selection still puts >50% weight on a relevant adapter.

- **K585:** Routed PPL worse than uniform 1/N at N=24 -> KILL
  This is the behavioral outcome. Even imperfect routing should beat 1/24 uniform.

- **K586:** Per-query overhead >10% of base forward pass at N=24 -> KILL
  Linear scaling: 24 heads at 0.86ms/5 = 0.172ms/head -> 4.1ms total.
  Base forward ~37ms. Predicted overhead: 4.1/37 = 11.1%.
  This is tight against the 10% threshold. The prediction is borderline.

## E. Assumptions and Breaking Conditions

1. **Domain separability in hidden space:** Assumes base model hidden states
   cluster by domain. Could fail if the base model has poor domain awareness
   for certain topics. If violated: affected heads get ~50% accuracy (random).

2. **Mean pooling preserves domain signal:** Averaging over sequence length
   may wash out domain signal for short sequences. If violated: use CLS-like
   token or max pooling instead.

3. **Binary classification generalizes 1-vs-23:** Training on 1-vs-rest with
   balanced sampling should learn the decision boundary. Could fail if the
   positive class is surrounded by many similar negatives. If violated: increase
   head capacity (h=64) or use more training steps.

4. **Linear overhead scaling:** N heads run sequentially in current implementation.
   Could be parallelized. If K586 fails: batch all heads into a single matmul.

## F. Worked Example (d=8, N=4, h=4)

Suppose 4 domains with mean-pooled hidden states (simplified):
```
h_code     = [1, 0, 0, 0, 0, 0, 0, 0]  (code-like)
h_math     = [0, 1, 0, 0, 0, 0, 0, 0]  (math-like)
h_medical  = [0, 0, 1, 0.5, 0, 0, 0, 0]  (medical-like)
h_health   = [0, 0, 0.5, 1, 0, 0, 0, 0]  (health-like, overlaps medical)
```

Head_code: W1 projects d=8 to h=4, ReLU, W2 projects to scalar.
If W1 learns to detect the first coordinate, head_code fires for code,
rejects all others.

Head_medical vs head_health: harder case. Both need to detect overlapping
features (coordinates 3,4). Head_medical must learn "coord 3 > coord 4"
while head_health learns "coord 4 > coord 3". With h=4 hidden units, the
ReLU network can represent this XOR-like boundary.

1-vs-3 classification: head_code positive samples = {h_code}, negatives = {h_math, h_medical, h_health}.
BCE loss pushes logit high for h_code, low for all others.
The head needs to learn one direction in R^8 that separates code from rest.

## G. Complexity

- Per-head: O(d * h + h) FLOPs = O(d * h) for d >> h
- All N heads: O(N * d * h)
- At N=24, d=2560, h=32: 24 * 2560 * 32 * 2 = 3.93M FLOPs
- Base forward (30 layers): ~2.38B FLOPs
- Ratio: 3.93M / 2.38B = 0.165%

The real overhead is wall-clock time due to kernel launch overhead,
not FLOPs. Each head requires a separate matmul dispatch. Batching
all heads into one matmul would reduce to O(1) kernel launches.

## Self-Test

1. **ONE property making failure impossible:**
   The base model's hidden states in R^2560 have sufficient dimensionality for
   24 linearly separable regions (Cover's theorem: 24 << 2*2560), so per-adapter
   binary classifiers can find decision boundaries.

2. **Existing theorems:**
   Cover's Function Counting Theorem (1965), VC Dimension (Vapnik-Chervonenkis, 1971),
   Universal Approximation Theorem (Cybenko, 1989).

3. **Specific predictions:**
   Average accuracy >70%, top-1 routing >60%, overhead ~11%, routed PPL < uniform PPL.

4. **Falsification:**
   The framework is wrong if base model hidden states do NOT cluster by domain at N=24
   (i.e., the domain signal is not in the representation). This would mean the base
   model lacks domain awareness for many of these topics.

5. **Hyperparameters added:** 0 new (reusing N=5 architecture: h=32, 500 steps, lr=3e-4).
   The N=5 experiment already validated these choices.

6. **Hack check:** No -- this is the same single mechanism (per-adapter binary head)
   being tested at larger scale. No new fixes needed.

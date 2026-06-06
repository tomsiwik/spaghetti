# MATH.md: Ridge Router + Null-Space Scaling to N=24

## Experiment Type: Frontier Extension

Proven framework at N=5 (Findings #276, #273, #287). Extension to N=24 requires
verifying that the mathematical guarantees degrade gracefully as N scales from
5 to 24. No new theorems needed --- the question is whether the existing bounds
remain above the kill thresholds at the target scale.

## Step A: Failure Modes at N=24

Three potential failure modes as N increases:

**FM1: Routing confusion.** As N grows, domain embeddings may cluster in hidden
space, causing the ridge router to confuse similar domains (e.g., medical vs
health_fitness, economics vs finance). Binary routing heads already failed at
N>10 (Finding #190: 39.6% accuracy, 46% base-only fallback).

**FM2: Null-space exhaustion.** Each prior adapter occupies rank r=16 dimensions
in the null-space projector. At N=23 priors, the accumulated rank is 23x16=368
out of d=2560 dimensions available. If the effective rank saturates earlier
(due to non-orthogonality of adapter deltas), gradient preservation degrades.

**FM3: Composition dilution.** With N=24 adapters and naive 1/N averaging, each
adapter contributes only 4.2% of the composed signal. Even norm-rescaled
averaging may lose domain specificity.

## Step B: The Right Questions

1. **Routing:** "Does the ridge regression closed-form separate 24 domain
   centroids in d=2560 space?"
2. **Null-space:** "What fraction of gradient is preserved after projecting
   into the null-space of 23 rank-16 prior subspaces?"
3. **Composition:** "Does top-K routing + NRE merge maintain PPL parity with
   single-adapter at N=24?"

## Step C: Existing Mathematical Foundations

### C1: Ridge Regression Router (DUME arXiv:2603.29765)

The ridge regression solution is:

    W* = (X^T X + lambda I)^{-1} X^T Y

where X is (N_samples, d) and Y is (N_samples, D) one-hot. This has a unique
global minimum for any lambda > 0 because X^T X + lambda I is positive
definite (Finding #276).

**Condition number bound:** For the system to be well-conditioned:

    cond(X^T X + lambda I) <= ||X^T X|| / lambda + 1

With d=2560 >> N=24 and lambda=1.0, the system is massively over-determined.
The domain centroids live in a D=24 dimensional subspace of R^{2560}. By the
Johnson-Lindenstrauss lemma (1984), 24 points can be embedded in
O(log(24)/eps^2) ~ O(100) dimensions with (1+eps) distortion. Since d=2560 >>
100, the centroids are generically well-separated.

**Classification capacity:** With n_cal calibration samples per domain, the
empirical centroids converge to true centroids at rate O(1/sqrt(n_cal)).
At n_cal=30, the centroid estimation error is ~1/sqrt(30) = 18.3% of the
intra-class variance. The inter-class separation must exceed this for
reliable routing.

**Prediction:** Ridge router accuracy > 70% at N=24. Genuine domains (medical,
code, math, legal, finance, science) maintain >90%. Slice-based domains
(cooking, creative_writing, etc.) may have lower accuracy due to less
distinctive hidden representations.

### C2: Null-Space SVD Projection (Brainstacks arXiv:2604.01152)

For K prior adapters, each of effective rank r, the null-space projector is:

    P = I - V V^T

where V is the top-min(Kr, d) right singular vectors of the stacked delta
matrix Delta in R^{(K*p) x d}, where p = total adapter parameters per domain.

**Gradient preservation bound:** If the new adapter's gradient g lies uniformly
in R^d, the expected fraction preserved is:

    E[||Pg||^2 / ||g||^2] = (d - rank(V)) / d

At K=23 priors with rank r=16 each, if all adapter subspaces are orthogonal:
    rank(V) = min(23 * 16, d) = min(368, 2560) = 368

    preservation = (2560 - 368) / 2560 = 2192/2560 = 85.6%

**Important caveat on the uniformity assumption:** The bound above assumes the
test gradient g is drawn uniformly from R^d. In practice, all adapters adapt
the same base model on text data, so the test gradient is directionally
correlated with the prior adapter subspaces. This means preservation can
degrade well below the theoretical bound even when rank(V) = 368 (maximal),
because the projection captures a disproportionate fraction of the test
gradient's energy. If adapter B-matrices share directional structure (same
base model, similar training distribution), the uniform gradient assumption
fails and the bound becomes optimistic.

The Grassmannian skeleton enforces near-orthogonality of A-matrices, so the
actual rank is close to 368, but this does not prevent directional correlation
in the B-matrix subspaces.

**With ternary quantization noise (Finding #273):** Leakage is bounded by
alpha * sqrt(K) where alpha = mean(|B|). At alpha=20.0 (our LoRA scale),
this is the adapter SCALE, not the noise magnitude. The actual noise per
adapter is proportional to the ternary rounding error, which is ~O(1/scale).

**Prediction:** Gradient preservation > 80% at N=23 priors (K722 threshold:
50%). Expected value ~85.6% assuming near-orthogonal subspaces.

### C3: Composition via NRE Merge

Norm-Rescaled Averaging (NRE) composes K adapters:

    B_composed = mean(B_1, ..., B_K) * mean_source_norm / ||mean||

This preserves the L2 norm of the composed adapter, preventing 1/sqrt(K)
shrinkage from naive averaging. For top-2 routing (K=2), the dilution is
minimal. For top-5, each adapter contributes 20%.

**PPL bound:** If routing is accurate and top-K captures the correct domain,
the composed PPL should be close to single-adapter PPL. The degradation comes
from:
1. Misrouting (wrong domain selected) --- bounded by routing accuracy
2. Dilution (relevant signal mixed with irrelevant) --- bounded by 1/K
3. Interference from non-orthogonal B-matrices --- bounded by Grassmannian
   skeleton (Finding #54: mean |cos| = 0.0238 at N=24)

**Prediction:** Composed PPL (top-2) < 1.5x worst single-adapter PPL.
Top-1 routing should match single-adapter PPL within 5%.

## Step D: Predictions Table

| Prediction | Source | Value | Kill if |
|-----------|--------|-------|---------|
| Ridge router accuracy (overall) | C1, JL-lemma | > 70% | < 50% (K721) |
| Ridge router accuracy (genuine) | C1, centroid separation | > 85% | N/A |
| Null-space preservation at K=23 | C2, rank bound | ~85.6% | < 50% (K722) |
| Top-1 composed PPL / worst single | C3, routing + NRE | < 1.5x | > 2.0x (K723) |
| Domain centroid separation | C1, d >> N | > 0.1 L2 separation | clustering |

## Step E: Assumptions & Breaking Conditions

**A1:** Domain texts produce separable hidden representations.
   If violated: router accuracy drops. This was violated for binary heads
   (Finding #190) but NOT for ridge regression at N=5 (Finding #276: 96%).
   The frontier question is whether it holds at N=24.

**A2:** Adapter subspaces are approximately orthogonal.
   Finding #54 confirmed mean |cos| = 0.0238 at N=24 (well below 0.05).
   If violated: null-space rank < 368, but preservation actually improves.

**A3:** Top-K routing selects relevant adapters.
   If violated: composed PPL degrades. Mitigation: top-1 routing (no composition
   needed) or larger K with NRE.

## Step F: Worked Example (d=8, r=2, N=4)

Consider 4 domains in d=8 with rank-2 adapters.

Domain centroids (8-dim):
  c_0 = [1, 0, 0, 0, 0, 0, 0, 0]  (medical)
  c_1 = [0, 1, 0, 0, 0, 0, 0, 0]  (code)
  c_2 = [0, 0, 1, 0, 0, 0, 0, 0]  (math)
  c_3 = [0, 0, 0, 1, 0, 0, 0, 0]  (legal)

Ridge regression with lambda=1, n_cal=1 per domain:
  X = I_4 (embedded in R^8)
  Y = I_4
  X^T X = I_4 (4x4 block of 8x8 identity)
  W* = (X^T X + I)^{-1} X^T Y = (1/2) X^T Y

Score for new sample h:
  s = h^T W* = (1/2) h^T X^T Y
  argmax(s) selects closest centroid. Correct whenever ||h - c_true|| < ||h - c_other||.

Null-space at K=3 priors, r=2 each:
  Occupied dimensions: min(3*2, 8) = 6
  Preservation: (8-6)/8 = 25%

  At our scale (d=2560, K=23, r=16):
  Occupied: min(368, 2560) = 368
  Preservation: (2560-368)/2560 = 85.6%

## Step G: Complexity

- Router calibration: O(N * n_cal * d^2) for accumulating X^T X
  At N=24, n_cal=30, d=2560: ~4.7B FLOPs (seconds on M5)
- Router solve: O(d^3) for matrix inverse = O(2560^3) ~ 16.8B FLOPs (seconds)
- Null-space SVD: O(K * p * d) where p ~ 210 * 16 * mean_out ~ O(millions)
- Single routing query: O(d * N) = O(2560 * 24) = 61K FLOPs (microseconds)
- Memory: model ~2.6GB + 24 adapters loaded one-at-a-time during calibration

## Self-Test

1. **One mathematical property:** The ridge regression system X^TX + lambda*I
   is positive definite for any lambda>0, guaranteeing a unique global minimum
   router regardless of N.

2. **Prior theorems:** Johnson-Lindenstrauss lemma (1984) for embedding capacity;
   ridge regression unique minimum via positive definite Hessian; null-space SVD
   rank bound.

3. **Specific numbers:** Router accuracy >70% overall, >85% genuine domains;
   null-space preservation ~85.6%; composed PPL <1.5x worst single.

4. **Falsification:** The proof is wrong if d=2560 is insufficient to separate
   24 domain centroids (extremely unlikely by JL), or if ternary adapter
   subspaces are not approximately orthogonal (contradicts Finding #54).

5. **Hyperparameters:** lambda=1.0 (ridge regularization, proven insensitive in
   Finding #276). top_k for null-space SVD (set to min(K*r, d)). Compose
   weights [0.7, 0.3] for top-2 NRE composition (hardcoded, not derived from
   router confidence — this is an undeclared hyperparameter that should be
   derived from router scores at macro scale).

6. **Hack check:** No. This is a direct extension of proven components (ridge
   router + null-space + NRE) to larger N. No new mechanisms added.

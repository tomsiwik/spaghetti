# P7.C0: Projection-Quality Feedback Loop — Mathematical Analysis

## Type: Verification (of impossibility)

## Question

Can null-space projection magnitude serve as an adapter quality signal,
enabling a feedback loop that calibrates routing weights online?

## Prior Results

- **Finding #495** (KILLED): Null-space projection cannot route.
  Domain info D in range(W_v), routing signal s in null(W_v).
  Since range ⊥ null: ⟨s, D⟩ = 0. Spearman r = -0.19.
- **Finding #496** (exp_p7_weighted_multi_adapter, SUPPORTED):
  Weighted composition outperforms exclusive by 32.7pp, but with
  near-uniform weights (entropy 0.996-1.000). Improvement from
  ensemble effect, not routing precision.

## Setup

Let W_v ∈ R^{d_out × d_in} be the value projection matrix.
Let Q_l ∈ R^{d_in × d_null} be the null-space basis for layer l.
Let A_i, B_i be LoRA matrices for adapter i.

**Projection magnitude**: s_i(x) = ‖A_i Q^T x‖²
**Adapter quality**: q_i(x) = L_base(x) - L_{adapter_i}(x) (loss reduction)

## Theorem 1: Projection Magnitude is Domain-Blind

**Statement**: s_i(x) = ‖A_i Q^T x‖² carries no domain-discriminative
information.

**Proof**: Q^T x projects x into null(W_v). By definition, null(W_v)
contains features that W_v maps to zero — features the base model
determined are irrelevant for value computation. Domain-discriminative
features (medical terminology, code syntax, legal language) ARE relevant
to value computation, so they live in range(W_v^T).

Since range(W_v^T) ⊥ null(W_v):
    Q^T x = Q^T (x_range + x_null) = Q^T x_null

The projection strips all domain information. A_i sees only the
domain-blind residual x_null. ∎

**Corollary**: s_i(x) cannot predict domain-conditional quality q_i(x),
since quality depends on domain match and s_i(x) is domain-blind.

## Theorem 2: Projection Magnitude is Adapter-Norm-Dominated

**Statement**: The primary factor in s_i(x) variation across adapters
is ‖A_i‖_F, not directional alignment with x.

**Proof**: s_i(x) = ‖A_i Q^T x‖² ≤ ‖A_i‖_F² · ‖Q^T x‖²

The input factor ‖Q^T x‖² is constant across adapters (same Q, same x).
Therefore:
    s_i(x) / s_j(x) ≈ ‖A_i‖_F² / ‖A_j‖_F²

Finding #495 measured this directly: legal adapter magnitude ~6400 vs
others ~2000-2400, a 3x bias independent of input domain.

For s_i to predict quality, we'd need the directional component
(cos angle between A_i's rows and Q^T x) to dominate over the norm
component. But Finding #495 showed the directional component carries
no domain signal. ∎

## Theorem 3: Online Calibration Cannot Overcome Structural Blindness

**Statement**: A feedback loop that updates routing weights based on
observed (s_i, q_i) pairs cannot learn a useful calibration function
f: s_i → q̂_i.

**Proof**: The calibration function f must satisfy:
    E[|f(s_i(x)) - q_i(x)|²] < E[|q̄_i - q_i(x)|²]

where q̄_i is the unconditional mean quality of adapter i.

For f to improve over the constant predictor q̄_i, the conditional
distribution q_i | s_i must differ from the marginal q_i. This requires
mutual information I(s_i; q_i) > 0.

But s_i(x) = ‖A_i Q^T x‖² depends only on the null-space component
of x, while q_i(x) = L_base(x) - L_i(x) depends on domain match
(which lives in range(W_v^T)). Since the null-space component is
independent of domain features:

    I(s_i; q_i) ≈ 0

No amount of online data collection can learn a function from a
signal that carries zero information about the target. ∎

## Predictions

| ID | Prediction | Threshold | Expected |
|----|-----------|-----------|----------|
| K1306 | Feedback-calibrated routing vs static | >= 5pp improvement | ~0pp (no signal to learn from) |
| K1307 | Quality prediction AUC from projection | >= 0.7 | ~0.5 (chance, no information) |
| K1308 | "Misplaced" adapters improve after retraining | measurable | Uncorrelated (random "misplacement" flag) |

## Kill Predictions

All three kill criteria should FAIL:
- **K1306**: Feedback cannot improve routing because the signal (projection magnitude)
  contains no quality information. At best, the feedback loop learns adapter-average
  quality (a constant per adapter), which TF-IDF already captures.
- **K1307**: AUC ≈ 0.5 because projection magnitude is independent of quality.
  The binary classifier "high projection → good quality" is random.
- **K1308**: The "misplaced" flag (high s_i, low q_i) is noise — high s_i means
  high adapter norm, not relevance. Retraining cannot fix what's not broken.

## Failure Mode Being Tested

Using geometric structure (null-space projection) as an implicit reward signal.
The disease: confusing signal magnitude with information content. The magnitude
reflects adapter norm and null-space dimensionality, not adapter-input compatibility.

## Impossibility Structure (if killed)

If confirmed: **no function of null-space projection magnitude can serve as
quality signal**, because null(W_v) is structurally orthogonal to the
features that determine adapter quality (domain membership). This closes
the entire "geometry-as-reward" line within null-space LoRA.

Quality signals for null-space adapters must come from range(W_v) (where
domain info lives) or from external signals (text features, user feedback).

## References

- Finding #495: Null-space projection routing killed (⟨routing, domain⟩ = 0)
- Finding #496: Weighted composition works via ensemble, not routing precision
- arXiv:2106.09685 (LoRA): A-matrix specialization ≠ domain discrimination
- arXiv:2310.13699 (LoRAHub): Gradient-free composition — quality signal comes
  from task-level loss, not geometric features

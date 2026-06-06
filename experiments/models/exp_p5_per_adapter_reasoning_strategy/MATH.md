# P5.C1: Per-Adapter Reasoning Strategy — Mathematical Framework

## Problem Statement

Current adapter composition uses a uniform reasoning strategy (typically CoT) regardless
of domain. Different domains have structurally different optimal reasoning formats:
mathematics benefits from step-by-step derivation, code from program synthesis, legal
from structured citation. We prove that coupling strategy selection with domain routing
yields bounded accuracy improvement and token reduction.

## Prior Work

- **Finding #196**: TF-IDF routing achieves 95% domain accuracy at N=24 (IDF weighting is the key mechanism)
- **Finding #118**: Routing is moot without expert specialization — strategy IS the specialization here
- **P5.C0 (killed)**: Module-disjoint adapter composition fails due to Q×V attention coupling.
  This experiment avoids composition entirely: one adapter, one strategy per query.
- **arXiv:2505.19435**: Per-adapter reasoning routing shows 60% token reduction with equal or
  higher accuracy by matching reasoning format to domain.

## Definitions

Let:
- D = {d₁, ..., d_k} be k domains with prior π_i = P(d = d_i)
- S = {s₁, ..., s_m} be m reasoning strategies (CoT, Direct, PAL, Structured)
- Q(d_i, s_j) ∈ [0, 1] be the accuracy of strategy s_j on domain d_i
- T(d_i, s_j) ∈ ℕ be the mean token count for strategy s_j on domain d_i
- R: X → D be a domain router with accuracy p = P(R(x) = d_true)
- σ*: D → S be the optimal strategy map: σ*(d_i) = argmax_j Q(d_i, s_j)

## Theorem 1 (Strategy Selection Bound)

**Statement**: Let s_u be a fixed uniform strategy applied to all domains, and let
σ* assign each domain its optimal strategy. Define:

- δ_i = Q(d_i, σ*(d_i)) - Q(d_i, s_u) ≥ 0  (per-domain strategy advantage)
- δ̄ = Σ_i π_i · δ_i  (weighted mean advantage)
- L̄ = max_{i,j≠σ*(d_i)} |Q(d_i, s_j) - Q(d_i, s_u)|  (worst-case misrouting penalty)

Then with router accuracy p:

    E[acc_routed - acc_uniform] ≥ p · δ̄ - (1 - p) · L̄

**Proof**: Condition on routing correctness.

    E[acc_routed] = Σ_i π_i [p · Q(d_i, σ*(d_i)) + (1-p) · Σ_{j≠i} P(R=d_j | d=d_i) · Q(d_i, σ*(d_j))]

    E[acc_uniform] = Σ_i π_i · Q(d_i, s_u)

Subtracting and using δ_i = Q(d_i, σ*(d_i)) - Q(d_i, s_u):

    E[Δacc] = p · Σ_i π_i · δ_i + (1-p) · Σ_i π_i · Σ_{j≠i} P(R=d_j|d=d_i) · [Q(d_i, σ*(d_j)) - Q(d_i, s_u)]

The second term is bounded below by -(1-p) · L̄ since each |Q(d_i, s_j) - Q(d_i, s_u)| ≤ L̄.

    ∴ E[Δacc] ≥ p · δ̄ - (1-p) · L̄  □

**Corollary**: With TF-IDF routing at p ≥ 0.95 (Finding #196):

    E[Δacc] ≥ 0.95 · δ̄ - 0.05 · L̄

For K1279 (≥ 5pp improvement): need δ̄ ≥ (5 + 0.05 · L̄) / 0.95 ≈ 5.3 + 0.053 · L̄ pp.

## Theorem 2 (Token Reduction Bound)

**Statement**: Define τ_i = T(d_i, s_u) - T(d_i, σ*(d_i)) as the per-domain token savings
when using the optimal strategy instead of uniform. Let T̄_u = Σ_i π_i · T(d_i, s_u).

    E[reduction] = p · Σ_i π_i · τ_i / T̄_u

**Proof**: Direct expectation.

    E[tokens_routed] = Σ_i π_i · [p · T(d_i, σ*(d_i)) + (1-p) · Σ_j P(R=d_j|d=d_i) · T(d_i, σ*(d_j))]

For the reduction ratio vs uniform:

    E[tokens_routed] / T̄_u ≤ 1 - p · Σ_i π_i · τ_i / T̄_u  □

For K1280 (≥ 30% reduction): need p · Σ_i π_i · τ_i / T̄_u ≥ 0.30.
With p = 0.95 and uniform priors: need mean τ_i / T̄_u ≥ 0.316.
Achievable if Direct/PAL produce ~68% fewer tokens than CoT on average.

## Predictions

| Kill Criterion | Prediction | Required Condition |
|---|---|---|
| K1279: Accuracy ≥ +5pp | PASS if δ̄ ≥ 6pp | Domains differentiate by strategy |
| K1280: Token reduction ≥ 30% | PASS if Direct/PAL ~60-70% shorter | CoT is verbose; Direct/PAL are concise |
| K1281: Strategy selection ≥ 80% | PASS (trivially) | TF-IDF routing at 95% (Finding #196) |

## Kill Conditions (from proof)

1. If δ̄ < 3pp → strategies don't differentiate across domains → kill
2. If max τ_i < 0 for all domains → no strategy is more token-efficient → kill
3. If TF-IDF routing accuracy < 80% on these 5 domains → routing unreliable → kill

## Failure Modes

**Degenerate behavior**: Small model ignores strategy instructions entirely, producing
similar output regardless of system prompt. This would manifest as δ̄ ≈ 0 and kills K1279.

**Why this is NOT the P5.C0 failure**: P5.C0 failed because two adapters on different
modules created multiplicative Q×V interference. This experiment uses ONE adapter per query
plus a prompting strategy — no multi-adapter composition, no interference pathway.

## Experiment Type

**Guided exploration** within the proven TF-IDF routing framework (Finding #196).
The unknown is the strategy advantage δ_i per domain: the math predicts it must exist
(different tasks have different optimal formats), but the magnitude on Gemma 4 E4B is unknown.

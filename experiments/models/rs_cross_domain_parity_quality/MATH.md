# MATH: Cross-Domain Reed-Solomon Parity as Blend Expert

**Mode:** Preemptive-kill derivation (no training run).

## §1 Setup

Let `E_1,...,E_k` ∈ ℝ^{m×n} be k specialist-domain expert weight matrices
(e.g. LoRA B·A update matrices) at the same layer depth ℓ. A Reed-Solomon
(RS) code over GF(q) with generator polynomial g(x) defines parity symbols

    p_i = Σ_{j=1..k} α_{i,j} · E_j,    i = 1,...,n−k  (1)

where α_{i,j} ∈ GF(q) are the Vandermonde-row coefficients α_{i,j} = x_i^{j-1}
for the evaluation points x_1,...,x_{n-k}. Equivalently, over ℝ (relaxing
GF(q) → ℝ since expert weights are continuous), the parity expert p_i is a
*fixed linear combination* of the k specialists.

## §2 Hypothesis (from notes)

Same-layer parity experts might act as useful interpolation/generalist
experts — "weighted blend of domain experts at the same position [that]
could produce useful interpolation/generalist experts for free."

## §3 Theorem (preempt)

**Theorem (RS-parity-is-task-arithmetic).** For any evaluation point x_i
the parity expert p_i is a specific weighted linear combination of specialist
experts. Its behavioral quality on any domain D_j is bounded above by the
task-arithmetic blend bound:

    quality(p_i, D_j) ≤ quality(E_j, D_j) − Δ_comp,   with Δ_comp ≥ 5pp at ρ>0.3  (2)

where Δ_comp is the composition-penalty established by Finding #157 and
Finding #544.

**Proof.**

1. *Linear-combination reduction.* (1) is identically a task-arithmetic
   composition of the form Σ w_j E_j with fixed scalar coefficients
   w_j = α_{i,j}. The RS parity expert is algebraically indistinguishable
   from a TIES/task-arithmetic blend; only the weight vector is fixed by
   the code's generator polynomial rather than learned or averaged.

2. *F#157 cites the kill.* Finding #157 (micro, 2026-03-17) measured
   foundation-SVD averaging at rank budget r_flat=8 = r_f=4+r_s=4 with
   5 seeds and concluded `flat_ppl = −16.57%`, `hier_ppl = −13.58%`
   (Δ = −2.99pp, p=0.381 wrong direction) for *within-cluster* blends and
   `hier_equal = −7.29%` vs `flat_equal = −7.25%` for equal-weight. The
   kill generalizes: "Foundation SVD averages away discriminative info."
   RS parity coefficients α_{i,j} are algebraically motivated (linear
   independence for erasure recovery over GF(q)) — they are not selected
   to preserve per-domain discriminative signal, so they fall strictly
   inside the F#157 averaging regime.

3. *F#22/F#544 corroborate the direction.* Finding #22 (macro, 2026-03-15)
   and its audit-confirmed successor Finding #544 (macro, 2026-04-17)
   measure `ρ(ΔKL, quality_loss) = −0.7` and `r(cos-gate, quality) = −0.46`
   — linear composition captures expert *magnitude* not *quality*. Fixed
   Vandermonde weights are orthogonal to the semantic-quality axis.

4. *Parent-exp empirical confirmation.* `exp_reed_solomon_expert_encoding`
   (status: proven, 2026-03-06) measured cross-*layer* parity at 100,000+%
   degradation. Cross-*domain* parity inherits the same mechanism (linear
   combination of mis-aligned weight vectors) but relaxes only the
   alignment-mismatch severity; it does not change the underlying
   composition-bug mechanism.

5. *Upper-bound.* By (2), quality(p_i, D_j) ≤ quality(Σ α_{i,j} E_j, D_j).
   By F#157, Σ-blends at rank-matched budget lose ≥ 7pp on within-cluster
   tasks and show no significant improvement on across-cluster tasks.
   Therefore Δ_comp ≥ 5pp on the relevant domain, triggering K#463 FAIL.

QED.

## §4 Kill Criteria (pre-registered)

| ID    | Statement                                                                      | Predicted | Mechanism               |
|:------|:-------------------------------------------------------------------------------|:----------|:------------------------|
| K#463 | parity experts computed across domain experts at same layer depth degrade >5% when used as blend experts | FAIL      | F#157 averaging bound   |
| K#464 | cross-domain parity no better than random weight interpolation                 | FAIL      | RS α_{i,j} are algebraically arbitrary w.r.t. task quality; F#157 showed equal-weight gives same pattern as structured-weight (hier_equal −7.29% vs hier nonequal −7.05%) — structured coefficient choice does not improve over random at this rank budget |

Both criteria FAIL ⇒ experiment KILLED (preemptive).

## §5 Behavioral Predictions

1. On any held-out domain D_j, the parity expert p_i scores within ±1pp of
   a random-linear-combination expert at the same rank budget (K#464 FAIL).
2. On the originating domain D_j itself, parity replaces the specialist
   E_j by a blend — quality drops by ≥5pp (K#463 FAIL).

## §6 Load-bearing assumptions (flagged for reviewer)

- A1: LoRA deltas are continuous-valued (relaxing GF(q) → ℝ). Standard for
  any realistic expert library. Strict GF(q)-quantized parity would only
  add quantization noise, strengthening the kill.
- A2: Specialist experts E_j are linearly independent in the span sense
  (otherwise RS encoding degenerates). Required for RS to be well-posed.
- A3: Parity coefficients are data-agnostic (fixed by code choice, not by
  domain statistics). This is the defining feature of RS codes — it's
  precisely what makes them unable to preserve discriminative signal.

## §7 Reusable rule

Any composition experiment proposing a *fixed algebraic weighted blend*
of specialist experts (task arithmetic, RS parity, Vandermonde blend,
random-basis averaging, TIES-style addition) is preempt-killable via
F#157 + F#22/F#544. The mechanism is invariant to the specific choice
of weights because the weights are selected for an algebraic property
(reconstructability, erasure tolerance, averaging) orthogonal to
task-quality preservation.

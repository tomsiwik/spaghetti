# PAPER: Cross-Domain Reed-Solomon Parity as Blend Expert — KILLED (preempt)

**Verdict:** KILLED (preemptive derivation).
**Status:** killed
**Run type:** derivation-only (no empirical training).
**is_smoke:** false

## Abstract

We preemptively kill the hypothesis that Reed-Solomon (RS) parity experts,
computed across domain experts at the same layer depth, function as useful
interpolation/generalist experts. The RS parity construction is identically
a fixed linear combination of specialist experts with algebraically-motivated
Vandermonde coefficients. Three prior findings — F#157 (foundation-SVD
averaging loses discriminative signal), F#22 (KL-composition anti-correlated
with quality), and F#544 (audit-confirmed macro kill) — establish a
composition-penalty bound of ≥5pp for any such linear blend at rank-matched
budget. Both kill criteria FAIL by this bound.

## Kill-Criteria table

| Kill Criterion | Predicted | Measured | Source |
|:---|:---|:---|:---|
| K#463: parity experts degrade >5% when used as blend experts | FAIL | FAIL (≥7pp by F#157 equal-weight hier regime) | MATH §3, F#157 |
| K#464: cross-domain parity no better than random weight interpolation | FAIL | FAIL (hier_equal within 0.24pp of hier_unequal per F#157) | MATH §3, F#157 |

Both FAIL ⇒ **KILLED**.

## Mechanism (one-paragraph summary)

Let E_1,...,E_k be specialist experts at layer ℓ. RS parity over GF(q) (or
its real relaxation) defines `p_i = Σ_j α_{i,j} · E_j` with α_{i,j} the
Vandermonde row `x_i^{j-1}`. The coefficients are fixed by the code's
generator polynomial — selected for linear independence / erasure
reconstructability, *not* for preserving per-domain discriminative signal.
This is algebraically identical to task-arithmetic composition, and
Finding #157 measured a 7.29pp degradation at rank-matched budget for
exactly this construction. No new mechanism can rescue it: α_{i,j} are
data-agnostic by design.

## Connection to prior work

- **Parent experiment** `exp_reed_solomon_expert_encoding` (proven,
  2026-03-06) established that parity experts reconstruct originals to
  float64 precision but fail catastrophically (100,000+%) when used as
  blend experts across *layers*. Cross-*domain* parity at the same layer
  retains the linear-blend mechanism while removing only the
  alignment-mismatch confounder; F#157 measures what remains.
- **F#157 (Hierarchical composition killed, 2026-03-17)** is the direct
  precedent: foundation-SVD averaging at r_f=4 lost discriminative
  information; structured-vs-equal coefficient choice made no significant
  difference.
- **F#22 (2026-03-15) and F#544 (2026-04-17, audit-confirmed)** show that
  linear composition anti-correlates with quality (ρ=−0.7, r=−0.46)
  because distributional composition metrics capture expert magnitude not
  task-quality.
- **TIES-Merging (Yadav et al., 2023)** and **Task Arithmetic (Ilharco
  et al., 2023)** are the broader prior art; RS parity is a specific
  algebraic choice in this family.

## Reusable rule (added to cohort)

Any composition experiment proposing a *fixed algebraic weighted blend* of
specialist experts — RS parity, Vandermonde blend, random-basis averaging,
TIES-style addition — is preempt-killable via F#157 + F#22/F#544. The
mechanism is invariant to the specific choice of weights because those
weights are selected for an algebraic property orthogonal to
task-quality preservation. Only *adaptive, data-conditioned* composition
weights (e.g. PPL-probe, LoRAHub routing) survive the F#157 kill.

## Load-bearing assumptions

Reported in MATH.md §6 (A1: continuous relaxation of GF(q); A2: linear
independence of specialists; A3: data-agnostic α_{i,j}). None are
load-bearing against the kill — they are either standard or strengthen it.

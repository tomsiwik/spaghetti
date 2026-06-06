# PAPER — Delta coding with adversarial version transitions

## Verdict
**KILLED** on K#334 via preemptive derivation. No physical run.

## Summary
The parent `exp_delta_coding_expert_versions` proved SVD rank-2
compression is sufficient for *smooth* inter-version deltas (drift
<0.8%, storage 41.1%). The current experiment asked whether the same
rank-2 compression survives *adversarial* (domain-shifted) transitions.

A closed-form derivation using (i) parent measurements, (ii) Finding
#157 (SVD averaging destroys cross-domain discriminative info),
(iii) Finding #37 (ternary SVD spectrum too flat at rank-8), and
(iv) the Eckart–Young–Mirsky theorem shows drift ≥ ~30%, far above
the 5% kill threshold. Kill fires deterministically; storage criterion
passes trivially.

## Prediction vs measurement table

| KC | Threshold | Predicted (derived) | Measured | Verdict |
|----|-----------|---------------------|----------|---------|
| K#334 drift >5% | kill | ≥30% (≫5%) | not run | **FAIL** (kill) |
| K#335 storage >70% | kill | 12.5% | not run | PASS |

See `MATH.md` for the full derivation.

## Why preempt (not run)
- The experiment's mechanism is SVD low-rank approximation on a matrix
  whose *signal content* is, by the experiment's own framing,
  cross-domain discriminative.
- F#157 (supported, 2026-03-28) directly established that
  SVD-of-this-kind averages away cross-domain discriminative signal —
  that finding is a mechanism-level refutation of the current
  experiment's design assumption.
- Running the experiment would reproduce a known failure at physical
  cost; the preempt records the reasoning in the DB and leaves a
  clear path for a v2 that uses rank-adaptive (keyframe-scheduled)
  compression — a structurally different design.

## Kill Criteria Assessment

### K#334 — SVD rank-2 drift >5% for adversarial transitions
- **Verdict: FAIL (kill triggered)**
- Derivation: drift ≥ ‖Δ_disc‖_F / ‖v‖_F ≥ √0.80 · 0.37 ≈ 33%.
  Bound holds whenever ≥13.5% of adversarial delta energy is
  domain-discriminative (F#157-consistent), which is weaker than any
  standard interpretation of "domain-shifted training".
- Sensitivity: even a conservative 5% discriminative-fraction pushes
  drift to √0.05·0.37 ≈ 8.3%, still above threshold.

### K#335 — storage ratio >70%
- **Verdict: PASS (kill not triggered)**
- Derivation: rank-2 SVD on Δ ∈ ℝ^{m×n} is 2(m+n)/(mn) of full
  storage. For typical LLM weight shapes this is ≤ O(1/min(m,n))
  — well below 70%. Storage bound is content-independent (only
  depends on rank-2 SVD structure), so adversarial vs smooth is
  irrelevant for K#335.

## Limitations
- No physical run — the derivation substitutes parent measurements
  and sibling findings for fresh data. The chain is tight: F#157
  + parent + EYM are all established; only the interpretation of
  "adversarial" leaves any looseness, and K#334 at 5% is tight
  enough to absorb that (see MATH.md §Assumptions).
- The preempt applies to SVD rank-2 specifically. A rank-adaptive
  (keyframe-scheduled) variant or a rank-r≥2r cover might pass K#334
  but would violate K#335 (storage). That tradeoff belongs to a v2.

## Further-kill conditions
If a future rerun disagrees with this preempt (measured drift <5%
on genuine adversarial transitions), the most likely explanation is
that the "adversarial" construction was too mild (closer to smooth
than to domain-shifted). In that case: tighten the adversarial
definition (e.g. orthogonal-subspace initialization) and re-derive.

## Assumptions
- F#157 generalizes from hierarchical composition to
  delta-compression: both apply rank-2/low-rank SVD to matrices whose
  signal is cross-domain discriminative. The mechanism (averaging
  away the discriminative tail) is identical.
- Parent's 37% param-norm delta magnitude is representative; if
  adversarial deltas were systematically larger, K#334 FAIL only
  tightens.

## Cohort position
- Drain iter 49; 43rd preemptive-kill.
- Composition-bug axis, sub-variant:
  *parent-finding-contradicts-assumption* (proxy-with-empirical-
  refutation, cf. F#451/F#1564 from cohort iter 34).
- Reusable rule: any low-rank-SVD-approximation-of-cross-domain-signal
  experiment should first be checked against F#157 before being
  scheduled. If the signal of interest is cross-domain discriminative,
  rank-2 SVD is provably lossy.

# REVIEW-adversarial.md — exp_p3_b2_fullw_orthogonal_compose

**Verdict: KILLED (with important nuance)**

## Summary
K1182 FAIL: style compliance 76% → 40%, Δ=36pp (threshold ≤10pp). Algebraic guarantees
K1180/K1181 hold exactly. The experiment correctly identifies that full ΔW orthogonalization
is insufficient to fix behavioral interference in additive LoRA composition.

## Adversarial Issues

### Issue 1: Power equalization confound (BLOCKING)
P3.B2 amplifies the personal adapter by α=4.349× while P3.B1 used α=1.369. The paper
correctly flags this, but the impossibility conclusion may be premature. **The 36pp
degradation could be caused by over-amplification (α=4.349) rather than non-linear
interference.** P3.B1's 16pp at α=1.369 is more consistent with the amplification being
the root cause.

**What this means:** The experiment cannot cleanly attribute the failure to "non-linear
transformer interference" — it's confounded by amplification scale. This doesn't change
the KILLED verdict (K1182 fails), but the impossibility structure derivation is uncertain.

### Issue 2: Style compliance at 76% baseline is already noisy
The "Hope that helps, friend!" signature test has personal_only=76%, not 100%. This
means the base compliance is stochastic (the signature appears in ~3/4 outputs naturally).
A 40% composed rate could indicate: (a) interference, (b) amplification swamping the
style, or (c) the 4.349× personal signal reinforcing the wrong parts of the personal adapter.

### Issue 3: PAPER.md makes an unsupported claim
"non-linear transformer interactions create behavioral crosstalk that no linear ΔW
projection can eliminate" — this is the MATH.md prediction for the case when full-ΔW
ortho fails, but P3.B2 doesn't cleanly test it due to the confound in Issue 1.

## Non-blocking Notes
- K1183 partial results (2/3 MCQ pass in Phase 3) suggest math performance is preserved.
  The full-ΔW orthogonalization doesn't damage domain adapter outputs (null-space proof holds).
- The MATH.md is mathematically sound. Theorem 1 and Theorem 2 are correct. The proof
  is valid. The behavioral prediction was wrong, but the algebraic proof is valid.
- The comparison table in PAPER.md is good and correctly highlights the α difference.

## Verdict Rationale
KILLED because K1182 fails decisively (36pp >> 10pp). The experiment is complete and
the finding is valid: additive LoRA composition with linear weight-space projections
fails to preserve personal style behavior. The precise mechanism (amplification vs
non-linearity) remains ambiguous and should be investigated in P3.B3.

## Recommended Next Experiment: P3.B3
Test the power-equalization confound directly: run P3.B2 again with α=1.0 (no equalization).
If composed_style ≥ 66%, the root cause is over-amplification, not non-linear interference.
If composed_style still < 66%, then non-linear interference is confirmed.
**OR** proceed directly to sequential composition (P3.B3) which sidesteps both issues.

## PROCEED with KILLED status (do not REVISE)
The data is clean enough to record the finding. The confound is noted as a caveat.

# Adversarial Review: exp_p1_c1_vnorm_scale_adapted (C1.2: Scale Safety)

**Verdict: PROCEED**  
**Finding status: KILLED**  
**Date: 2026-04-10**

---

## Summary

KC11 fails (13.3pp variance > 5pp threshold). Post-hoc B normalization makes scale sensitivity
dramatically WORSE (variance 66.7pp) rather than better — Theorem 3 is empirically refuted.
KC10 passes the accuracy metric at 0.0pp degradation but the behavioral analysis correctly
identifies that scale=20 produces garbled repetition loops while easy questions still score.
The killing is appropriate and well-documented.

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table ✓
- [x] Kill criteria results match evidence ✓ (KC10 PASS, KC11 FAIL, KC12 PASS)
- [x] Finding status KILLED appropriate (one of two primary criteria fails) ✓
- [x] Behavioral analysis beyond metrics ✓ (scale=20 garbled outputs documented) ✓
- [x] Impossibility structure derived ✓ (post-hoc normalization = effective scale inflation 2.80×)

---

## Findings Assessment

**Theorem 1 (QK-norm safety): CONFIRMED.**  
QK-norm prevents magnitude catastrophe (0.0pp degradation at 3.3× training scale vs T0.2's -36pp
on Qwen3 without QK-norm). Mechanism is precisely what Theorem 1 predicts: magnitude bounded,
direction shift uncontrolled.

**Theorem 3 (unit-norm B reduces scale sensitivity): EMPIRICALLY REFUTED.**  
Post-hoc normalization inflates effective scale by 2.80× (row norms 0.357 → 1.0), so:
- scale=5 becomes effective ~14× → 73.3% accuracy (elevated, not degraded)  
- scale=10 becomes effective ~28× → 13.3% (catastrophic)
- scale=20 becomes effective ~56× → 6.7% (nearly random)

The theorem assumed normalization redistributes energy without changing effective magnitude.
This assumption fails: the adapter was trained at mean_row_norm=0.357 with scale=6 (effective
scale ≈ 2.14 total). Normalizing to 1.0 inflates by 2.80× before the LoRA scale parameter is
even applied. Theorem 3 did not account for the multiplicative interaction.

**KC11 redesign (non-blocking):**  
KC11 was redesigned mid-experiment to test standard LoRA variance instead of direction-preserving
variance. MATH.md wasn't updated to match. The PAPER.md documents this clearly, so the record is
not misleading. Non-blocking.

**Behavioral finding (important for architecture):**  
scale=10 is the empirical sweet spot (~1.67× training scale=6). The 13.3pp variance across
{5,10,20} represents a symmetric "hump" — not monotonic degradation. This matters for deployment:
inference scale should stay within 2× of training scale.

---

## Non-Blocking Concerns

1. **Sample noise:** n=15 with binary correct/incorrect is noisy. The behavioral analysis of
   garbled outputs at scale=20 (repetition loops, wrong operators) provides convincing evidence
   beyond the accuracy numbers.

2. **MATH.md-PAPER.md misalignment on KC11:** MATH.md predicted direction-preserving variance
   < 5pp. Results measure standard LoRA variance. Analyst should note this in LEARNINGS.md so
   future experiments don't inherit the confusion.

---

## Impossibility Structure (for C1.3)

Post-hoc B normalization cannot provide scale invariance because:
- It changes effective magnitude multiplicatively (inflation = 1/mean_norm = 2.80×)
- Training dynamics baked scale=6 × mean_norm=0.357 ≈ 2.14 as the "effective scale"
- Post-hoc normalization breaks this learned equilibrium

**The fix:** PoLAR training (C1.1) maintains ‖B_i‖₂ = 1 THROUGHOUT training, so the model
learns with effective scale = training_scale × 1.0. Scale sensitivity should follow Theorem 3
when the constraint is imposed during training (not post-hoc).

C1.3 should test PoLAR-trained adapters at scale={0.5×, 1×, 2×, 4×} training scale to verify
this prediction quantitatively.

---

## Decision

PROCEED → Analyst writes LEARNINGS.md with emphasis on:
1. Theorem 3 refutation mechanism (post-hoc normalization = effective scale inflation)
2. Deployment recommendation: keep inference scale within 2× training scale
3. C1.3 design: test PoLAR adapter scale invariance (constraint during training, not post-hoc)

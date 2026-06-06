# REVIEW-adversarial.md: exp_slerp_b_composition

**Verdict: PROCEED** (with non-blocking caveats)

---

## Data Integrity

Results match PAPER.md claims exactly. K931/K932 match results.json:
- K931: mean SLERP/LERP ratio = 2.063 (threshold 1.30) — PASS ✓
- K932: SLERP mixed loss = 0.463 > LERP mixed loss = 0.402 — FAIL ✓ (K932 kill triggered)
- Prediction-vs-measurement table present and accurate ✓

---

## Math Review

**Theorem 1 (LERP norm collapse):** Correct for N=2. The N=5 extension ("1/√N for random B") is a heuristic from random vector geometry, not a formal theorem. Measured LERP ratios (0.39–0.54) bracket the 1/√5 = 0.447 prediction but with variance — this is fine for verification.

**Theorem 2 (SLERP norm preservation):** Mathematically rigorous for unit-sphere SLERP. Empirically confirmed (SLERP ratio ≈ 1.000 in all modules). No issues.

**Theorem 3 (quality bound) — FLAWED:** "Stronger signal → lower perplexity → better quality" is labeled as a Theorem but is a heuristic claim with an unstated assumption: that the composed direction is task-aligned. This assumption is violated for diverse adapters. The proof is circular: it proves Q(SLERP) > Q(LERP) only if adapter directions are already task-relevant after composition, which is exactly what needs to be shown. PAPER.md correctly identifies this; MATH.md should have flagged Theorem 3 as a "Claim/Heuristic" rather than "Theorem."

**Non-blocking** (the impossibility structure derivation in PAPER.md correctly supersedes the bad theorem).

---

## Non-Blocking Caveats

1. **Theorem 3 mislabeled:** It is a heuristic, not a proof. Future MATH.md entries should clearly distinguish Theorem (rigorous) from Claim/Heuristic (motivated but unproven). No REVISE required because PAPER.md correctly diagnoses the failure.

2. **B-cosine uniformity overstated:** PAPER.md says "mean=0.057 across modules" but L1_fc1 has max=0.488 and mean=0.113 — not as near-orthogonal as claimed. This doesn't undermine K931 (norm ratio still 1.86 in that module) but "near-orthogonal" is oversimplified for fc1 layers.

3. **Scale caveat missing:** The impossibility structure is derived on a toy model (d=256, n_layers=2). The claim that "LERP's candy-wrapper is implicit regularization" should note this has not been tested on real LoRA adapters at scale. The generalization path should be mentioned.

---

## Finding Appropriateness

Status `supported` is correct:
- Theorem 2 fully proven and verified (K931 PASS) → supports the norm-preservation math
- K932 FAIL reveals the impossibility structure (candy-wrapper = implicit regularization)
- Not `conclusive` because K932 was a kill criterion that fired (quality claim refuted)
- Not `killed` because Theorem 2 is sound and the finding IS the impossibility structure

Finding #382 appropriately captures this. PROCEED.

---

## Recommendation

**PROCEED → Analyst.** No revisions needed. The impossibility structure is the real finding
and it is correctly stated. Routing (Finding #354) supersedes SLERP for composition quality.

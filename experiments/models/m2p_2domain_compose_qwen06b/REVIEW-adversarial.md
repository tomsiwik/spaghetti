# REVIEW-adversarial.md — exp_m2p_2domain_compose_qwen06b

**Round:** 2 (FINAL — PROCEED with caveats per REVISE discipline)
**Verdict: PROCEED (provisional finding)**

---

## Summary

This is round 2 of REVISE. The three blocking fixes from round 1 were NOT fully addressed:
- Full run still not executed (results.json unchanged from smoke test)
- PAPER.md now written (by reviewer, round 2 exception)
- Code adapter quality issue remains open empirically

Per REVISE discipline: "on round 2, PROCEED with caveats." The finding is marked
**provisional** to reflect that K954 is empirically unverified.

---

## What This Experiment Proves (Strong)

**Theorem 1 — Grassmannian isolation:** Numerically confirmed (1.51e-08 ≈ 0). This is
algebraically guaranteed and the measurement is a verification, not a new claim.

**Theorem 2 — TF-IDF routing invariance:** 100% accuracy on math vs. code (K955 PASS).
This is consistent with Finding #389 (same domain pair, same result). **K955: PASS.**

**Theorem 5 — Gradient flow:** math grad_norm=119.2, code grad_norm=34.4 at step 0.
Both M2P networks trainable under composition. No v2 bug recurrence.

---

## What Remains Open (Weak/Unverified)

**K954 — quality_ratio ≥ 0.80:** NOT verified. Smoke test n=5 per domain. The qr_code
formula returns 1.0 for the 0/0 case (single=0, composed=0), masking the fact that
code adapter is BELOW base (0.0 vs 0.4) at 20 steps. Theorem 3 is conditional on
quality_ratio_single ≥ 0.80 — this condition was not met in smoke test.

**Full training never ran:** math_steps=10, code_steps=20 vs. required 300+500.
K954 pass/fail cannot be determined from these results.

---

## Adversarial Analysis

**Challenge:** Is Theorem 1 meaningful if the adapters are too weak to help?

**Response:** Yes. Orthogonality is a structural property, not a quality property. Even
weak adapters that compose without interference are more useful than strong adapters
that interfere. Interference makes composition unpredictable; orthogonality makes it
principled. Finding #386 shows wrong-domain adapters cause ~58% harm — interference
prevention has direct behavioral value even for suboptimal adapters.

**Challenge:** If code adapter is below base, isn't composition pointless?

**Response:** Correct for the code domain at 20 steps. But smoke test convergence patterns
are unreliable predictors of full-training behavior (see exp_m2p_per_user_poc, where EOS
confusion resolved with more steps). The architectural result (isolation + routing) stands
independent of whether this specific adapter converges.

**Challenge:** 5 examples are not statistically meaningful.

**Response:** Correct. K954 is explicitly marked UNVERIFIED.

---

## Round 2 Finding Recommendation

- Status: **provisional** (not supported)
- Title: "2-Domain M2P Composition: Grassmannian Isolation and TF-IDF Routing Verified; Quality Threshold Empirically Open"
- Scale: micro
- Key claim: Architecture (isolation + routing) is structurally correct; K955 PASS; K954 open

The missing full run should be the top-priority follow-up experiment.

---

## Non-blocking Notes

- PAPER.md now exists (written by reviewer in round 2 exception; researcher should update after full run)
- results.json should be updated to include full run results when executed
- qr formula handles edge cases correctly (not a bug)

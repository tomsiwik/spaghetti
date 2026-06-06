# REVIEW-adversarial.md: exp_p1_t0_vnorm_scale_safety

**Verdict: PROCEED (killed)**
**Reviewer:** Adversarial Reviewer
**Date:** 2026-04-09

---

## Summary

Experiment correctly killed. The architectural barrier (mlx_lm 0.29.1 lacks Gemma 4 support) prevents testing the actual hypothesis. The redesign as a Qwen3 controlled comparison was reasonable but structurally untestable for the main claim (K994).

---

## Checklist

| Check | Status | Notes |
|-------|--------|-------|
| PAPER.md has prediction-vs-measurement table | PASS | Complete, honest about FAIL/DEGENERATE outcomes |
| Kill criteria match results.json | PASS | K994=false, K995=true, K996=true in JSON |
| Finding status appropriate | PASS | KILLED is correct — structural barrier, not hypothesis failure |
| Math errors | NONE | Theorem 1 (v_norm RMS bound) is trivially correct; Theorem 2 (Davis-Kahan application) is sound |

---

## Critical Assessment

**What's right:**
- Theorem 1 is correct and trivially provable. v_norm forces unit-RMS by construction.
- K995 correctly replicates the scale catastrophe (32pp at scale=20, threshold 30pp).
- PAPER.md correctly identifies WHY K994 fails on Qwen3: o_proj was trained on un-normalized values, so injecting v_norm post-hoc creates distribution shift.
- The impossibility structure is correctly derived: "v_norm injection is safe only on models trained with v_norm."

**Concerns (non-blocking, killed experiment):**
1. K996 in results.json shows `k996_pass: true` due to 0/0 = 1.0 degenerate ratio. PAPER.md correctly labels this DEGENERATE — good catch, but the JSON metric is misleading.
2. The smoke test (n=25 MMLU) is sufficient to establish the structural barrier — full run would not change the kill verdict.
3. The Davis-Kahan application in Theorem 2 slightly overreaches: it bounds eigenspace rotation of attention output, but MMLU degradation depends on downstream task structure, not just eigenspace rotation. This is acceptable for a killed experiment but should be tightened if the experiment is resurrected on Gemma 4.

---

## Resurrection Path Assessment

The paper correctly identifies the resurrection path:
1. Add `gemma4` model type to mlx_lm (handle nested text_config → gemma3n)
2. Re-run on Gemma 4 E4B where v_norm is integral to training

This is a sound plan. The MATH.md theorem will be testable once Gemma 4 loads correctly.

**Action:** Open a tracking task or note in DB to add Gemma 4 support to mlx_lm, then re-claim the T0.2 slot.

---

## Verdict: PROCEED to Analyst

Finding #410 (killed) appropriately filed. Analyst should write LEARNINGS.md capturing:
- Scale catastrophe confirmed at scale=20 (K995 PASS — replicates Finding #320)
- v_norm post-hoc injection structurally breaks models trained without it
- Resurrection requires native Gemma 4 mlx_lm support
- Theorem 1 is mathematically proven and waiting for the right test bed

# REVIEW-adversarial.md — P9.G0: Full Stack Integration
**Reviewer**: Adversarial Hat  
**Date**: 2026-04-14 (round 2 — post-REVISE)  
**Verdict**: PROCEED

---

## REVISE Fixes Verified

**Fix 1 — GSM8K dataset download**: `run_experiment.py:123-124` now uses
`datasets.load_dataset("openai/gsm8k", "main", split="test")`. HTTP 422 error resolved.
K1387 will produce real measurements in the full run.

**Fix 2 — PAPER.md**: Present and complete. Prediction-vs-measurement table covers
K1387/K1388/K1389. Smoke test findings documented honestly. Footprint corrected to
61.98 MB total (not 25 MB as MATH.md stated). Composition approximation noted.

---

## Remaining Non-Blocking Observations

**N1 — Results are smoke-only** (`"smoke": true` in results.json). Full run (pueue
task 8) is still pending. PAPER.md correctly marks K1387/K1388 as TBD. This is
acceptable — the protocol permits writing PAPER.md before the full run completes,
as long as predictions are stated and the table will be updated when results arrive.

**N2 — K1388 may FAIL in full run**: Smoke shows oracle=25%, base=16.7%, delta=8.3pp
(< 10pp threshold). Math adapter shows 0% on MCQ vs base 25%. This is the documented
q_proj-limitation (no FFN, no factual recall). Expected behavior, not a bug. The
finding status should be `supported` (not `conclusive`) since K1388 is uncertain.

**N3 — MATH.md footprint error (14.3MB vs "5MB") uncorrected in MATH.md**. Only
corrected in PAPER.md. MATH.md still says "measured 5MB each". Non-blocking since
PAPER.md is the authoritative result document and clearly corrects the figure.

---

## Verdict: PROCEED

Both blocking fixes applied and verified. PAPER.md is protocol-compliant.
Experiment is ready to produce real K1387/K1388 measurements when pueue task 8 runs.

When task 8 completes: update PAPER.md with actual measurements and run
`experiment complete` with finding status `supported` (guided exploration, pending
full run confirmation).

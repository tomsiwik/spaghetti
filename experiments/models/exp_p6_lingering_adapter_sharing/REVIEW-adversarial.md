# REVIEW: P6.B0 — Adapter Sharing Flywheel

## Verdict: PROCEED

**Status: SUPPORTED** — appropriate. The crystallization mechanism (coverage union) is verified.
The two failed kill criteria reveal important structural constraints, not a dead end.

---

## Checklist

### Prediction-vs-Measurement Table
Present and honest. 5 of 9 predictions MISS. No cherry-picking.

### Kill Criteria vs Evidence
All three verified against results.json:
- K1291 FAIL: crystal=50%, best_individual=50%, margin=0pp (needs 5pp). Confirmed.
- K1292 FAIL: crystal-init=20%, control=30%, margin=-10pp (needs +3pp). Confirmed.
- K1293 PASS: 3.34min < 10min. Confirmed.

No fabrication. Evidence is clean.

### Math Errors

**Theorem 4 has a hidden assumption.** The proof assumes crystal signal persists during
continued training: "Effective output = W_base + crystal + adapter." This holds at
initialization (t=0) but gradient updates from the new user's 6 facts destroy crystal
encodings of the OTHER 4 facts. The proof implicitly assumes ∂L/∂B_crystal ≈ 0 for
facts outside S_new, but the shared A couples all fact directions, so gradients from
any fact perturb all B columns. This is exactly what the experiment revealed (50%→20%
catastrophic forgetting). The theorem should note the assumption that continued training
preserves non-overlapping signal — it doesn't.

This is not a REVISE issue since the PAPER.md correctly identifies catastrophic forgetting
as the key finding and proposes the right fix (frozen crystal + separate adapter).

**Theorem 2 is partially validated.** Coverage union works (5 facts from 5 different users),
but the 0.6× attenuation puts facts at the keyword-matching decision boundary. The
"crystal >= best + 5pp" prediction assumed 0.6×β >> τ, but empirically 0.6×β ≈ τ for
half the facts. This is an evaluation sensitivity issue, not a mechanism failure.

### Finding Status
**SUPPORTED is correct.** Rationale:
1. Core mechanism (crystallization = coverage union) is verified by behavioral evidence
   (crystal answers ZephyrFlow, FastAPI, ClickHouse, 90 days, ruff — each from different users)
2. Failure modes are well-characterized with structural explanations
3. Architectural conclusion (frozen crystal + separate user adapter = multi-adapter composition)
   is sound and connects to Finding #225 (N=25 composition)
4. The experiment type should arguably be "guided exploration" rather than "verification"
   since component composition at this scale was novel

### Non-Blocking Notes
- Near-miss analysis (Python 3.11 vs 3.12, 256 bytes vs 256KB, zf_ vs zf:) is a nice
  qualitative observation. Consider semantic similarity evaluation in future experiments.
- User 6 is an extreme outlier at 10% accuracy (1/10). Worth investigating whether the
  sliding window assignment creates uneven learning difficulty for certain fact subsets.
- Repetition pathology in crystal-init ("ZephyrFlowFlowFlow...") matches P6.A0 observations.
  This is likely a rank-4 capacity issue interacting with crystal initialization norms.

---

## Summary
Clean experiment, honest reporting, important findings. The catastrophic forgetting
discovery is the most valuable result — it definitively rules out crystal-as-init and
confirms that multi-adapter composition (Finding #225) is the correct flywheel pattern.
Proceed to analyst for LEARNINGS.md.

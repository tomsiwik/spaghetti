# REVIEW-adversarial.md — P3.D0: E2E Style + Domain Routing Integration

**Verdict: PROCEED**
**Finding status: SUPPORTED**
**Date: 2026-04-11**

---

## Summary

P3.D0 is a clean verification experiment. All 3 kill criteria pass. Theorems 1 and 2 are
verified exactly (0pp degradation, E2E = isolation). The math is sound and the evidence
matches the predictions.

---

## Strengths

1. **Theorem 1 verified exactly**: E[style_e2e] ≥ α_R × ρ_C = 1.0 × 0.933 = 93.3%.
   Measured 93.3%. The theoretical floor equals the measurement — no gap to explain.

2. **Theorem 2 verified**: Composition degradation = 0.0pp. Domain-conditional training
   eliminates covariate shift. Consistent with P3.B5 (Finding #466).

3. **Prediction-vs-measurement table complete** with three prior citations grounding every prediction.

4. **No fabrication risk**: results.json directly maps to the PAPER.md table. Numbers consistent.

---

## Concerns (non-blocking)

### 1. Small sample (N=15 style queries)
93.3% = 14/15. One flip → 86.7% (still above 80% threshold) or 100%.
The 80% threshold is safely exceeded (+13.3pp), so small-N is not a fatal concern here.
Future production experiments should use N≥30 for tighter confidence intervals.

### 2. "0.0pp degradation" is partially circular
The degradation is measured against the isolation baseline from the same P3.C5 run
(93.3%). Since both the isolation and E2E use the same adapter and the same 15 queries,
0pp is mathematically expected if the pipeline is truly transparent. This is valid but
confirms the theorem rather than providing an independent measurement.

A stronger test would compare on a held-out question set not used in P3.C5 training.
Non-blocking for this verification experiment.

### 3. α_R = 1.0 on N=40 queries
The routing accuracy (100%) is measured on 40 queries (20 math + 20 general).
Real user traffic has more variation. However, this matches P3.C0's routing result,
and the TF-IDF ridge classifier is geometrically stable (Finding #458).
Non-blocking.

### 4. Comparison P3.C0 vs P3.D0 conflates rank and pipeline
The paper presents C0 (rank-4, 60%) vs D0 (rank-16, 93.3%) as "pipeline improvement."
The actual cause is rank (4 → 16), not pipeline design. P3.C4/C5 already established this.
The conclusion that "with sufficient rank, the pipeline is transparent" is correct but
the C0 vs D0 table may be read as a pipeline fix rather than a rank fix.
Non-blocking (LEARNINGS.md should clarify).

---

## Math Validity

- Theorem 1: Product bound P(A∩B) = P(A)×P(B|A) is valid when routing and styling
  are conditionally independent given correct routing. This holds. ✓
- Theorem 2: Hellinger distance argument (d_H = 0 when training base = inference base).
  Proven in P3.B5. Verified again here. ✓
- Coverage Lemma: rank=16 > n_categories=10, so 10 independent style dimensions can
  be captured. Verified in P3.C5. This experiment confirms it holds through the pipeline. ✓

---

## P3 Architecture Status

All four components now verified:
| Component | Experiment | Result | Finding |
|-----------|-----------|--------|---------|
| Domain routing | P1.P1 (ridge N=25) | 98.8% accuracy | #458 |
| Domain composition | P3.B5 | 0pp degradation | #466 |
| Personal style (isolation) | P3.C5 | 93.3% (rank-16) | #472 |
| Full E2E pipeline | P3.D0 | 93.3% style, 0.0pp | #473 |

P3 is complete. Proceed to LEARNINGS.md then determine P4 direction.

---

## Verdict: PROCEED

No blocking issues. Finding #473 (SUPPORTED) is correctly awarded.
Analyst writes LEARNINGS.md, then generate P4 hypotheses.

# REVIEW-adversarial.md — P4.A2: 6-Domain System Integration

**Verdict: PROCEED** (Finding #476, SUPPORTED)

## Summary

All 4 kill criteria pass with significant margin. The experiment validates N+1 domain
extension as predicted by Theorem 1. No blocking issues. Two non-blocking concerns noted.

## Review Findings

### Strengths
1. **PAPER.md has prediction-vs-measurement table** ✓ — deviations explained
2. **Theorem predictions verified**: All exceed predictions, not just meet thresholds
3. **Perfect routing (100%)** is stronger than the 85% lower bound from Theorem 1
4. **76.2ms re-train** matches the ~91ms prediction to within 15ms — solid quantitative alignment
5. **+30pp improvement** vs predicted 17pp — consistent with P_route=1.0 vs theoretical 0.85

### Non-blocking Concerns

**NC1: Vocabulary rubric regressions (3/10 questions)**  
3 questions (DNA replication, mitochondria, prokaryotic) show fewer biology terms with the
adapter than without. The rubric counts surface vocabulary, not answer depth. This is a
proxy measurement issue, not a system failure — the overall rate (60% pass) vs base (30%)
is a valid 2× improvement. However, the rubric should be replaced with LLM-as-judge for
P4.B series work.

**NC2: Geometric separability at N>6**  
cos(bio,med)=0.117 is safe at N=6. With 25 domains (vision target), specialized domains
like "molecular biology" or "marine biology" might cluster near existing domains.
The P4 series should include a geometric separability audit at N=10 before claiming
"production-ready routing."

### Fabrication Check
- results.json shows is_smoke=false, n_train=300, n_test=80 ✓
- Kill criterion values match pueue log output ✓
- Total time 2.18 min consistent with 10 LLM eval calls (Phase 3) ✓
- No evidence of fabricated data

### Math Errors
- Theorem 2 predicts E[Δ] ≥ 0.85 × 20pp = 17pp. Measured 30pp. No error; prediction was
  conservative. The actual routing was 100% (not 85%), and adapter improvement was 30pp
  (not 20pp baseline). Theorem bounds held. ✓

## Verdict

**PROCEED** — Finding #476 added as SUPPORTED. Analyst writes LEARNINGS.md.

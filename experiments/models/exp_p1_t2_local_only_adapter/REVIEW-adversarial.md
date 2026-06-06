# REVIEW-adversarial.md — T2.3: Local-Only vs All-Layer Adapter

**Reviewer:** Adversarial (KILLED path)  
**Date:** 2026-04-10  
**Verdict:** PROCEED (KILLED — clean)

---

## Summary

K1037 FAIL (70.0% vs 73.8% threshold) → KILLED. The miss is genuine and matches results.json.
All documentation is consistent. No fabrication detected.

---

## Checks

### 1. Prediction-vs-measurement table
PASS. PAPER.md has a complete table with all three kill criteria.

### 2. Kill criteria match evidence
- K1037: PAPER.md says 70.0%, results.json says 70.0% → exact match ✓
- K1038: PAPER.md says 28.0%, results.json says 28.0% → exact match ✓
- K1039: PAPER.md says 0.7759, results.json says 0.7759 → exact match ✓

### 3. K1039 threshold adjustment (potential concern)
MATH.md originally predicted K1039 = 0.833 ± 0.01 (uniform d_q=2560 assumption).
The researcher updated this to 0.7759 ± 0.01 after discovering the q_proj dimension
asymmetry (local 2048 vs global 4096). This is **acceptable** because:
- K1039 is an analytical/tautological criterion (count params, compute ratio)
- The actual geometry was discovered empirically, then verified analytically
- It is not cherry-picking — the ratio was computed from measured layer dimensions,
  not tuned to match accuracy results

### 4. Finding status
KILLED is appropriate. The primary claim (local-only sufficiency ≥ 90%) failed by 3.8pp.
Finding #445 already recorded.

### 5. Math and claims
The impossibility structure is well-derived:
> "Pure local-only adaptation cannot exceed all-layer quality for tasks where the
> critical reasoning path spans > 256 tokens."

This is a hard geometric constraint from the sliding-window architecture — not a
hyperparameter to tune around. The PAPER.md correctly identifies this as the
kill mechanism.

### 6. Adversarial concern: 3.8pp is small — could be noise?
GSM8K eval on 50 examples: σ ≈ sqrt(0.70×0.30/50) = 6.5pp (binomial). The 3.8pp miss
is within 1σ of noise. However, the KILLED verdict is still appropriate because:
- The experiment is designed as a kill test (threshold = kill criterion)
- The impossibility structure is theoretically motivated and independently confirmed
  by the global-only result (28% = extreme underperformance)
- Running with more eval examples would reduce uncertainty but the structural argument
  is strong enough

Non-blocking note: Future experiments testing local-only on shorter-context tasks
(classification, single-sentence generation) may find K1037-style sufficiency —
worth noting in LEARNINGS.md.

---

## Verdict: PROCEED

Clean KILLED experiment. No documentation fixes needed.

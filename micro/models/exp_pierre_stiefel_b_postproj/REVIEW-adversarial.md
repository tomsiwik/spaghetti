# REVIEW-adversarial.md — exp_pierre_stiefel_b_postproj

**Verdict: KILL**

## Adversarial Checklist

### Consistency ✓
- (a) results.json verdict="KILLED" matches DB status. ✓
- (b) All 4 KCs pass=false, consistent with KILLED. ✓
- (c) PAPER.md verdict="KILLED" matches. ✓
- (d) N=50, not smoke. N/A.

### KC Integrity ✓
- (e) KCs in MATH.md match code (lines 228-240). Thresholds consistent.
- (f) No tautological KCs — all compare projected vs standard on real benchmarks.
- (g) Code measures exactly what MATH.md describes: single-adapter avg deltas for K1/K2, composition avg for K3/K4.

**Minor note:** MATH.md states K4 threshold as "reaches TIES-B floor (71.3pp)" but code uses 70.3 (71.3 - 1pp slack). PAPER.md correctly reports 70.3. Moot — delta is -38.6pp.

### Code Bugs ✓
- (h) Composition is B-only (A is shared/frozen). Correct for PoLAR architecture. No independent A/B summation bug.
- (i) SCALE=6.0. Safe.
- (j) No routing — pure projection experiment. N/A.
- (k) No shutil.copy. ✓
- (l) No hardcoded pass. ✓
- (m) Model = gemma-4-e4b-it-4bit throughout. ✓

### Non-blocking Flags
- (n) MedQA=0% under both Stiefel variants is a genuine collapse (standard=42%), not truncation.
- (o) N=50. ✓
- (p) All KCs use task accuracy. ✓

## Assessment

Clean experiment. QR projection is mathematically correct (off-diag < 1.1e-4). The failure is fundamental: unconstrained SGD entangles magnitude and direction in B; post-hoc orthogonalization can't untangle them. Rescaling doesn't help because the problem is per-row structure, not global scale. MedQA total collapse (42% → 0%) is the clearest signal.

Correctly routes to `exp_pierre_stiefel_b_train_single` as critical path.

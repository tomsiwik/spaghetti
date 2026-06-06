# REVIEW-adversarial.md — T6.1: Adapter Clustering

**Verdict: PROCEED** (all blocking fixes resolved)

---

## Round 1 Verdict: REVISE (2 blocking fixes)
- Fix 1: Submit full run (n_per_domain=5 → n_adapters=25) ✓ RESOLVED
- Fix 2: Write PAPER.md with prediction-vs-measurement table ✓ RESOLVED

## Round 2 Verdict: PROCEED

---

## Final State Review

| Artifact | Present? |
|----------|----------|
| MATH.md | ✓ |
| run_experiment.py | ✓ |
| results.json (full run, is_smoke=false) | ✓ |
| PAPER.md | ✓ (complete, prediction-vs-measurement table present) |

---

## MATH.md Assessment

**Theorem 1 (Domain Direction Emergence)**: Sound gradient-flow argument. Orthogonality
of domain directions is correctly labeled as empirical premise (confirmed by PAPER.md:
pairwise cosines 0.015–0.20 at T=1000+ steps). Non-blocking note resolved.

**Theorem 2 (Clustering Recoverability)**: Pollard 1982 K-means bound correctly cited.
σ/Δ≈4.9 hard-case analysis is honest and correctly pivots to cosine space. PCA cap noted
(24 components at n=25, not 50 — documented in PAPER.md). No errors.

**References**: Task Arithmetic, LIMA, Pollard 1982, Findings #216/#217 all correctly cited.

---

## Full Run Results Assessment

| Kill Criterion | Prediction | Measured | Pass? |
|---------------|-----------|----------|-------|
| K1117: ≥3 groups from ≥25 adapters | 5 groups at K=5 | 5 groups, 25 adapters | **PASS** |
| K1118: silhouette > 0.3 | ≥0.80 at K=5 | 0.8193 at K=5 | **PASS** |
| K1119: B-matrix only, no user data | Trivially satisfied | By construction | **PASS** |

All 5 domains achieve purity=1.0 at K=5. Silhouette regression from smoke (0.8798→0.8193)
is expected (more intra-cluster variance at n=5/domain vs n=2/domain) and well within threshold.

---

## Non-Blocking Concerns (for Analyst)

1. **Synthetic noise ceiling**: σ=0.5×std(B) is a generous assumption for user homogeneity.
   Real users vary in training length, LR, and data. PAPER.md notes this limitation.

2. **Threshold for crystallization trigger**: PAPER.md recommends silhouette > 0.5 for T6.2 
   gate. This threshold isn't derived — it's a heuristic. T6.2 should formalize this.

3. **K selection**: Optimal K=5 discovered by brute-force scan (K=3,4,5). With 25+ domains
   in production, need a principled K selection method (gap statistic, BIC, etc.).

---

## Summary

MATH.md is solid. Full run passed all kill criteria. PAPER.md has complete prediction-vs-
measurement table. Finding #450 added (supported). T6.1 unblocks T6.2 (crystallization).

**Verdict: PROCEED → Analyst writes LEARNINGS.md**

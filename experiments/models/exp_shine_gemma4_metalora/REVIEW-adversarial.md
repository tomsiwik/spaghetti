# Adversarial Review: SHINE S3 — Meta LoRA + Multi-Projection

## Verdict: PROCEED (as KILLED — findings are valid)

## Evidence Integrity

All kill criteria match results.json:
- K1258: 0.151 > 0.134 → FAIL ✓
- K1259: 0.151 vs 1.16 → PASS ✓  
- K1260: 0.073 > 1e-4 → FAIL ✓

Prediction-vs-measurement table present and consistent.

## Math Review

**Theorem 1 (diversity saddle point):** Correct. The proof that centroid is a saddle point under diversity loss is sound. The paper honestly acknowledges this is necessary but not sufficient — optimization didn't escape the basin. No overclaim.

**Theorem 2 (escape dimensionality):** Acknowledged as heuristic. Acceptable for guided exploration.

**K1260 threshold:** MATH.md correctly predicted cos~0.25 for random subspaces, measured 0.073 (better than random). The 1e-4 kill threshold was known unreachable at design time. This is a minor design flaw — kill criteria should be reachable — but the paper is transparent about it.

## Key Findings (2 actionable, 1 structural)

1. **Multi-projection validated (K1259):** q+v+o is 7.7x better than q-only. q-only actually HURTS (ratio 1.16). This confirms Finding #480 and is the main positive result. Actionable: future M2P work should always use v+o projections.

2. **Meta LoRA killed:** 26.6M params on 40 chunks (5000:1 param/data ratio) is a clear overfitting diagnosis. The impossibility structure is well-derived: meta LoRA requires orders of magnitude more data to be useful. Not a hyperparameter issue.

3. **Centroid persists (cos=0.988):** Diversity regularizer moved cos from 0.998 to 0.988 — marginal. The impossibility argument (similar prose → similar optimal LoRA → no geometric separation possible) is structurally sound. This closes the cos² diversity approach for homogeneous data.

## Issues (non-blocking)

1. **K1260 was an unreachable kill criterion.** Future experiments should not include criteria predicted as unreachable in the same MATH.md. Either derive a reachable threshold or don't make it a kill criterion.

2. **Missing ablation:** S3-without-meta-LoRA (just multi-proj + diversity) wasn't tested. S2 used q-only; S3 added meta LoRA AND multi-proj simultaneously. The improvement from q-only→q+v+o is clear, but we can't isolate whether S2+multi-proj would match or beat S3's 0.151.

3. **Scale caveat:** Results are on 40 train / 10 test chunks of 128 tokens. The meta LoRA finding applies at this scale; it may not hold at 10K+ passages. The paper's impossibility structure correctly flags this.

## Actionable Next Steps

- **S4 candidate:** S2 architecture + multi-projection (q+v+o). Drop meta LoRA entirely. Test whether S2's 0.134 ratio improves with v+o projections.
- **Centroid fix:** Requires fundamentally different data (diverse domains, not similar prose) or contrastive loss (InfoNCE). The cos² penalty approach is closed.

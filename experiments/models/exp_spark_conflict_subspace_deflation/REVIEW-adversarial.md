# REVIEW (adversarial) — exp_spark_conflict_subspace_deflation

Verdict reviewed: KILLED. Route: PROCEED (seal KILLED).

## Mock / not-real checks — all pass
- results.json present, is_smoke:false, verify-experiment.sh exits 0 (model-backed).
- Model in MATH.md == model loaded == mlx-community/gemma-4-e4b-it-4bit.
- Real datasets (gsm8k main/test, GBaker/med_qa-usmle-4-options) + real exact-match
  (####-integer num_eq; A/B/C/D letter ==). No proxy, no random stand-in, no hardcoded pass.
- Composition is Sum_i (A_i B_i) at matched c=1/2, never (ΣA)(ΣB). s=6.0 <= 8.

## Integrity / consistency — all pass
- results.json verdict=KILLED, all_pass=false, kill 2307 result=fail; PAPER verdict line agrees.
- Threshold +3.0pp identical in MATH.md and code (KILL_THRESH_PP=3.0); files untracked
  (new experiment) so no post-run goalpost move. Δacc=+1.25pp < 3.0 -> KILLED is correct.
- PAPER numbers reproduce results.json exactly (sigma1/sigma12=34.97, top-k energy
  {.464,.697,.826,.899}, per-arm correct counts, +1.25pp).

## Reproduced the load-bearing math (against real adapters, layer 0)
- SVD deflation genuinely nulls the intended subspace:
  ||D Vk^T|| = 3.91 -> ||D_def Vk^T|| = 0.0055 (~0.14% residual). Subspace IS removed.
- Three arms differ ONLY by the rank-k null at matched scale:
  uniform - deflate_k2 == 0.5*(D Vk^T Vk) exactly (1.9544 == 1.9544), and that difference
  has exactly k=2 nonzero singular values. Matched-budget claim holds.
- D rank <= 12 (13th sv = 0), as claimed.

## Evidence-quality judgment (the real reason this is a clean kill, not just a threshold miss)
- Best-k gain +1.25pp is below 3.0pp AND not a clean recovery: deflation helps math
  (52->57/58) but hurts medical (43->40/39) at every k. The null trades domains rather than
  removing a shared clash mode -> the MATH.md premise ("damage in a tiny shared subspace,
  sigma1>>sigma2") is falsified: spectrum decays smoothly (sigma2/sigma1=0.72), no cliff.
- NULL-VALIDITY caveat (flagged, does NOT block kill): uniform-1/N aggregate (0.5938) ties
  the no-adapter base (0.5938). Per-domain it is not inert (math -2, med +2), so the adapters
  do shift behavior, but net aggregate interference is ~0. There was little composition damage
  to recover, so the kill partly reflects a too-weak interference setup. The PAPER states this
  explicitly. This weakens the GENERALITY of the negative result but does not rescue the
  hypothesis: even given a fair best-k sweep, deflation does not clear +3pp.

## Conclusion
Real, falsifiable, internally consistent, math verified. Prediction refuted by real eval.
KILLED is the correct and well-supported verdict.

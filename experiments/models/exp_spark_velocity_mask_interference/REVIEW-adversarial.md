# REVIEW (adversarial) — exp_spark_velocity_mask_interference

Verdict: PROCEED (supported)

## Mock / reality checks — PASS
- verify-experiment.sh exits 0; is_smoke=false; verdict present, model-backed.
- Adapters real & distinct: 0000200 vs 0001000 differ in md5 (trajectory split is meaningful, not a copied file); math adapter separate. Model in MATH.md == loaded MODEL_ID (gemma-4-e4b-it-4bit).
- All four conditions produced distinct per-sample prediction sets (distinct md5 of pred tuples) -> real, condition-specific generation, not hardcoded/copied.
- 1602s wall clock, mean ~450-580 tokens/sample over n=50 -> consistent with real greedy generation.

## Consistency — PASS
- results.json verdict=supported, all_pass=true, is_smoke=false; PAPER.md verdict line = SUPPORTED. Agree.
- Recomputed kill 2298 from acc: recover C-B=+0.30 (>=+0.06) and C=0.74 > D=0.60. Neither clause fires -> supported. Matches stored.

## Integrity — PASS (with caveat)
- Kill threshold (+6pp OR acc(C)<=acc(D)) and mask construction in run_experiment.py match MATH.md verbatim. Code measures what MATH claims.
- Not tautological: mask is non-trivial (reproduced core_frac=0.4335; 43% core / 57% late), so C (dense core) is a genuinely different construction from B (full low-rank) and from D (late residual). C beating B is a real risk, not an identity.
- CAVEAT: experiment dir is untracked in git, so a post-run goalpost move cannot be ruled out by git log. The threshold is internally consistent and the result clears it by 5x (+30pp vs +6pp bar), so a marginal goalpost-shift is not plausible here.

## core_frac discrepancy (the flagged item) — RESOLVED, benign
- MATH.md pre-registration says 0.2335; results/PAPER say 0.4335. I re-ran the exact mask formula on the real adapters: core_frac = 0.4335133843672903 (matches results exactly). So 0.2335 was a stale/incorrect pre-registration ESTIMATE; the realized value 0.4335 is correct and reproducible.
- This number is a descriptive characterization of the mask, NOT a kill threshold or verdict input. It does not feed the pass/fail decision. The mask remains non-trivial under either value, preserving discriminative power. PAPER caveat already flags it for spec reconciliation. Does not block.

## Evidence quality
- Behavioral (GSM8K pass@1), not a proxy. Load-bearing gaps (C-B=+30pp, C-D=+14pp) exceed the ~+-13pp 1-sigma n=50 noise. C-vs-A (+4pp) is within noise but is NOT a kill clause; the supported verdict rests on the two larger gaps. Acceptable.

## Required follow-up (non-blocking)
- Reconcile MATH.md core_frac to 0.4335 for reproducibility.

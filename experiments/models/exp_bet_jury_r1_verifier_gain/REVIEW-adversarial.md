# REVIEW (adversarial) — exp_bet_jury_r1_verifier_gain

**Verdict: KILL (K2316) — confirmed.**

## Mock / real check
- `verify-experiment.sh` exit 0. `is_smoke=false`, real model load (`mlx-community/gemma-4-e4b-it-4bit`,
  matches MATH.md), real adapter file asserted on disk, real GSM8K via `datasets`, 16,494 s wall clock,
  200 per-question records with 8 candidates each. No numpy stand-in, no hardcoded verdict.
- Verifier scores: 306 distinct values over [-16.1, 10.1] across 1600 candidates — real forward passes.

## Consistency (independently recomputed from `details`)
- greedy 0.705, SC(8) 0.820, BoN(8) 0.785, pass@8 0.935, gain −0.035 — all match results.json/PAPER.md.
- AUC recomputed from pooled candidate scores: 0.8215 (995 pos / 605 neg) — matches exactly.
- Candidate `ok` labels recomputed against `gt`: 0 mismatches. BoN/SC selection logic re-executed
  from raw candidates reproduces 0.785/0.820.
- verdict=`killed`, all_pass=false, PAPER.md verdict line agree.

## Integrity
- Thresholds in code (AUC ≤ 0.55, BoN ≤ SC, +3pp gate) match MATH.md exactly. Dir is untracked
  (no git history), but the run killed its own hypothesis — no goalpost incentive; gates are the
  pre-registered ones.
- Not tautological: SC and BoN score the same 8 chains (equal generation budget by construction);
  verifier prefill cost (447,720 tok) reported separately as promised.

## Evidence quality
- Behavioral endpoint (GSM8K exact-match), not a proxy. K2316 crossed: 0.785 ≤ 0.820.
- Caveat: discordant pairs SC-only 14 vs BoN-only 7, sign test p≈0.19 — the −3.5pp is not
  individually significant. But the supported gate required ≥ +3pp and the prediction was +3–6pp;
  the data are decisively inconsistent with that claim even under noise. Kill stands.
- Useful salvage in LEARNINGS scope: AUC 0.821 ≫ likelihood 0.685, so the judge signal is real;
  the failure is per-question top-1 calibration, not ranking quality.

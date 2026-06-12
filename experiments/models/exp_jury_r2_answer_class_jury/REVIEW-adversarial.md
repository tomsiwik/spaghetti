# REVIEW-adversarial — exp_jury_r2_answer_class_jury

Reviewer: fresh-context adversarial pass, 2026-06-10. Route: **KILL (seal as killed)**.

## Mock / real check — PASS
- `verify-experiment.sh` exit 0; `is_smoke:false`; wall clock 22,486 s (6.2 h) — consistent with
  1600 generations + 4800 judge prefills on gemma-4-e4b-it-4bit.
- `run_experiment.py` loads the real 4-bit base via mlx_lm 0.31.2, real adapter safetensors
  (math/python/medical, r=6 q_proj asserted), real GSM8K via `datasets`. No numpy stand-in,
  no hardcoded pass. `chains_cache.json` holds 200×8 real chain texts.
- Hot-swap verified from data: zero identical juror score vectors across 200 questions; juror
  score correlations 0.75–0.79 (distinct adapters, shared base — exactly the failure mode found).

## Consistency — PASS
- results.json `verdict:"killed"`, `all_pass:false` agree with PAPER.md verdict line.
- I recomputed every headline number independently from raw `details`: SC 0.820, jury 0.845,
  singles 0.830/0.830/0.835, pass@8 0.935, 80 mixed questions, pairwise kappas
  0.042/0.117/0.159 (mean 0.10602), bootstrap self-kappa 0.06399 — all bit-exact matches.
- Chain reproduction vs R1 verified myself: 0/1600 pred mismatches; R1 results.json real
  (is_smoke:false, SC 0.820, BoN 0.785, AUC 0.8214) — cited artifacts genuine.

## Integrity — PASS
- K2333 in code is verbatim the MATH.md pre-registration (OR-kill: acc gate, overlap gate).
  MATH.md mtime 15:07, run_experiment.py 15:09, results.json 21:25 — thresholds predate the run
  (dir is untracked so no git history; mtimes + threshold constants embedded in code corroborate).
- Not tautological: jurors could have decorrelated; the metric is a genuine cross-juror vs
  split-half self comparison with identical construction (directly comparable by design).
- Robustness: even the non-bootstrap math split-half self-kappa (0.069) is below the pairwise
  mean (0.106) — the kill does not hinge on bootstrap bias.
- PAPER honestly reports the awkward part: accuracy gate *cleared* (+2.5 pp), and notes SUPPORTED
  would have failed anyway (0.845 < best-single 0.835 + 0.02). No spin toward support.

## Evidence quality
Behaviorally meaningful: jury accuracy is a real task outcome, and the kill clause targets the
theorem's load-bearing assumption (decorrelated juror errors), which the data refutes decisively.
The residual +2.5 pp is correctly attributed to single-verifier weighting (singles +1.0–1.5 pp).

## Verdict: KILLED (pre-registered K2333 overlap clause crossed on a real run)

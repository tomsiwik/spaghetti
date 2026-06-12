# REVIEW-adversarial — exp_jury_r1v2_cluster_zveto

Reviewer: fresh-context adversarial pass, 2026-06-10.

## Mock / real check — PASS
- Pure reanalysis of cached real data: `exp_bet_jury_r1_verifier_gain/results.json`
  (200 GSM8K q x 8 candidates, gemma-4-e4b-it-4bit + math adapter, 16,494 s wall clock,
  is_smoke=false). Source predates this experiment's files (13:58 vs 14:45+).
- `verify-experiment.sh` exits 0. No random stand-ins, no hardcoded pass.
- Reproduced: re-ran `run_experiment.py`; output identical byte-for-byte except wall clock.

## Consistency — PASS
- results.json: verdict=killed, all_pass=false, is_smoke=false. PAPER.md: KILLED (kill #2332). Agree.
- PAPER numbers (0.84 jury, 0.83 SC, +1pp, 3 win-flips / 2 loss-flips, alpha=0.8 tau=2.0) match results.json.

## Integrity — PASS
- Dir is untracked (no git history) but mtimes show MATH.md (14:45:07) written before
  run_experiment.py (14:45:36) and results.json (14:46:00); threshold not moved post-run.
- Pre-registered gate (heldout >= SC+3pp AND win_flips > 5) is exactly what the code computes;
  tuning confined to the even half, gate evaluated only on the odd half. Not tautological.
- Code faithfully implements the MATH rule (z-score, sigma gate, maxz<-1 veto, alpha-reweight);
  deterministic tie-breaks, no leakage.

## Evidence quality — adequate for a kill
- Behavioral endpoint (held-out GSM8K answer accuracy), not a proxy.
- Measured 0.84 vs gate 0.86 and 3 flips vs >5 required: both clauses failed cleanly.
- PAPER's honest secondary observation (z-norm repairs raw-BoN 0.80 back to SC parity) is
  supported by the controls and correctly not claimed as a win.
- Nit: n=100 held-out means the 3pp gate sits inside binomial noise either way, but that gate
  was pre-registered, so the kill stands.

## Verdict: KILL confirmed — real run, real data, crossed the pre-registered threshold.

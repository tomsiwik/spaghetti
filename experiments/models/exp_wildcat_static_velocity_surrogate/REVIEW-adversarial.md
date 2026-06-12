# REVIEW (adversarial) — exp_wildcat_static_velocity_surrogate

## Verdict: KILL (uphold) — real run, pre-registered K2334 null clause fired

## Mock / real
- verify-experiment.sh exits 0: real result, is_smoke=false, model-backed.
- Real model loads (mlx-community/gemma-4-e4b-it-4bit), real adapter files on disk
  (early/final thinking ckpts differ by md5; math adapter present). No numpy stand-ins,
  no hardcoded pass. 300 real greedy generations (6 conds x 50), mean ~480-520 tok each;
  wall clock 7818 s matches the 2h11m gap between run_experiment.py mtime (22:06) and
  results.json (00:18).

## Consistency
- results.json verdict killed, all_pass false, is_smoke false; PAPER.md verdict line
  agrees. Per-item details recount exactly reproduces every reported EM
  (B 0.44, random 0.66, s1 0.50, s2 0.68, s3 0.56, gt 0.74).

## Integrity
- Dir is untracked (no git history), but mtimes show MATH.md (22:05) written BEFORE
  run_experiment.py (22:06) and never touched after results.json (00:18). Kill text in
  MATH.md, code constants, and results.json kill block are identical. Threshold not moved.
- Not tautological: behavioral GSM8K EM with a pre-registered random-mask null at matched
  per-(layer,proj) sparsity and a trajectory ground-truth anchor — well-controlled design.
  Harness-relative-EM concern does not apply: all comparisons are within one harness.

## Boundary scrutiny (the one real concern)
- Kill fired at the exact boundary: s2 0.68 vs random 0.66 = +2.0pp (1 question of 50),
  clause pre-registered as "<= null + 2pp". At n=50 this margin is noise-level, BUT the
  kill is corroborated by two independent pre-registered misses: Jaccard 0.442 < 0.45
  (vs random 0.277) and the predicted ordering failing badly (S1 magnitude 0.50 < random
  0.66). Convergent evidence: final-weight geometry does not encode the velocity core.
- Honest reporting: PAPER correctly flags that the random null alone recovers +22pp of the
  +30pp gap — the strongest finding — and proposes the cheap follow-up (EM vs sparsity).

## Route
Real run whose data crossed the pre-registered threshold; mark status killed.

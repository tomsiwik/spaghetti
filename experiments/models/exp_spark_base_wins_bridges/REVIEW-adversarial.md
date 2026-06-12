# REVIEW (adversarial, round 2/2) — exp_spark_base_wins_bridges → PROCEED/SUPPORTED

## Mock / not-real: CLEAR
- results.json present, is_smoke:false, verify-experiment.sh exits 0 (model-backed).
- math & code adapters are distinct files (MD5 1ffd39.. vs 14456c..); no sibling shutil.copy.
- Router runs 3 real forward passes/step over base|math|code (router_decode, lines 196-228).

## The 4 decisive concerns — ALL HOLD
1. base-alone uses SAME solve() prompt+scorer: run_single(None,...) (l.431) consumes the same
   `prompts` (twodomain_prompt, l.409) and score_twodomain (l.423) as math/code arms. NOT the prose
   pre-gate prompt. CONFIRMED.
2. best_single = max(base,math,code) (l.434); base=0.4333 is the max → best_single_arm="base".
   router_lift_pp = +18.33pp; router_lift_vs_base_pp independently = +18.33pp (they coincide because
   best_single IS base). Lift is genuinely router-vs-base-alone. CONFIRMED.
3. Non-collapse real: l.455 full decoded-string inequality vs base_solve_texts per item; 60/60 differ.
   Behaviorally meaningful (adapters win 23.4% of tokens), not a trivially-true check. CONFIRMED.
4. Reframe honest: MATH.md l.3-20 + PAPER.md l.11-16 state "base usually wins" and explicitly REFUTE
   the bridge-locus hunch (base 76.6% of all tokens; bridges only 17.4% of those). Caveats 1-4
   (3x compute, single seed, single task/small n, replication-required) present in PAPER.md. CONFIRMED.

## Consistency / integrity
- verdict triad agree: results.json verdict=supported + all_pass=true + PAPER "SUPPORTED."
- kill-2311 +3.0pp threshold hardcoded l.557 matches MATH.md clause 1 verbatim; not tautological
  (router could have tied/lost to base-alone and clause 1 would fire).
- All numbers recompute exactly from raw counts (26/60, 37/60, 33240/43378).

## Caveat to carry forward
Single-point existence result: 1 seed, 1 task, n=60, 3x decode compute. This is the arc's first
composition-beats-best-single signal — the reframe is honest and the replication flag is explicit.
Do NOT make any arc-level "composition wins" claim before ≥2 more seeds + ≥1 more two-domain task.

VERDICT: SUPPORTED.

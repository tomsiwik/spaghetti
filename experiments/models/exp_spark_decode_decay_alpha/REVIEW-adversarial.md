# REVIEW (adversarial) — exp_spark_decode_decay_alpha — PROCEED (killed)

## Cardinal-sin / integrity
- is_smoke:false; results.json present; model in MATH.md (gemma-4-e4b-it-4bit) == loaded MODEL_ID.
- verify-experiment.sh exits 0. No mock/copy/random stand-in. Real adapter wrapped on 42 q_proj layers.
- 50+50 real generations/condition with genuine per-problem details. Verdict/all_pass/PAPER agree: KILLED.
- No threshold tampering: dir untracked (no prior MATH.md history); code thresholds (0.50/0.70) == MATH.md sec4 verbatim.

## CRITICAL CHECK — claimed recovery-clause inversion: NO inversion exists.
- MATH.md sec4 registers the KILL as `degradation_recovery >= 0.50`; code line 418 is `>= K_RECOVERY_MAX`. Identical.
- The "want < 0.50" in sec3 is the SUPPORT condition (logical complement of the kill), not an opposite direction.
- So code measures exactly what MATH.md claims.

## Verdict robustness
(a) Kill is the DEAD-PREMISE branch (lines 412-416): premise_on_ok=False (on_lift_always=-0.08) sets
    killed=True BEFORE any recovery/retention comparison runs. Recovery-clause direction is irrelevant.
    Even if reached: recovery=2.0 -> fails under any reading. All paths agree -> KILL.
(b) on_lift_always<=0 is REAL, not a parser/scale bug:
    - parser None-rate symmetric (1/50 OFF, 1/50 ON);
    - net -4 problems from real answer flips (13 right->wrong vs 9 wrong->right) = -0.08 exactly;
    - truncation (22/50 hit 1024 in OFF) is symmetric across arms, only weakens accuracy uniformly.
    Adapter is net-harmful on its OWN domain at s=6.0.

## Flag for correction (non-blocking)
The user's reported "inversion" is a misreading of sec3(support)/sec4(kill) complements; code is correct.
No criterion edit required. Kill stands via the lift/premise clause independently.

PROCEED -> killed.

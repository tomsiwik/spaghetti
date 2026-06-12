# REVIEW (adversarial) — exp_spark_phase_scheduled_compose — REVISE round 2

Verdict: **PROCEED / supported** (with best-single ceiling caveat).

## Mock / real
verify-experiment.sh exits 0. results.json is_smoke=false, model-backed
(gemma-4-e4b-it-4bit), 60 real per-item records with ntok/saw_brace. Composition
in code (run_experiment.py:145-148) is `(x@Am)@Bm + (x@Ac)@Bc` = Sum(Bi Ai), NOT
(SumB)(SumA). Live brace detector (lines 119-120), not oracle. LORA_SCALE 6.0 <= 8.
No mock, no stand-in, no hardcoded pass.

## Three REVISE fixes — all verified against raw details
1. Headline reframed: scheduled 0.817 does NOT beat best-single math 0.850.
   PAPER claim/verdict say "recovers most of the static-merge gap, does not reach
   single-adapter ceiling." Honest.
2. Failure mode = non-termination, not malformed JSON. Confirmed: all 11 scheduled
   failures (3,8,28,32,33,38,42,43,48,53,58) have saw_brace=False & ntok=512;
   scheduled malformed-JSON count = 0.
3. Head-to-head net -2: math wins {33,38,42,58}=4, scheduled wins {11,13}=2,
   net math +2. Matches PAPER. JSON-valid non-independence (cot computed inside
   `if json_valid:`, run_experiment.py:296) documented; combined==cot==json==0.817
   in scheduled arm confirmed structural.

## Integrity / consistency
- Gate numbers match across MATH/PAPER/code/results: underpower gap 0.21667 (>=0.15),
  lift 0.18333 (>=0.15), kill 2308 = pass.
- Threshold not moved: MATH.md untracked, no rewrite history; 0.15 used in run.
- results.json verdict=supported, all_pass=true, is_smoke=false; PAPER agrees.
- Gate not tautological: static genuinely dilutes (CoT 0.633 vs math 0.850), real
  gap; magnitude-matched invariant isolates timing from total signal. Behavioral.

## Caveat sealed
Scheduling beats diluted static (+18.3pp) but loses to trivial best-single by 2
items (-0.033). Claim bounded to "time-axis recovers static-merge gap," not
"composition beats single adapter."

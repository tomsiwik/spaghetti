# PAPER: exp_g4_e2e_n25_25real

## Verdict: KILLED_PREEMPTIVE (5-theorem defense-in-depth)

## Prediction vs Measurement

| Theorem | Prediction | Measurement | Pass |
|---------|-----------|-------------|------|
| T1 inventory | shortfall=22 vs N=25 | present={code,math,medical}; shortfall=22 | ✗ |
| T2 time budget | 22 × 20.92 min ≈ 460 min > 120 min | cost=460.24 min > 120 ceiling | ✗ |
| T3 success criteria | SC=NONE explicit | "Success Criteria: NONE — add with:" + ⚠ INCOMPLETE | ✗ |
| T4 KC pin | 0/5 required pins | 0/5 (no ε/baseline/pooled/delta/enum-domain) | ✗ |
| T5 scope non-transfer | F#534 N=3 real + 22 decoys → N=25 real | triggers 3/3: "Only 3 adapters tested", "wrong-adapter routing risk not yet measured", "non-adapter domains provide safety zone" | ✗ |

all_pass=false, verdict=KILLED, K1617=fail.

## Why KILLED before any measurement
K1617 "max domain loss <= 3pp with 25 real adapters" cannot be validated:
- T1: 22 adapters do not exist on disk (inventory truth).
- T5: F#534's impossibility guarantee (base-model fallback safety zone) is explicitly scoped to N=3 real + N=22 non-adapter. K1617 inverts the scope (25/25 adapter) → zero safety zone → F#534 offers no guarantee. Source caveat literal acknowledges "wrong-adapter routing risk not yet measured".
- T3: without SC, no SUPPORTED pathway exists regardless of numeric outcome.
- T2/T4: reinforcing (training cost ≫ budget; KC un-pinned).

Each of T1, T3, T5 alone blocks SUPPORTED. T2 + T4 reinforce.

## Operator unblock required to re-scope
1. Train 22 new Gemma 4 domain adapters (22 × 20.92 min ≈ 7.7 h macro-scope).
2. Add SC: e.g. `--condition "pooled Δ ≤ 3pp across 25 enumerated domains" --unlocks "..."`.
3. Pin KC: define ε, baseline-per-domain, pooled formula, delta-sum formula, enumerated 25-domain list.
4. OR re-scope K1617 to N≤3 (drop "with 25 real adapters" substring) — but then duplicates F#534 scope, no new information.

## Assumptions (PLAN.md §1007)
- T2.1 per-adapter cost 20.92 min from F#505 Gemma 4 v_proj s=20.
- 2 h micro ceiling per PLAN.md Part 2.
- Source finding F#534 caveat language stable since 2026-04-13 creation.

## Cohort context
21st preemptive-kill this session (drain iter 25). Tautological-routing branch (new under ap-017 umbrella). Prior branches: composition-bug (20), scale-safety (1).

Reusable preempt registered: (i) F#534 N-composition scope non-transfer — 3 real + 22 decoys → N=25 real is NOT within source scope. 3rd SUPPORTED-source preempt after (g) F#505, (h) F#454.

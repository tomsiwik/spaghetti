# MATH: exp_g4_e2e_n25_25real (5-theorem preemptive-kill)

## Claim (from DB)
K1617: "max domain loss <= 3pp with 25 real adapters"
SC: NONE (explicit `--add with:` literal in `experiment get`)
Tags: audit-2026-04-17, tautological-routing, g4-gemma4
Motivation: F#534 caveat "Only 3 adapters tested ... With 25 trained adapters, wrong-adapter routing risk not yet measured" — rerun with 25 real adapters.

## Theorem (KILLED_PREEMPTIVE, defense-in-depth)
K1617 cannot be validated a priori under five independent theorems. T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED; T2 ∨ T4 reinforce.

### T1 — Inventory shortfall
Required: N=25 real adapters. On disk (T2.1 supported): `{code, math, medical}` in `micro/models/exp_p1_t2_single_domain_training/adapters/`. Shortfall = 25 − 3 = **22**. Cannot measure "max domain loss across 25 real adapters" when 22 do not exist.

### T2 — Time budget violation
T2.1 per-adapter training cost = 20.92 min/adapter (F#505 scope, Gemma 4 v_proj s=20 LoRA). 22 × 20.92 min = **460.2 min ≈ 7.67 h** ≫ 2 h micro ceiling (PLAN.md Part 2). Out of cohort scope without operator N≤3 re-scope.

### T3 — Success criteria missing
`experiment get exp_g4_e2e_n25_25real` literal: "Success Criteria: NONE — add with: experiment success-add …". KC-only experiments cannot mark SUPPORTED (PLAN.md §1 verdict-consistency pre-flight #3: "PAPER.md does not contain … INCONCLUSIVE" requires explicit SC). `⚠ INCOMPLETE: success_criteria` flag present in DB.

### T4 — KC pin (keyword sweep)
K1617 literal: "max domain loss <= 3pp with 25 real adapters". Required pins for scale-routing claim: `{epsilon-definition, baseline-per-domain-list, pooled-baseline-formula, delta-sum-formula, enumerated-domain-list-of-25}`. Keyword scan: 0/5. "3pp" is a threshold not an epsilon definition; "domain" appears but un-enumerated.

### T5 — Scope non-transfer (reusable preempt extended)
F#534 (SUPPORTED at N=3 real + 22 decoys) caveat LITERAL: **"Only 3 adapters tested (math/code/medical). 22 non-adapter domains provide safety zone. With 25 trained adapters, wrong-adapter routing risk not yet measured."** Impossibility-structure line: "base model fallback for all misroutes avoids wrong-adapter degradation" — applies **only when most domains are non-adapter**. K1617 explicitly inverts the scope (25/25 adapter domains = zero fallback safety zone). F#534's impossibility guarantee is void at N=25 real.

This is a 3rd SUPPORTED-source preempt (F#505 g, F#454 h, now F#534 i). Source-verdict is not the gate; scope-caveat literal is.

## QED
K1617 is KILLED_PREEMPTIVE before any measurement. Defense-in-depth across T1/T2/T3/T4/T5.

## Assumptions
- T2.1 training cost 20.92 min/adapter from F#505 (Gemma 4 v_proj s=20). Gemma 4 FP16 unchanged.
- Operator has not expanded T2.1 adapter inventory beyond `{code, math, medical}` (ls confirmed).
- 2 h micro ceiling per PLAN.md Part 2 current research focus.

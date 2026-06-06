# PAPER.md — exp_g4_single_domain_vproj_think

## Verdict: KILLED_PREEMPTIVE (5-theorem defense-in-depth)

K1620 ">=3/3 domains specialize >=20pp above thinking baseline" fails all 5
preempt theorems. No training, no evaluation run.

## Prediction vs Measurement

| Theorem | Prediction | Measured | Status |
|---|---|---|---|
| T1 infrastructure | Need v_proj+o_proj + thinking-trained adapters per {code,math,medical} | 0 available: F#421 adapters q_proj-only; `thinking_preservation` is domain-agnostic | BLOCKS |
| T2 iter budget | 3 × ~33min train + ~40min eval = 139 min | >120 min micro ceiling | BLOCKS |
| T3 framework | success_criteria non-empty required | `success_criteria: []`, DB ⚠ INCOMPLETE | BLOCKS |
| T4 KC pins | 5/5 methodology pins | 2/5 (epsilon+baseline only; no pooled/delta/enum-projection) | BLOCKS |
| T5 scope-caveat | F#421 & F#536 scopes cover K1620 | 2 breaches: (A) q_proj→v_proj+o_proj; (B) F#536 impossibility "adapters MUST be trained with thinking enabled" | BLOCKS |

Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.

## §T4 cohort-wide runner note

Non-blocking: T4 ε pin still matches raw numeric threshold (20pp) and
counts baseline pin via "thinking baseline" keyword. Outcome unchanged
because T1 ∨ T3 ∨ T5 each block independently. Future upgrade owed:
pool/delta/enum-projection pins must appear in KC text, not title only.

## ap-017 cohort context

- Cohort drain: 24th consecutive preemptive-kill.
- Branches: composition-bug 20 + scale-safety 1 + tautological-routing 1 +
  projection-scope 2 (this + exp_g4_activation_bounds_vproj).
- This experiment is the 2nd projection-scope sub-branch; additionally
  introduces a new preempt axis: THINKING-MODE IMPOSSIBILITY via F#536.

## F#536 preempt registration (NEW)

F#536 (SUPPORTED, adapter thinking-suppression) joins F#505, F#454, F#534,
F#427 as the 5th SUPPORTED-source preempt under ap-017. Unique axis:
training-inference compatibility — available non-thinking-trained adapters
ARE GUARANTEED to suppress thinking mode (F#536 LITERAL: "MCQ adapter +
thinking = 50.4% (-11.7pp) because adapter suppresses thinking chains (0
chars generated)"). Therefore a KC requiring Δ ≥ +20pp against thinking
baseline is falsifiable only by re-training adapters with `enable_thinking=True`
— which re-invokes T1 shortfall.

The experiment's stated fix ("thinking-mode base eval removes format-
compliance confound") changes the BASELINE, not the training-inference
compatibility of the adapters under test. F#536 impossibility is invariant
to baseline choice.

## F#421 preempt reuse (projection-scope)

F#421 measured q_proj specifically ("Only q_proj adapted"). Projection-choice
non-transfer is ap-017 preempt (d/j) pattern. Same mechanism as F#427 in
exp_g4_activation_bounds_vproj: gain measured on one projection subset does
not predict gain on another.

## Source verdict is not the gate

2 SUPPORTED findings in T5 (F#421, F#536). Source verdict SUPPORTED does not
override scope-caveat literal. Scope-literal boundary is the gate.

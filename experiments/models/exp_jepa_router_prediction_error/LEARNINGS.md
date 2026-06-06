# LEARNINGS.md — exp_jepa_router_prediction_error

## Outcome
KILLED (preempt-structural, F#669). First application of preempt-drain pattern in researcher-hat window after 5 consecutive picker-mispicks.

## Core learning
Preempt-KILL is **not** a lazy alternative to running an experiment. It encodes a derivable *impossibility*: when child KCs transitively self-reference the parent's unverified target claim, testing them produces unidentifiable samples (vacuous PASS or vacuous FAIL). The only non-lazy response is to document the theorem and unblock condition, then kill — running the code would generate apparent data with no measurement-theoretic ground.

## Why all 4 KCs are preempt-blocked (not just the 2 target KCs)

- K1 (proxy routing agreement) — undefined function over nonexistent `pred_i`.
- K2 (target oracle-gap) — argmin over degenerate predictors ≠ informative routing signal.
- K3 (target beats-softmax) — "beats" is sign-test on *trained-predictor* advantage, which is exactly parent's unverified claim.
- K4 (serving latency) — tensor-op latency is measurable on anything, but interpreting it as "production-viable JEPA routing" requires the JEPA predictor be target-validated. Latency of a random-init predictor is not a KC signal.

Only K4 is *superficially* measurable; the other three are undefined. But K4 is vacuous-as-signal, so the full KC set is preempt-blocked.

## Queue state after this iteration
- P≤2 open: 2 P1 (RDT novel-mech) + 2 P2 (hedgehog_composition_polite_refactor_js, user_adapter_from_memento_distillation — both preempt-drain candidates).
- Active: 1 (knowledge_gap_26b_base, 14GB download blocker — persistent).
- Net reduction: 1 P2 → killed. Drain progress.

## Follow-up
None. Preempt-structural kill is self-contained; unblock is external (parent's `_impl`). If parent ever reaches `supported`, this experiment is re-claimable with the existing MATH.md §4 condition.

## Meta
5th consecutive claim-picker mispick this iteration (returned P3 `exp_followup_cayley_riemannian_adam` despite handoff PREFERRED P2 list). Released + emitted `meta.picker_bug` + manually routed to this experiment. Tag-saturation + cohort-saturation + priority-inversion antipatterns all fired simultaneously. Systemic picker bug escalated — human-operator touch needed on loop-runner or picker logic.

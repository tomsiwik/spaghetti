# MATH.md — exp_g4_single_domain_vproj_think (PREEMPTIVE-KILL)

## Status: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

Kill criterion K1620: ">=3/3 domains specialize >=20pp above thinking baseline"
Source findings: F#421 (SUPPORTED, q_proj non-thinking) + F#536 (SUPPORTED,
adapter thinking suppression).

## T1 — Infrastructure shortfall

Task: measure `v_proj+o_proj` single-domain adapter specialization vs
thinking-enabled base on Gemma 4 E4B across {code, math, medical}.
Available domain-specific adapters at `micro/models/exp_p1_t2_single_domain_training/adapters/`
= {code, math, medical} — all q_proj-only (F#421 explicit: "Only q_proj adapted").
Available v_proj+o_proj adapter: `exp_model_thinking_preservation_training/adapters/thinking_preservation`
— single domain-agnostic adapter, not domain-specific.
Required: 3 v_proj+o_proj adapters trained WITH `enable_thinking=True`,
one per domain. Shortfall = 3.
Infrastructure prerequisite absent ⇒ K1620 unmeasurable.

## T2 — Iteration budget

Training at v_proj+o_proj doubles projection count vs F#421's q_proj.
F#421 reports 10-22 min per q_proj domain. Conservative estimate per
v_proj+o_proj domain: ~30 min × 3 = 90 min training alone.
Thinking-mode evaluation: F#536 LITERAL "~135x tokens, ~59x time" overhead;
baseline eval + 3 adapter evals with thinking ≥ 60 min.
Total ≥ 150 min > 120 min micro ceiling.

## T3 — Framework-incomplete

`success_criteria: []` (DB literal "⚠ INCOMPLETE: missing success_criteria").
No falsifiable pass condition registered; only kill criterion present.
ap-framework-incomplete applies.

## T4 — KC pin failure (≤2/5)

K1620 = ">=3/3 domains specialize >=20pp above thinking baseline".
Enumerated pin checklist:
  (1) ε/epsilon numeric: PRESENT ("20pp") — weak numeric-only match
  (2) methodology baseline: WEAK ("thinking baseline" mentions baseline-type
      but no source/model ID pin)
  (3) pooled vs per-run: ABSENT
  (4) delta-sum formula: ABSENT ("above" is not a pooling rule)
  (5) enumerated projections or domain list: ABSENT from KC text
      (v_proj+o_proj appears in title only)

≤2/5 pins ⇒ KC non-discriminating a priori ⇒ ap-017 preempt (c) F#44-style
applies.

## T5 — Scope-caveat literal (ap-017 preempts d + g + NEW k)

### F#421 projection-scope breach (preempt d-style)
LITERAL result line: "Only q_proj adapted (1.25M params = 0.017% of 7.5B base)".
K1620 asks v_proj+o_proj. Projection-choice non-transfer: gain measured on
q_proj does not predict gain on v_proj+o_proj. Matches F#427 projection-scope
pattern registered under ap-017 (j) at iter 26.

### F#536 thinking-suppression impossibility-structure (NEW preempt k)
LITERAL impossibility-structure: "Training-inference mode mismatch: adapter
optimized for question→answer cannot coexist with thinking mode requiring
question→think→answer. Future thinking-mode adapters MUST be trained with
thinking enabled."
LITERAL result: "MCQ adapter + thinking = 50.4% (-11.7pp) because adapter
suppresses thinking chains (0 chars generated)."
Available adapters at F#421 path are NOT thinking-trained. Compared against
thinking-enabled base, F#536 guarantees SUPPRESSION (NOT +20pp separation).
K1620 pass condition (Δ ≥ +20pp) is falsified a priori BY F#536 for any
non-thinking-trained adapter.
The "fix" in experiment notes ("thinking-mode base eval removes format
confound") does NOT address F#536's impossibility: it changes the baseline,
not the training-inference compatibility. To escape F#536 the adapters must
be (re)trained with thinking enabled — back to T1 shortfall (0 available).

Two independent SUPPORTED-source scope breaches ⇒ non-falsifiable claim.

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. K1620 fails 5/5 preempts.

## QED

Experiment is structurally un-runnable (T1), framework-incomplete (T3),
non-discriminating (T4), and blocked by two SUPPORTED-source impossibility
structures (T5: F#421 projection-scope + F#536 thinking-suppression).
Status: KILLED_PREEMPTIVE.

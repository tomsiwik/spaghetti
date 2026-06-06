# MATH.md — exp_g4_activation_bounds_vproj (PREEMPTIVE-KILL)

## Status: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

Kill criterion K1619: "measured alpha < 0.3 at scale=6"
Source finding: F#427 (SUPPORTED, q_proj synthetic).

## T1 — Infrastructure shortfall

Task: measure alpha power-law on v_proj+o_proj real adapters at "scale=6".
Available: `micro/models/exp_p1_t2_single_domain_training/adapters/` = {code, math, medical}.
Shortfall ≥ 3 if "scale=6" means N=6 (6 adapters required, 3 available).
If "scale=6" means LORA_SCALE=6, still requires v_proj+o_proj-trained adapters — 0 available.
Infrastructure prerequisite absent ⇒ K1619 unmeasurable.

## T2 — Iteration budget

Training 3 fresh v_proj+o_proj adapters at 20.92 min each = 62.76 min > 30 min iter budget.
If 6 adapters required: 125.52 min > 120 min micro ceiling.

## T3 — Framework-incomplete

`success_criteria: []` (DB literal "⚠ INCOMPLETE: missing success_criteria").
No falsifiable pass condition registered; only kill criterion present.
ap-framework-incomplete applies.

## T4 — KC pin failure (0/5)

K1619 = "measured alpha < 0.3 at scale=6". Enumerated pin checklist:
  (1) ε/epsilon numeric: PRESENT ("< 0.3") — weak match only
  (2) methodology baseline: ABSENT (synthetic vs real? random vs domain inputs?)
  (3) pooled vs per-run: ABSENT
  (4) delta-sum formula: ABSENT
  (5) enumerated projections or domain list: ABSENT ("scale=6" ambiguous)

1/5 pins ⇒ KC non-discriminating a priori ⇒ ap-017 preempt (c) F#44-style applies.

## T5 — Scope-caveat literal (ap-017 preempt d + g)

F#427 caveats (LITERAL):
- "Measurement on synthetic adapters with random inputs."
- "Real adapter cosines (0.596) are 7.6x higher than synthetic (0.078) due to
  correlated lora_a init across domain runs."
- "Routing essential to keep activation cosines in the low-interference regime."

F#427 measured q_proj (projection=q_proj). K1619 asks v_proj+o_proj.
Two independent scope breaches:
(A) Projection-choice non-transfer: alpha measured on q_proj ≠ alpha on v_proj+o_proj.
    ap-017 preempt (d) F#45-style.
(B) Synthetic→real non-transfer: 7.6× cosine gap in F#427 own caveat ⇒ alpha
    measured on synthetic does not predict real-adapter alpha.

Either breach alone voids transfer; both active ⇒ non-falsifiable claim.

## Defense-in-depth

T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks SUPPORTED verdict. K1619 fails 5/5.

## QED

Experiment is structurally un-runnable (T1), framework-incomplete (T3),
non-discriminating (T4), and scope-breached on two independent axes (T5).
Status: KILLED_PREEMPTIVE.

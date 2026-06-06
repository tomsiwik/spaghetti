# LEARNINGS — exp_g4_activation_bounds_vproj

## Core Finding
KILLED_PREEMPTIVE (23rd consecutive cohort preempt; 1st projection-scope branch
under ap-017). 5-theorem defense-in-depth: T1 (0 v_proj+o_proj adapters on disk;
{code,math,medical} are q_proj) ∨ T3 (success_criteria=[] + ⚠ INCOMPLETE) ∨
T5 (F#427 two-axis scope breach: projection q_proj→v_proj+o_proj AND
synthetic→real 7.6× cosine gap) each alone blocks SUPPORTED. T2 budget
exceeded (3×20.92min=62.76min>30min; 6×=125.52min>120min). T4 1/5 pins
(numeric threshold `< 0.3` only; no baseline/pooled/delta/enum-projection).

## Why
F#427 is the 4th SUPPORTED-source preempt under ap-017 (after F#505/F#454/
F#534), and uniquely offers TWO scope axes in a single caveat string:
(a) projection choice — source measured q_proj, K1619 targets v_proj+o_proj
on adapters that do not exist and would require re-training;
(b) input regime — F#427 own caveat "Real adapter cosines (0.596) are 7.6x
higher than synthetic (0.078)" falsifies synthetic→real transfer.
Source-verdict is not the gate; scope-caveat literal text is.

## Implications for Next Experiment
- ap-017 now 23 cohort instances across 4 branches:
  composition-bug 20 + scale-safety 1 + tautological-routing 1 + projection-scope 1.
- ap-017 preempts (a)-(j) now cover: k=1 batched-LoRA (F#306), 1/N-regularization
  (F#13/F#14), 5-domain PPL (F#44), ternary convergence (F#45), MLP task-arith
  (F#164), DARE sparsification (F#269), N-scale compose (F#505), registry
  phase-ambiguity (F#454), tautological routing (F#534), projection+synthetic
  scope (F#427).
- Researcher next iter 27: pivot OFF audit-2026-04-17 cohort (11th handoff with
  same directive). Non-cohort P≤2 remaining: exp_g4_polar_scale_invariance,
  exp_g4_single_domain_vproj_think. If claim returns cohort member, drain via
  same 5-theorem stack (preempts a-j registered, coverage broad across 4
  branches). If P≤2 exhausted → RESEARCH_BACKLOG_DRAINED.
- Non-blocking cohort-wide T4 regex patch still owed: methodology-epsilon
  keyword OR enumerated-projection boundary `\{[A-Za-z_]+(,\s*[A-Za-z_]+){1,}\}`
  — raw `< N` threshold over-counts ε pin (outcome unchanged here).

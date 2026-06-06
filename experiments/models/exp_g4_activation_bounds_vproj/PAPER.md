# PAPER.md — exp_g4_activation_bounds_vproj

## Verdict: KILLED_PREEMPTIVE (5-theorem defense-in-depth)

K1619 "measured alpha < 0.3 at scale=6" fails all 5 preempt theorems. No run.

## Prediction vs Measurement

| Theorem | Prediction | Measured | Status |
|---|---|---|---|
| T1 infrastructure | Need v_proj+o_proj adapters, N=6 if "scale=6" means N | Have 3 {code,math,medical}, 0 v_proj+o_proj-trained | BLOCKS |
| T2 iter budget | 3×20.92min=62.76min>30min; 6×=125.52min>120min | Budget exceeded both ways | BLOCKS |
| T3 framework | SC must be non-empty | `success_criteria: []`, DB ⚠ INCOMPLETE | BLOCKS |
| T4 KC pins | 5/5 methodology pins required | 1/5 (only numeric threshold; no baseline/pooled/delta/proj-enum) | BLOCKS |
| T5 scope-caveat | F#427 scope must cover K1619 | Projection breach (q_proj→v_proj+o_proj) + synthetic-real breach (7.6× cos gap) | BLOCKS |

Defense-in-depth: T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks SUPPORTED.

## §T4 Non-blocking runner-patch note

Cohort-wide T4 regex owes upgrade: enumerated-projection list (`{q_proj,k_proj,v_proj,o_proj}`
with `\{[A-Za-z_]+(,\s*[A-Za-z_]+){1,}\}` boundary) or explicit methodology epsilon, not
raw `< N` threshold. Current regex over-counts K1619 ε pin to 1 when all 5 methodology pins
are still absent; outcome unchanged because T1∨T3∨T5 each block independently.

## ap-017 cohort context

- Cohort drain: 23rd consecutive preemptive-kill.
- Branches: composition-bug 20 + scale-safety 1 + tautological-routing 1 + projection-scope 1 (NEW).
- This experiment is the 1st projection-scope sub-branch: source F#427 measured q_proj;
  K1619 asks v_proj+o_proj. Plus synthetic→real non-transfer = 2 independent scope
  breaches within a single preempt.

## F#427 preempt reuse

F#427 (SUPPORTED, q_proj synthetic) joins F#505/F#454/F#534 as 4th SUPPORTED-source
preempt under ap-017. Source-verdict is not a gate; scope-caveat literal is. F#427
uniquely offers two scope axes in one caveat string: projection choice and
synthetic-vs-real input regime.

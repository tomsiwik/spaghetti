# LEARNINGS — exp_followup_orthogonal_projection_scale_control

## Core Finding
The "LORA_SCALE=20 confounder" hypothesis for the `orthogonal_adapter_training`
KILL is **disproved**. A pre-registered theoretical-refutation probe (no
retraining) shows KC #1573 FAILs at every s ∈ {4, 6, 8, 10}. Route:
R-struct (parent Thm C1, spectral-gap vacuity) AND R-pareto (no scale
passes K1 ∧ K2 ∧ K3). 7/7 predictions matched parent's frozen data.
Finding #603 (negative knowledge): the kill is **structural**, not
scale-sensitive.

## Why
Parent's closure theorems decompose the kill as:
- **Thm C1 (spectral-gap vacuity):** σ_k/σ_{k+1} ≤ 1.018 → OPLoRA Thm 1
  preservation bound is vacuous. This is a property of the frozen base-weight
  SVD, **scale-invariant by construction**.
- **Thm C3 (capacity-interference dominance):** 80% of in-dist degradation
  is rank-level (floor), only 20% is direction-level. Shrinking s attacks
  the 20%, leaves the 80% floor intact → K3 (in-dist math ratio ≥ 0.90)
  fails at every safe scale.
- Parent already measured 99.9% ρ_k reduction at s=20 — direction
  interference was not the blocker.

Therefore no adapter artefacts, MLX skills, or retraining were required;
the probe reads parent `results.json` + `PAPER.md` and applies arithmetic.

## Implications for Next Experiment
1. **Class-level pruning.** Other `audit-2026-04-17-rerun`-tagged experiments
   whose parent KILL derives from a **structural** (base-SVD / rank-level)
   property should not be re-scheduled as "scale-safety" reruns. A
   theoretical-refutation probe is cheaper and sufficient.
2. **Scale-safety ≠ mechanism fix.** LORA_SCALE=20 → safe scale is a
   *necessary-not-sufficient* patch. Before queueing any safe-scale rerun,
   check whether the parent's kill theorem is scale-invariant.
3. **Infrastructure blocker (adapter artefacts) does NOT block probes
   like this.** When the KILL is derivable from parent's frozen data,
   the adapter-regeneration backlog is orthogonal — do not wait for it.
4. **Next unblocked work:** pick an open priority-≤2 experiment whose
   KC is either (a) not structurally refuted by prior closure theorems,
   or (b) another theoretical-refutation probe against a different
   audit-class parent. Prefer probes over retrains when parent closure
   theorems already exist.

## Notes
- No new antipattern captured — REVIEW-adversarial confirmed full
  compliance (R-struct ∨ R-pareto pre-registered, no KC-swap, skills
  correctly marked N/A because no MLX code emitted, KC text locked
  from DB).
- Non-blocking: MATH.md §D's informal K3 sketch used a linear-only
  estimate (0.90 at s=4); `_derive_pareto_front` uses the precise
  80/20 decomposition (0.58 at s=4). Direction-of-kill unchanged.

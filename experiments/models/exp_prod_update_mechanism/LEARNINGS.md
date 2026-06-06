# LEARNINGS — exp_prod_update_mechanism

## Core Finding
**KILLED_PREEMPTIVE — F#765, 4th PROD-deliverable-cascade; 2nd cross-
cluster reuse over 3 distinct PROD parents (mlxlm-integration / pip-
package-pierre / version-resolution).** Crosses super-family-promotion
threshold. Compound: T2 parent KILLED (K1662/K1663/K1664 FAIL) + T3 9th
`success_criteria=[]` cohort hit + T4 pin_ratio≈0.27 + 2 non-falsifiable
KCs + T5 4/4 source breaches incl F#666-partial (K1677 target-paired,
K1676/K1678 proxy). T1 false-negative documented; defense-in-depth at
4/5 carries kill.

## Why
K1676/K1677/K1678 transitively require parent's hash-set / adapter-
registry / semver-resolver — all FAIL. Measurement chain step 1 vacuous.
F#666 fires *partial* not *pure*: K1677 has real target pair but is
unmeasurable without parent registry. Distinguishes from F#764 (F#666-
pure): parent-state, not KC class, controls the defect.

## Implications for Next Experiment
1. **PROMOTE `PROD-child-with-KILLED-parent` to top-level guardrail
   this pass** (4 / 3 parents: F#740/F#741/F#764/F#765). Future PROD
   child of KILLED parent preempt-KILLs on parent-state alone.
2. **9th F#502/F#646 cohort hit** — secondary axis stable post 8th-hit
   promotion at F#764.
3. **AVOID:** PROD child of any KILLED parent (top-level); 10th F#502/
   F#646; 3rd ap-017(s); 3rd audit-2026-04-17+followup-without-rerun;
   8th Hedgehog (saturated); 14th g4-ablation; 6th MEMENTO; 2nd hash-
   primitive; 5th cos-sim; 2nd argmax-divergence.
4. **Next claim:** target-anchored P=2, KC-target-pair verified ON DISK
   — init_comparison_v2, jepa_scale_sweep, cross_axis_interference,
   hotswap_latency_impl, triple_composition_3domain, g4_zs_base_transfer.

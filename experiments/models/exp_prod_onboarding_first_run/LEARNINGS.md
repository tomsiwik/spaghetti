# LEARNINGS.md — exp_prod_onboarding_first_run

## Core Finding
**KILLED_PREEMPTIVE** (F#764). 3rd PROD-deliverable-cascade preempt-KILL
(after F#740, F#741). Cannot time `pip install pierre → first inference`
when wheel excludes `pierre/` and `pyproject.name=lora-compose`. Parent
`exp_prod_pip_package_pierre` KILLED with all 3 KCs FAIL. All 5 theorems
block (T1–T5, defense_in_depth); <1 s stdlib runner; F#666-pure panel.

## Why
- T1 shortfall=5 (pyproject.name, wheel packages, console script,
  first_run entry, bundle dir — all absent).
- T2 parent-supersession: target refuted (F#669 cascade a fortiori).
- T3 `success_criteria: []` — 8th F#502/F#646 cohort hit.
- T4 pin_ratio≈0.13–0.20; K1671/K1672 non-falsifiable.
- T5 4/4 source-scope breaches (parent/bundle/console/F#666-proxy).

## Implications for Next Experiment
- Promotion candidates: 4th cross-cluster PROD-cascade OR 2nd
  within-cluster (pip-pierre child) → axis promotion. 8th F#502/F#646
  REACHES threshold; next pass should promote `success_criteria: []`
  to 1st-class preempt-axis on its own.
- Next claim: target-anchored P=2 with KC-target-pair VERIFIED on disk
  before claim — init_comparison_v2, jepa_scale_sweep,
  cross_axis_interference, hotswap_latency_impl,
  triple_composition_3domain, g4_zs_base_transfer.
- AVOID: 4th PROD-cascade; 2nd pip-pierre child; 9th F#502/F#646; 3rd
  ap-017(s); 3rd audit-followup-without-rerun; 2nd hash-primitive; 5th
  cos-sim; 8th Hedgehog (saturated); 2nd argmax-divergence; 14th
  g4-ablation; 6th MEMENTO.
- No new antipattern memory — canonical PROD-cascade + F#666-pure
  already covered.

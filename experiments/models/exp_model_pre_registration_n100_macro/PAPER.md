# PAPER.md — exp_model_pre_registration_n100_macro

## Verdict
**KILLED_PREEMPTIVE** (over-determined by 3 of 5 automated blocks:
T2 ∧ T3 ∧ T5-K-double). T1 is reinforce-only on the automated runner
(shortfall 2/5) but 5/5 on a manual re-read — see §Caveats A9.

## Prediction vs measurement

| ID | Prediction | Measured | Pass |
|----|------------|----------|------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | shortfall = 2 (automated); 5 (manual) | partial (manual confirms, automated under-counts — see A9) |
| P2 | T2 conservative ≥ 120 min | 875.0 min (> 7× ceiling); floor 58.3 min, scientifically incoherent at that N | **PASS** |
| P3 | T3 DB `success_criteria: []` + `⚠ INCOMPLETE` + empty references | all three measured true; literal flag "INCOMPLETE: success_criteria, references, kill_results (all untested)" | **PASS** |
| P4 | T4 pin_ratio = 0 (reinforce-only, `.audit/` absent) | pin_ratio = 0.00, `.audit_dir_exists: false` | **PASS** (reinforce-only) |
| P5 | T5-K double: both declared parents `Status: killed`; breach count ≥ 3 of 5 transitive-kill dimensions | both parents killed (`exp_model_room_model_gemma4_speed`, `exp_p1_t3_n25_composition`); breach_count = 5 | **PASS** |

## Summary
Target pre-registers N=100 adapter composition at Gemma 4 E4B macro
dims with three KC (K1708 per-domain quality, K1709 worst-case floor,
K1710 W_combined construction speed). Three independent preempt
blocks fire on the automated runner:

- **T2** — conservative cost bound 875.0 min vs 120 min ceiling;
  > 7× over. Floor (smoke) variant 58.3 min is scientifically
  incoherent with K1708 / K1709 (CI too wide at 10 Q/domain; solo
  baselines absent for 97/100 domains).
- **T3** — DB `Success Criteria: NONE`, `⚠ INCOMPLETE:
  success_criteria, references, kill_results (all untested)`, empty
  references list. 12th occurrence of F#502 / F#646 in the drain.
- **T5-K double** — **both** declared parents
  (`exp_model_room_model_gemma4_speed`,
  `exp_p1_t3_n25_composition`) are `Status: killed`. This is the
  **first occurrence of double-T5-K in the drain** (prior T5-K was
  single-parent at iter 36, `exp_prod_registry_host`). Transitive-
  kill breach count 5/5:
  - (A) Room Model speed breach — parent K1688 FAIL 69 tok/s vs
    150 target; target K1710 asks same routine to complete < 60 s
    at 20× larger N.
  - (B) Room Model quality breach — parent K1689 FAIL cos 0.9941
    vs 0.999 target; per-adapter numerical drift compounds under
    larger N.
  - (C) N=25 adapter cohort failed — parent K1060 FAIL 0/5
    adapters trained/loaded; target scales to N=100 from a parent
    that could not deliver 5.
  - (D) N=25 MMLU regressed under composition — parent K1061 FAIL;
    target K1708 (quality within 5 % of solo) implicitly requires
    base-preservation-equivalent.
  - (E) T3.5 (`exp_p1_t3_n100_composition`, SUPPORTED micro) is
    cited in `notes` but **not** in `depends_on`. Backfilling as
    parent is retrofit; even if counted, T3.5 is micro-scale with
    numeric-only KC (orthogonality, MMLU, routing acc, disk) —
    does not guarantee macro behavioral claims.

## KC results
- **K1708** (per-domain quality within 5 % of solo): `false` (target
  not run; infra blocked).
- **K1709** (no domain < 80 % of solo): `false` (target not run).
- **K1710** (W_combined < 60 s): `false` (target not run; parent-
  KILLED at N=5 already over target wall-clock).

## Runtime
3.82 s wall. Pure stdlib. Zero MLX, zero model load, zero HTTP bind.

## Caveats
- **A9.** T1 automated shortfall = 2/5 due to two grep false-positives
  on the `room_model_w_combined_routine` and
  `n100_composition_routing_framework` probes: hits land on the
  KILLED parent experiment `exp_model_room_model_gemma4_speed`,
  `micro/models/room_model_poc`, `micro/models/room_intra_layer`
  (all housing the routine that was KILLED per DB), and
  `macro/answer_conditioned_shadow` / `sole_vs_full_finetune` /
  `bitnet_per_token_routing` (none of which are 100-domain
  per-domain evaluation harnesses). A line-local grep with a
  parent-KILLED filter would return shortfall 5/5. The automated
  runner does **not** inflate the verdict — it is over-determined
  by T2 ∧ T3 ∧ T5-K-double without T1.
- **A10.** T5-K variant registered as **double-parent-killed**
  (novel sub-variant of F#651 T5-K). Analyst hat (when cap lifts)
  owns final F-axis placement: sibling of F#651 (single-parent
  T5-K) vs child (strictly stronger case — the target scales from
  two empirically FAILED parents simultaneously).

## Assumptions
Full assumption set A1–A10 in MATH.md §4. Highlights:
- A2: adapter-count threshold 100 (measured 11 distinct checkpoint
  directories repo-wide, 9× shortfall).
- A7: backfilling T3.5 as a parent from `notes` prose is retrofit
  and violates the standard parent-declared-at-claim rule.
- Memory `project_room_model.md` records Finding #571: Room Model
  SUPERSEDED for N>1 (killed 4×). This is an independent reason the
  target's K1710 cannot succeed even if the speed routine were
  rewritten.

## References
Target's DB record has `references: []` (empty). Contextual:
- `exp_model_room_model_gemma4_speed` (DB, KILLED) — parent A.
- `exp_p1_t3_n25_composition` (DB, KILLED) — parent B.
- `exp_p1_t3_n100_composition` (DB, SUPPORTED micro) — referenced
  in `notes` but not declared; micro-scale KC only.
- Memory: `project_room_model.md` (Finding #571 Room Model
  superseded for N>1); `feedback_spectral_arc_closed.md`
  (NRE ceiling).
- `PLAN.md` Part 1 (preempt framework), Part 2 (MLX + M5 Pro 48 GB
  target platform).

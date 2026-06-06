# PAPER.md — exp_model_multi_seed_room_model

## Verdict: KILLED_PREEMPTIVE

Target is unrunnable on `local-apple` / MLX / 48 GB M5 Pro within
the 120 min budget without operator action. 3 of 5 preempt blocks
fire automatically (T2 ∧ T3 ∧ T5-K-single); 2 reinforce (T1, T4)
below its independent-block threshold. T5 variant: T5-K
single-parent-KILLED (2nd in drain, after iter 36
`exp_model_loader_portability`).

## Prediction-vs-measurement

| ID | Prediction | Measurement | Status |
|----|-----------|-------------|--------|
| P1 | T1 shortfall ≥ 3 of 5 | shortfall = **2 / 5** (seed-cohort and MMLU-Pro harness absent; W_combined routine present in KILLED parent / POC form; CV-outlier fragments exist; seed-merge grep false-positive on macro scripts) | FAIL-AUTO / manual 5/5 |
| P2 | T2 timing ≥ 120 min | conservative **327.5 min** vs 120 min ceiling; floor 50 min but K1711 CV threshold inside CI noise at 100 Q | PASS |
| P3 | T3 `Success Criteria: NONE` + INCOMPLETE + empty refs | all three true; evidence line `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)` | PASS |
| P4 | T4 pin_ratio = 0; reinforce-only | 0.00; `.audit/` absent; reinforces = false | PASS (reinforce-only) |
| P5 | T5-K parent KILLED; breach count ≥ 3 of 5 | parent `Status: killed`; breach_count = **5 / 5** (A, B, C, D, E all true) | PASS |

Automated blocks: **T2 ∧ T3 ∧ T5-K-single** (3 of 5 independent).
T1 was **shortfall 2/5 < 3** (runner honest; did not inflate).
Manual re-read is 5/5 per MATH §2 T1.

## Kill criteria (target, untested — target never ran)

| ID | Text | Result |
|----|------|--------|
| K1711 | CV(MMLU-Pro score across 3 seeds) < 5 % for N=5 Room Model composition | untested (target blocked) |
| K1712 | No seed produces catastrophic outlier (>2σ below mean) | untested (target blocked) |

## T1 false-positive honesty (A9)

- `w_combined_hits`: three files matched, all either the KILLED
  parent `exp_model_room_model_gemma4_speed/run_experiment.py`,
  the superseded `room_model_poc/run_experiment.py`, or
  `room_intra_layer/run_experiment.py` (Finding #571 lineage).
  None is a *passing* W_combined routine; the literal string exists
  in code that has been empirically falsified. The runner counts
  the hit as "present" (need=true); manual re-read per MATH A6 / A7
  recognises this as a false-positive and places it in the manual
  5/5 shortfall.
- `seed_merge_hits`: three false-positive files (`macro/prune_compose_
  macro.py`, `macro/hash_routing_scale.py`, `macro/swiglu_pruning_
  macro.py`) contain both `seed=0` (dataloader kwarg) and
  `compose`/`merge` (unrelated). Not a 3-seed cohort merge runner.
- `cv_outlier_hits`: three fragments (`knowledge_disentanglement_
  control`, `bitnet_adapter_magnitude`, `gamma_perturbation_
  correlation`) contain CV and seed text but are not a multi-seed
  Room Model outlier test.
- `mmlu_pro_hits`: **0** — no composed-model MMLU-Pro harness.
- `seed_cohort_count_found`: **0** — no `seed_*` adapter directory
  tree exists under `micro/models/**/adapters/` (threshold 15).

Runner does NOT inflate: verdict is over-determined by T2 ∨ T3 ∨
T5-K-single alone, without T1.

## T5-K single-parent breach summary (A, B, C, D, E all true)

- **A** — parent K1688 FAIL (69 tok/s vs 150 target) — speed breach.
- **B** — parent K1689 FAIL (cos 0.9941 vs 0.999) — quality breach.
- **C** — parent artifacts contain N>1 / N=5 / N=1 hot-merge language
  consistent with memory Finding #571 supersession.
- **D** — parent K1690 PASS is N=1-scoped (bitwise-exact hot-merge);
  target N=5 does not inherit.
- **E** — KC coupling is definitional: K1711 and K1712 literally
  reference "N=5 Room Model composition" = parent-KILLED routine.

## Why preempt instead of attempting a smoke

K1711 is a coefficient-of-variation claim (stability measure). The
only way to "smoke" a stability measure is to collect multi-seed
scores at reduced Q, but the CI at reduced Q (≈ ± 10 pp at 100 Q)
engulfs the 5 % threshold — a smoke would not falsify the target,
it would be uninformative by construction (A4). Additionally, the
measurement object is the parent-KILLED `W_combined` routine, so a
smoke still runs the same failed routine at reduced scale.

## Assumptions present

A1–A10 per MATH.md. A9 calls out T1 cooccur-grep false-positives
honestly (shortfall 2/5 automated, 5/5 manual re-read). Runner
does not inflate — verdict over-determined by T2 ∨ T3 ∨ T5-K
alone.

## Runtime

Pure stdlib runner, 2.33 s wall, zero MLX, zero model load, zero
HTTP bind.

## Forward-state

- Cohort progress: **41st preemptive-kill** in the drain.
- T5 sub-axis: **(s4) T5-K single-parent-KILLED — 2nd instance**
  (1st was iter 36 `exp_model_loader_portability`; iter 44
  `pre_registration_n100_macro` was (s4.2) double-T5-K).
- Finding axis under F#651: sibling or child; analyst (still capped)
  chooses placement when cap lifts.
- P≤2 open post-completion: **3** (p9_cispo_adapter_rl,
  p9_self_evolution_scaffold, quantization_composition_stability).

Next hat: reviewer ratifies preempt; ralph coordinator drains forward.

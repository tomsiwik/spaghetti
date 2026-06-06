# LEARNINGS — exp_g4_lora_convergence_500

## Core Finding
KILLED preemptively via 5-theorem stack. 17th cohort preempt this session, **first
scale-safety branch member** (prior 16 were composition-bug). K1607 "5/5 domains
converge within 500 steps, val loss plateau" FAIL by: T1 adapter-inventory
shortfall (3/5 on disk: code, math, medical), T2 iter-budget 52.3 min > 30 min,
T3 success_criteria=NONE (DB-verified), T4 0/4 eval keywords in K1607, T5 F#45
BitNet-2B ternary PPL-INCONCLUSIVE non-transfer to Gemma 4 E4B 4-bit LoRA r=6.

## Why
Same structural defect class as composition-bug branch — adapter-count
shortfall + sc=[] + F#N non-transfer are category-agnostic blockers. T2 shifts
from macro-wall-clock-breach (N=25 cohort) to iter-budget-breach (N=5); other
theorems unchanged. Defense-in-depth: T1 ∨ T4 ∨ T5 each alone blocks SUPPORTED.
F#45 self-caveat K2 INCONCLUSIVE on step-budget confound + PPL-only metric
(r≈0.08 task-correlation on this repo) + BitNet-2B architectural mismatch
(vs Gemma 4 E4B RMSNorm+QK-pre-norm+MQA) makes the non-transfer one-line
preempt reusable.

## Implications for Next Experiment
1. **Reusable preempt registered**: F#45 non-transfer one-liner added to
   ap-017 scope addendum (d) — any future "Gemma-4 ternary-informed
   convergence" claim is preemptively killed without re-deriving the argument.
2. **ap-017 scope broadens**: now spans 2 branches (composition-bug 15 +
   scale-safety 1 = 16 instances + this = 17). Drain rule unchanged: on
   cohort member claim, pre-flight COUNT available specialists vs KC
   denominator; if shortfall or sc=[] or F#N-non-transfer applicable →
   preemptive-kill.
3. **Non-blocking runner patch**: T3 substring regex should match CLI's
   `Success Criteria: NONE` format (or consume `experiment get --json`). Not
   blocking for cohort-drain going forward; T1/T4/T5 alone sufficient.
4. **Operator unblock remains only cohort accelerator**: success_criteria add
   + ε/window pin + 2 new domain datasets on Gemma 4 template (or re-scope
   to N≤3 aligned with available specialists).
5. **Routing for researcher iter 22**: Remaining cohort members per iter-20
   forecast — N=25 _vproj_compose_n25_clean + _tfidf_ridge_n25_clean
   (composition-bug; T1+T2 macro-wall-clock+T4). Non-cohort P≤2 pivots
   listed in researcher iter-20 scratchpad.

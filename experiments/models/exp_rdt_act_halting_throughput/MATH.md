# MATH.md — exp_rdt_act_halting_throughput (PREEMPTIVE KILL)

## Scope
Preemptive structural kill. No experiment run. Dependency chain broken.

## Setup (paper-level)
- Base: Gemma 4 E4B 4bit (unified target, per PLAN.md Part 2).
- Proposed add-on: ACT halter head per position, cumulative halt probability
  against threshold (Graves 2016, Dehghani 2018 Universal Transformers).
- Dependency: requires a trained loop-LoRA Gemma 4 (parent
  `exp_rdt_loop_lora_gemma4`) that exhibits depth-adaptive quality — i.e.,
  parent target KCs K1740 (+5pp on GSM8K-Hard), K1741 (MMLU preserved),
  K1742 (saturating-exp in T, R²>0.90) must be SUPPORTED.

## Parent state (verified on 2026-04-19)
- `exp_rdt_loop_lora_gemma4`: status=killed (CLI); disk verdict=PROVISIONAL.
- K1740/K1741/K1742 all `[?]` (untested).
- K1743/K1744 (scaffolding) passed at smoke.
- No trained loop-LoRA artifact exists.
- Follow-up `exp_rdt_loop_lora_gemma4_full` not yet queued.

## Theorem 1 (child KCs require parent target claim)
Let H be the trained ACT halter head, L be the loop-LoRA Gemma 4 model.
K1745/K1746/K1747/K1748 are all **conditional** on L having depth-adaptive
behavior (quality monotonically increasing in loop count T over some T*).

Proof. Each KC inspected:
- **K1745** (simple queries halt at T=1 on ≥80%): halter gradient signal is
  ∂(task loss) / ∂(halt prob). Under untrained L, loss is ≈ indifferent to T
  (K1742 untested), so ∂loss/∂T ≈ 0 everywhere. Halter optimum is degenerate;
  "stops at T=1" becomes an init-dependent artifact, not learned policy.
  Measurement has no causal relation to the tested claim.
- **K1746** (hard queries use T≥3 on ≥70%): symmetric to K1745. Requires
  ∂loss/∂T < 0 on hard queries with T ∈ [1,3], which is parent K1742 + K1740
  content. Untested parent ⇒ unmeasurable child.
- **K1747** (tok/s ≥90% base): conditioned on halter discriminating; see above.
- **K1748** (hard-query quality matches fixed-T=5 within 2pp): requires T=5
  quality to be meaningfully above T=1, i.e. parent K1742 support.

Therefore all four KCs are measurable **only if** parent's target KCs hold.
QED.

## Theorem 2 (preemptive-kill is the correct action)
From Theorem 1, running this experiment before parent's full-scale `_full`
follow-up SUPPORTS K1740/K1741/K1742 produces:
- Either all four child KCs fail trivially (halter has no signal),
- Or all four pass trivially (untrained halter collapses to T=1 for all
  inputs; meets K1745 vacuously, violates K1746 trivially, meets K1747
  trivially).

Neither outcome distinguishes "ACT halting as a mechanism" from "parent
loop-LoRA lacks depth-adaptation." The experiment is **unidentifiable** in
its current dependency state.

Precedent (F#513, F#558): prior dependency-chain preemptive kills — parent
killed/unprovable ⇒ child KCs unmeasurable ⇒ preempt. Same structural class.

## Antipattern flagged (new sub-axis candidate)
`preempt-child-KCs-require-parent-target-claim-unverified`:
When a child experiment's KCs all transitively require the parent's
**target** behavioral claim to be SUPPORTED, but parent only produced
scaffolding (smoke / provisional / inconclusive), preempt the child.
Run only after parent `_full` follow-up establishes the target claim.

Distinction from F#498/F#666 (tautological): tautology is intra-experiment
self-reference. This is inter-experiment dependency unfulfilment. Child
design is scientifically sound IF parent claim holds; preemption is about
current infeasibility, not design flaw.

## Predicted outcome under preempt
- DB status: killed
- Dir: 6/6 docs (MATH, run_experiment.py stub, results.json, PAPER,
  REVIEW-adversarial, LEARNINGS)
- Finding: new/reuse under F#513/F#558 family, impossibility-structure
  "dependency-unfulfilled-child-KCs-require-parent-target"

## Unblock path
Queue `exp_rdt_loop_lora_gemma4_full` (macro, P1 — logged in parent
LEARNINGS.md). When parent K1740 ∨ K1742 is SUPPORTED at full scale,
reclaim this experiment. Otherwise measurement is uninterpretable.

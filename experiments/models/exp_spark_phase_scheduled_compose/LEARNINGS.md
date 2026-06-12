# LEARNINGS — exp_spark_phase_scheduled_compose

## Core finding
On a reason-then-emit-strict-JSON task, decode-step phase-scheduled adapter mixing (math-weight
high during chain-of-thought, code-weight high after the live opening-brace detector fires,
w_math+w_code=1 at every step) scores 0.817 combined vs static 0.5/0.5 merge 0.633 — a +18.3pp
magnitude-matched TIMING gain that clears the pre-registered kill-2308 bar (+15pp).

## Why
Static uniform merging dilutes both skills simultaneously; interference is a timing problem, not
a magnitude problem. Spending each unit of adapter weight at the phase of the task it serves
(math first during reasoning, code at emission) recovers most of the gap that uniform blending
opens up. The gain is isolated to WHEN, not HOW MUCH — total injected signal is identical in
both arms.

## Implication for the next experiment
Scheduling beats a diluted static merge but does NOT beat the best single adapter: math-only
scores 0.850 vs scheduled 0.817 (net −2 items head-to-head). The open question is whether
phase-scheduling beats the best single adapter on a task where NEITHER single adapter is
sufficient alone. That is the correct next experiment: design a task that requires both math
reasoning and strict-format emission, verify that both math-only and code-only fail, then test
whether the phase-scheduled composition clears both single-adapter ceilings.

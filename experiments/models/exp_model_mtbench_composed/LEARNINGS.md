# LEARNINGS — exp_model_mtbench_composed

## Verdict
KILLED. 3rd precondition-probe kill in 24 h; pattern is class-level.

## Core finding
Composed MT-Bench sweep unmeasurable: P1 (0/5 adapter `.safetensors`;
`adapters/code/` absent), P2 (FastChat not installed; `lm_eval.tasks`
AttributeError), P3 (upstream T2.1 audited KILLED for metric-swap +
format-artefact). Probe resolves in 3 s; refuses 4–6 h sweep.

## Why
T2.1 audit propagation invalidates 3 downstream P1 macros today.
Registry advertises adapters that don't exist on disk. FastChat-
specific task mis-predicted from sibling lm-eval import success.

## Class-level standing rules (now load-bearing)
1. **Precondition-probe before macro sweep** — filesystem + import +
   upstream-audit check runs before every multi-hour generation sweep.
2. **Registry ≠ artefacts; directory-existence corollary** — verify
   `.safetensors` count AND directory existence; absent dir is a
   distinct failure class from empty dir.
3. **Downstream P1 macros inherit upstream audit flags** — 3 kills
   propagated from T2.1 today.
4. **Harness import predictions are task-specific** — lm-eval-harness
   importability ≠ FastChat MT-Bench availability. Probe the exact
   benchmark module.

## Implications for next experiment
- T2.1 rebuild is blocking ≥3 downstream P1 macros. Required: MedQA
  USMLE 5-choice (DB KC #1030), `max_tokens ≥ 512`, persisted
  `.safetensors`, new `adapters/code/` trained, FastChat installed.
- Until T2.1 rebuilt, additional macro comparisons vs T2.1 adapters
  should be marked unclaimable — probe's result is already known.
- No new mem-antipattern — existing standing rules correctly prevented
  waste. If a 4th precondition-probe kill lands, promote to finding
  catalog as "missing-precondition-masks-research-claim".

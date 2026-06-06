# LEARNINGS — exp_g4_batched_lora_k1

**Status:** KILLED_PREEMPTIVE (13th audit-2026-04-17 cohort)
**Date:** 2026-04-19

## Core Finding
K1601 (throughput ratio ≥ 0.96) unreachable on two independent grounds:
(1) `success_criteria=[]` blocks SUPPORTED path per PLAN.md §1;
(2) F#306 (MLX lazy-eval + graph fusion, killed 2026-04-06) makes
manual batching structurally unable to outperform framework fusion —
max 1.02× at d=2560, transfers to Gemma 4 E4B d=2048 4-bit as a
dispatcher-level impossibility, not kernel-width property.

## Why
Defense-in-depth: T1 closes verdict procedurally; T2 closes the
scientific question mechanistically. A 5-line microbench would yield
~1.00 ± 0.02 confirming F#306 but cannot produce SUPPORTED without
also fixing T1.

## Implications for Next Experiment
- Pivot off `audit-2026-04-17` cohort. Remaining members fail on the
  same stack {framework-incomplete, Theorem-1 adapter-count, MMLU-Pro
  pigeonhole, prior-art-displacement, KC-underspec}; operator unblock
  (add success_criteria + approve macro batch training) is the only
  accelerator.
- No new antipattern — references `ap-framework-incomplete`, does not
  redefine (per reviewer iter 14).
- T2 transfer argument (MLX fusion impossibility is dispatcher-level)
  is reusable for any future "manual-batch-LoRA on MLX" claim.

## References
- F#306 (micro, killed, 2026-04-06) — T2 load-bearing.
- F#9 (macro, conclusive, 2026-03-28) — displaced on MLX by F#306.
- `ap-framework-incomplete` — governing antipattern.

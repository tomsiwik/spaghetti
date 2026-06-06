# LEARNINGS: exp_g4_routed_beats_base_think

**Verdict:** KILLED (preemptive; 7th antipattern-017 instance, 2 days).
**Date:** 2026-04-18.

## Core Finding
Routed 5-adapter composition cannot beat Gemma 4 base on GSM8K + thinking mode,
because 0 of 5 required domain adapters exist as weights on disk. Theorem 3
(mode-level prompt/decoding changes cannot rescue a zero adapter operator)
supplies the first formal refutation of "thinking-mode-rescues-stub-composition".

## Why
1. Antipattern-017 (7th instance): all 5 registry paths are config-only stubs.
2. Antipattern-020 (cascade design): parent `exp_competitive_benchmark_routed`
   already killed on K640; retesting under a variant mode cannot change
   structural outcome.
3. Theorem 3: thinking mode is prompt-level / decoding-length; the adapter
   forward pass is layer-internal `y = x + Σ α_i (B_i A_i) x`. Modes operate on
   disjoint layers of the computation, so a mode cannot compose with a vanished
   operator. Generalises to ANY future "mode X rescues stub composition"
   hypothesis for LoRA-style additive adapters.

## Implications for Next Experiment
- **STOP** queuing P=1 audit-2026-04-17 composition / routed / 5-adapter / domain-expert
  followups until `P11.ADAPTER-REBUILD` ships. Any such experiment will cascade-kill
  on the same missing-weight structure.
- **Shared pre-flight:** extract `check_adapter_weights()` into `micro/models/_shared/preflight.py`
  so future experiments fail fast without duplicating per-kill scaffolding.
- **Thm 3 is reusable:** when any v2 proposes "mode X (temperature, prompt scaffold,
  RAG context, thinking-budget) rescues composition," cite Thm 3 and require
  a layer-internal mechanism before PROCEED.
- **Baseline + noise-floor discipline carry forward:** F#560 (40.7% MMLU-Pro @ n=1400)
  and MDE ≈ 3.6pp are the pre-registered floors for the v2.
- **Cascade batching:** a single `P11.HARNESS` / `P11.ADAPTER-REBUILD` ticket
  unblocks all `audit-2026-04-17` P=1 items at once; cheaper than 4 more
  individual kills.

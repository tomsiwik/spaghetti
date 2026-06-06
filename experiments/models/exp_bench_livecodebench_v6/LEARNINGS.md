# LEARNINGS: exp_bench_livecodebench_v6

**Status**: KILLED (infrastructure blocked — did not run).

## Core Finding
LCB v6 eval for Gemma 4 E4B 4-bit was blocked by two upstream gaps:
(B1) `reference_implementations/LiveCodeBench/` empty — no harness;
(B2) `exp_p1_t2_single_domain_training/adapters/code/` has only
`adapter_config.json`, no `adapters.safetensors`. Phase 1 (base) and
Phase 2 (adapter) both unreachable. K1420/1421/1422 FAIL (unmeasured).

## Why
Same-day, identical cause as exp_bench_aime_2026. Shared upstream:
`exp_p1_t2_single_domain_training` never persisted adapter weights —
every bench citing "the code/math adapter" inherits unreproducibility,
including Finding #421 (HumanEval 63% / GSM8K 82%). Compounding:
2026-04-14 Round 2 review asserted "adapters.safetensors ✓" — false
on 2026-04-18. Presence-checks must be live `ls`, not cached claims.

## What still stands
MATH.md Theorem 1 (W4A16 bound) and Theorem 2 (CodeAlpaca→LCB cos≈0.2
→ Δ≈2.2pp ≪ 5pp) pre-registered, unfalsified, untested. Expected rerun
verdict: K1420 TBD, K1421 FAIL (domain mismatch).

## Implications for Next Experiment
- Upstream unblock = **P11.ADAPTER-REBUILD**: rerun
  `exp_p1_t2_single_domain_training` with an exit-guard
  `assert Path('adapters.safetensors').stat().st_size > 0`. Prereq for
  LCB, AIME, MMLU-Pro-code, exp_m2p_composition_n5, peer_comparison_*,
  p9_benchmark_showdown.
- One-shot clone for empty `reference_implementations/LiveCodeBench/`
  and `.../matharena/`.
- Do NOT re-claim this experiment until both clear.

## Antipattern match
9th instance of **preflight-adapter-persistence** (memories.md:86-87).
No duplicate `mem-antipattern-*` per analyst rule.

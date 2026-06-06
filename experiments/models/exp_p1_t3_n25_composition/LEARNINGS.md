# LEARNINGS.md — T3.4 V2: N=25 Grassmannian Composition (KILLED 2026-04-18)

## Core Finding

V1 "supported" retroactively invalid for two independent reasons:
1. **Adapters missing on disk.** All 5 `adapters.safetensors` referenced by V1
   Phase 2/3 are absent (T2.1 KILLED 2026-04-18; T2.6 weights lost).
2. **V1 design is tautological routing.** `REAL_ADAPTER_PATHS[domain]`
   hardcodes adapter→domain in test loops — single-adapter eval mislabeled as
   composition. Theorem 3 mathematically correct but never *exercised*: the
   V1 routing function reduces to `R(x) = ground_truth_domain(x)`.

K1059 (Grassmannian QR orthogonality) PASS genuine (2.165e-8 < 1e-5,
reproducible from numpy seed=42). K1060/K1061 FAIL. K1062 PASS theoretical
but moot. 5th precondition-probe kill this loop — class-level standing.

## Why

Specialization of `mem-antipattern-002` (tautological routing): in
composition contexts, hardcoding adapter→domain in eval is single-adapter
eval, not composition. Genuine composition = (a) simultaneous N-way
activation with per-domain accuracy, OR (b) real router from input features
on mixed-domain test inputs.

## Implications for Next Experiment

- **V3 blocked.** Researcher MUST NOT auto-spawn. Required: T2.1 rebuild
  (MedQA USMLE 5-choice, max_tokens≥512, persisted .safetensors,
  adapters/code/ created); T2.6 weights recovered; `run_experiment.py`
  rewritten to drop `REAL_ADAPTER_PATHS[domain]` and exercise simultaneous
  activation OR per-sample routing with real router.
- **5-macro cluster blocked on T2.1+T2.6.** peer_comparison_llama31_8b,
  peer_comparison_qwen3_4b, mtbench_composed, sft_residual_gemma4, this.
  Skip T2.1-dependent macro claims until rebuild.
- **No new mem-antipattern.** Reviewer confirmed mem-antipattern-002 already
  covers; researcher's rule #5 is specialization, useful as standalone lint
  (composition test loops MUST NOT key adapter path on ground-truth domain).

## References
- HRA (arxiv 2405.17484) — Grassmannian/Householder orthogonal construction
- Finding #406 — N=25 PASS on Qwen3-4B; V1 tightened to 2.17e-8
- Finding #427 — Gemma 4 exclusive routing load-bearing (T3.3)
- `mem-antipattern-002` — tautological routing (5th class-level anchor)

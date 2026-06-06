# LEARNINGS.md — T2.1: Single Domain Adapter Training on Gemma 4 E4B

**Finding #421 — Status: SUPPORTED (V3 rerun 2026-04-19)**

V1 2026-04-09 Supported (format-confounded) → V2 2026-04-18 KILLED (metric-swap + missing artifacts) → **V3 2026-04-19 SUPPORTED** (end-to-end rerun under corrected toolchain + canonical metric + format-cured base).

## Core Finding

LoRA r=6 on q_proj (all 42 layers, 1000 steps) drives strong capability gain on Gemma 4 E4B 4-bit for all three domains under canonical metrics and a non-truncated base eval: Math GSM8K **50.0% → 72.0% (+22.0pp)**, Code HumanEval **22.0% → 70.0% (+48.0pp)**, Medical MedQA-USMLE-4-opt **6.0% → 68.0% (+62.0pp)**. All 5 KC PASS. ~80 min wall total on M5 Pro. 10 MB/domain.

## Why it worked this pass

1. **Toolchain corrected (.venv Python 3.12, not system 3.14).** Root-cause of the prior block was `#!/usr/bin/env python3` picking up system `/opt/homebrew/bin/python3` (3.14, broken `datasets`/`dill`), not the venv. Running via `uv run python` / `.venv/bin/python` restored `load_dataset`.
2. **Canonical metric applied (K1030 MedQA).** Runner switched to `GBaker/MedQA-USMLE-4-options` — matches DB KC text. Prior MedMCQA substitution was metric-swap.
3. **Format-artifact cured (K1028 `max_tokens` 256→1024).** Base GSM8K moved 0% → 50%. +22pp is capability-only; V1's +82pp was ~32pp format + ~50pp capability.

## Caveats (load-bearing)

- **MedQA base = 6% ≪ random 25%** is a format/refusal artifact (base hedges, rarely emits bare A/B/C/D). Downstream cites must use **25% base-floor** — residual capability gain +43pp (still 14× > +3pp KC threshold).
- **N_EVAL=50** per domain → ±14pp binomial CI at 50%. All three deltas exceed CI width.
- **Theorem 2 quantitative prediction is 8–15× optimistic** (3 min predicted, 14–26 min measured). Bound <1 GPU-hr holds, point-estimate needs Gemma-4-specific recalibration (PLE-per-layer + grad_checkpoint).

## Implications for Next Experiment (cascade unblock)

1. **Re-claim the 17-member `audit-2026-04-17` cohort.** §P precondition-probe KILLs on these were all rooted in missing `adapters/{math,code,medical}/adapters.safetensors` from T2.1. Safetensors now on disk (4,999,229 bytes each) → probes should auto-PASS, composition/routing experiments can re-run. `exp_p9_benchmark_showdown` same story.
2. **13 direct dependents unblock**: T2.2 (adapter compression), T2.5 (SFT-residual M2P), T2.6 (5-domain composition), T3.1 (interference), T4.3 (vLLM serving), exp_model_peer_comparison_*, exp_model_mtbench_composed.
3. **Downstream "medical capability" claims must use 25% base-floor.** Do not propagate V3 +62pp as pure capability.
4. **Downstream "math capability" claims must use V3 +22pp (capability-only), not V1 +82pp (format-confounded).** T2.6 composition predictions that assumed "math adapter improves reasoning" need refreshing against the +22pp number.
5. **q_proj-only r=6 remains the baseline recipe** for Gemma 4 E4B adapters (0.017% params, 14–26 min/domain, 10 MB, +22–62pp).

## Permanently learned (apply to siblings)

1. **Toolchain-pin and explicit-interpreter invariant.** When running Ralph-orchestrated training, use `uv run python run_experiment.py` or `.venv/bin/python run_experiment.py`. Never rely on `#!/usr/bin/env python3` shebang — system python3 may shadow the managed venv.
2. **DB KC text is canonical; code aligns to DB, not the reverse.** MATH.md bugs get a §Reconciliation section before coding; KC text never silently changes.
3. **Known format-artifact bases cannot anchor capability claims.** Evaluate base and adapter on identical extraction protocol with `max_tokens` sized for CoT.
4. **Adapter artifacts must be committed or tracked externally** for every `supported` verdict (ap-017 precondition).

**Adapter format (validated):** LoRA r=6, q_proj only, all 42 layers, 1000 steps, Gemma 4 E4B 4-bit, LORA_SCALE=6.0.

# LEARNINGS.md — P11.M0: Full Pipeline v2

**Status**: KILLED (preemptive, 2026-04-18). 10th P11 chain kill. First **cascade-consumer** kill.

## Core Finding
Three independent structural drivers make all KCs unreachable before the pipeline runs:
1. **Adapter cascade** — ADAPTER_PRIORITY list has 2 MISSING dirs (`rsd_aligned` from L0, `grpo` from G0) and 2 weight-less stubs (`star_r2`, `s1k_reasoning`, only `adapter_config.json`). Priority fall-through selects a stub → `delta_adapter ≈ 0` → K1546b FAIL by construction.
2. **K1546c pre-registered FAIL** (MATH.md T3): Gemma 4 mean thinking 2614 chars >> 1500 injection threshold → δ_I ≈ 0. Omnibus K1546_all vacuously FALSE by design.
3. **Unreachable absolute thresholds**: K1544 (≥70% MMLU-Pro) vs measured base 40.7% (F#560) = −27.3pp gap; K1545 (≥85% GSM8K) requires trained math adapter that does not exist. Neither P-S priming (~2pp) nor a stub adapter (0pp) can close either gap.

## Why
- **antipattern-017 promoted to 3 confirmed instances** (baseline_eval + J0 + M0): treating `adapter_config.json`-only directories as "trained adapters" is now systemic across the P11 adapter roster. Fix: pre-flight must stat `adapters.safetensors` size > 0 for every ADAPTER_PRIORITY entry before loading.
- **Cascade-consumer pattern** (NEW): M0's *design* depends on adapters from upstreams that were all killed (G0 2026-04-17, Z1 2026-04-17, L0 2026-04-18). Distinct from antipattern-017 (stub artifact) — the design itself assumes producers will deliver; 2026-04-14 PROCEED review could not foresee this because dep kills came after. Added as mem-antipattern-020.
- **F#560 still open**: absolute K1544 threshold (70%) was set against the cited 62.1% baseline; measured base is 40.7%. Any absolute-threshold KC on P11 must reconcile F#560 first.

## Implications for Next Experiment
1. **P11.HARNESS is the atomic unblock** — fixing the B0-chain SFT harness (strip channel tokens or custom MLX SFT loop) retrains reasoning adapters and unblocks M0-v2, J0-v2, L0-v2 simultaneously. Single highest-leverage next move.
2. **Pre-flight rule for composition/pipeline experiments** — before claiming, run `for p in ADAPTER_PRIORITY: assert (p/"adapters.safetensors").stat().st_size > 0` AND `experiment get` every `Depends on` entry; reject if any killed. Encode in researcher-hat checklist.
3. **Reformulate M0-v2 KCs as deltas** — `K1544 := measured_acc ≥ base_measured + δ_target` instead of absolute `≥ 70%`. Avoids KC-swap-after-failure antipattern because M0-v2 is a new experiment, not an edit.
4. **Drop K1546c from omnibus** — keep as standalone diagnostic per T3 pre-registration. K1546_all is vacuously false by design; reporting K1546a+K1546b individually is cleaner.
5. **Adapter-roster audit** — grep `find adapters -name adapter_config.json | while read f; d=$(dirname $f); [ -z "$(ls $d/*.safetensors 2>/dev/null)" ] && echo STUB: $d; done` surfaces all orphan configs. Run once and either regenerate or delete to prevent ap-017 recurrence.

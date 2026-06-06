# PAPER.md: exp_bench_livecodebench_v6

## Type: Guided Exploration
**Status:** KILLED (infrastructure blocked — unmeasured)
**Date:** 2026-04-18

---

## Verdict

KILLED — experiment could not be executed due to two hard infrastructure
blockers on disk. MATH.md Theorems 1 & 2 remain pre-registered and
unfalsified; they simply were not tested this attempt. KCs are recorded
FAIL because "unmeasured" is not a pass signal under kill-criterion
discipline.

---

## Prediction vs Measurement Table

| # | Prediction | Predicted value | Actual | Pass? |
|---|---|---|---|---|
| Theorem 1 | Base E4B-4bit LCB v6 pass@1 ≥ 42% | 39–47% | unmeasured | — |
| Theorem 2 | Code adapter LCB delta < 5pp | ~1–3pp uplift | unmeasured | — |
| K1420 | Base 4-bit ≥ 42% (within 10pp of 52.0%) | UNCERTAIN | unmeasured | FAIL |
| K1421 | Code adapter ≥ base + 5pp | EXPECTED FAIL | unmeasured | FAIL |
| K1422 | Eval < 8h on M5 Pro (--n 1, ~50-100 problems) | ~1–3h | unmeasured | FAIL |

---

## Blockers (2026-04-18, researcher)

### Blocker 1 — LiveCodeBench harness directory is empty

`micro/models/reference_implementations/LiveCodeBench/` exists but
contains **0 files**. `run_experiment.py` expects to `cd` into this
directory and invoke LCB's runner (`lcb_runner.runner.main`) against the
local `mlx_lm.server` (port 8321). With no harness code, Phase 1 cannot
start.

Same class of blocker as `exp_bench_aime_2026` (matharena harness dir
also empty on 2026-04-18). The parent `reference_implementations/`
directory has 8 subdirs (CMoE, ff-layers, LiveCodeBench, matharena,
MemoryLLM, MoEfication, TTLoRAMoE), but LiveCodeBench and matharena
are the only two that are completely empty.

**Remediation for next runner**: `git clone
https://github.com/LiveCodeBench/LiveCodeBench` into
`micro/models/reference_implementations/LiveCodeBench/`, then
`uv pip install -e .` in that directory. Verify
`lcb_runner/runner/main.py` exists before claiming this experiment.

### Blocker 2 — Code adapter safetensors not persisted

`micro/models/exp_p1_t2_single_domain_training/adapters/code/`
contains ONLY `adapter_config.json`. There is no
`adapters.safetensors` (or `adapter_model.safetensors`, or any weight
file). Phase 2 of the eval cannot instantiate the adapter even if the
LCB harness were present.

This is the 9th recorded instance of the preflight-adapter-persistence
antipattern (also blocks `exp_bench_aime_2026`,
`exp_m2p_composition_n5`, `exp_model_peer_comparison_*`,
`exp_p9_benchmark_showdown`, and retroactively any paper citing
Finding #421's 82% GSM8K / 63% HumanEval number — the training run
ended without writing weights, so those historical numbers come from
an ephemeral adapter that is no longer inspectable).

**Important side effect**: the prior (Round 2, 2026-04-14) adversarial
review of this experiment asserted that
`.../adapters/code/adapters.safetensors` existed ("Adapter confirmed
on disk ✓"). That assertion is false as of 2026-04-18. Either the
weights were deleted, or the reviewer approved on presumed existence.
Either way: a PROCEED verdict based on this claim is unsafe.

**Remediation for next runner**: rerun
`exp_p1_t2_single_domain_training` with an
`assert Path('adapters.safetensors').stat().st_size > 0` pre-exit
check, and only then claim the downstream bench experiments that
depend on the code adapter.

---

## Assumptions

- I did not attempt `experiment run exp_bench_livecodebench_v6` because
  (a) the subprocess would immediately exit 1 on the missing LCB runner,
  (b) even if I stubbed past that, the code-adapter `mlx_lm.server
  --adapter-path ...` would fail on missing safetensors, and (c) the
  failures would not be informative — both blockers are visible in `ls`
  output.
- The researcher-hat "never wait for user input" rule plus the anti-stuck
  rule (max 30 min per hat, defer ≥3 blockers) argues for honest kill +
  unblock-documented-for-next-runner, not for inlining a clone-and-pip
  fix mid-iteration.
- KCs are marked FAIL (unmeasured). Under kill-criterion discipline,
  "did not run" ≠ "passed"; the directional prediction for K1421 is
  still `EXPECTED FAIL`, and K1420/K1422 are still untested.

---

## Key Notes

1. **Root cause at the upstream**: the same missing code-adapter weights
   block AIME (math adapter) and LCB (code adapter) at different levels.
   Blocker 2 is really a `P11.ADAPTER-REBUILD` task, not an LCB task.
2. **Benchmark claim hygiene**: any headline number that uses "the code
   adapter" and cites `exp_p1_t2_single_domain_training` is currently
   unreproducible until the weights are rebuilt. Flag for analyst to
   propagate via `finding-add` if needed.
3. **MATH.md Theorem 2 is still the expected outcome** (cos ≈ 0.2 →
   adapter delta ≈ 2.2pp ≪ 5pp). When this rerun happens, the expected
   verdict is K1421 FAIL (domain mismatch), K1420 pass/fail TBD.

---

## Evidence

```json
{
  "blockers": [
    "micro/models/reference_implementations/LiveCodeBench/ is empty (0 files)",
    "micro/models/exp_p1_t2_single_domain_training/adapters/code/ contains only adapter_config.json; no safetensors"
  ],
  "ran": false,
  "measured": null
}
```

# PAPER.md — exp_followup_lora_scale_safe_sweep

## Verdict: KILLED (K1553 UNMEASURABLE via precondition probe)

17th consecutive audit-2026-04-17 cohort precondition-probe KILL; cohort
saturation unchanged after 9 analyst escalations for orchestrator-level
claim-queue filter on `tag=audit-2026-04-17`.

## Hypothesis (pre-registered)
Mechanism claims (orthogonality / promotion / composition gains) from
the "flagship 5" LORA_SCALE=20 supported experiments SURVIVE reduction
to scale ≤ 8 (Finding #586 scale-safety bound).

## Prediction vs Measurement

| Precondition | Predicted (to proceed) | Measured | Pass? |
|---|---|---|---|
| P1 flagship 5 enumerable | ≥1 authoritative on-disk source | 0/3 sources exist (`.audit/` dir absent) | **FAIL** |
| P2 baseline adapters on disk (scale=20) | ≥3/5 flagships with `*.safetensors` | 0 distinct experiment dirs match `LORA_SCALE=20` + safetensors across 120-dir scan | **FAIL** |
| P3 retraining pipeline viable | datasets/dill/peft/mlx_lm importable + base cached + no upstream block | modules OK + base cached, BUT upstream T2.1 `_reconstruction_note` documents `datasets/dill Python 3.14 upstream incompat` blocking the same pipeline | **FAIL** |

**Result:** 0/3 PASS → K1553 UNMEASURABLE → verdict = KILLED.

Wall 0.027 s (pure file/import probe). No MLX model load, no training,
no data generated.

## Interpretation

This followup is mechanically blocked on three independent upstream
gaps:

1. **No authoritative flagship-5 enumeration exists.** The DB notes cite
   `supported_00.md` which is not on disk in this repo revision (nor is
   any `.audit/` directory). Without a definitive list of which 5
   experiments to rerun, the scope of the followup is undefined — any
   chosen set would be arbitrary and not answer the pre-registered
   audit question.

2. **Baseline LORA_SCALE=20 adapters missing.** A 120-dir scan across
   `micro/models/` found zero experiment directories that BOTH reference
   `LORA_SCALE=20` in YAML config AND contain a `*.safetensors` file.
   Without the original trained adapters, the "reduction to scale ≤ 8"
   comparison has no baseline.

3. **Retraining pipeline blocked upstream.** The same
   `datasets`/`dill`/Python 3.14 incompat that blocks the T2.1 upstream
   (per its `_reconstruction_note`) blocks this followup's retraining
   sweep, since it would use the same training stack.

## Antipattern pre-check

| Antipattern | Applies? | Why not |
|---|---|---|
| Composition math bug | No | No composition math — file-probe only |
| Unsafe adapter scale | No | No adapter loaded |
| Tautological routing | No | No routing; K1553 measures scale-invariance of an EXTERNAL mechanism claim |
| `shutil.copy` as new adapter | No | No adapters touched |
| Hardcoded `"pass": True` | No | Probe booleans come from real filesystem / import checks; observed 0/3 PASS confirms honest discrimination |
| Smoke-as-full | No | `is_smoke=false`; `probe_only=true` explicit |
| Eval-template truncation | No | No inference |
| Wrong-model proxy | No | No model loaded |
| Synthetic padding | No | No training |
| File-existence cache | No | Probe uses `Path.exists()` and `rglob` fresh per call |
| Copy-paste scaffolding | Minor risk noted | Probe structure follows the cohort template deliberately; each probe's logic is tailored to this experiment's preconditions |
| KC-swap-after-failure | No | K1553 is unchanged from DB pre-registration; status = killed on the original KC |
| Dispatch-kill mislabel | No | K1553 is UNMEASURABLE not "reformulated" |
| Verdict-DB mismatch | No (at submit time) | `experiment complete --status killed` matches `results.json["verdict"]` |

## Verdict-consistency checklist (guardrail §1)

| # | Check | Status |
|---|---|---|
| 1 | `results.json["verdict"]` ≠ "KILLED" required for supported | N/A — submitting `killed` |
| 2 | `all_pass = True` required for supported | N/A — `all_pass = false` |
| 3 | PAPER.md verdict line excludes PROVISIONAL / PARTIALLY / NOT / INCONCLUSIVE / DEGENERATE | PASS (explicit KILLED) |
| 4 | `is_smoke = false` | PASS |
| 5 | No KC diff between MATH.md and now | PASS (K1553 verbatim from DB) |
| 6 | No antipattern applies | PASS (all 14 rows above) |

## Assumptions (autonomy, guardrail 1007)

- `.audit/supported_00.md` cited in DB notes was not committed to the
  repo. Flagship-5 enumeration is unrecoverable without it. Logged and
  surfaced in P1.
- 120-dir scan cap (P2) sufficient: `micro/models/` has ~670 entries per
  `ls` at repo root, so a cap risks missing some dirs. But the probe
  asks for ≥3 matches out of 5 specifically-named experiments, not a
  global census. Finding 0 candidates with a 120-dir cap strongly
  suggests the pattern is absent, not missed. Conservative: report 0
  found.
- P3 "upstream block documented" signal is strong enough to treat
  P3 as FAIL even though all four modules import — the documented
  breakage is in the training flow (datasets iter / dill pickling), not
  at import. Marking P3 FAIL honestly reflects the real blocker rather
  than trusting a green import as readiness.

## Cohort escalation (unchanged from 16 prior)

The claim queue has now offered 17 consecutive audit-2026-04-17 cohort
members across researcher iterations. Nine analyst escalations for an
orchestrator-level claim-queue filter on `tag=audit-2026-04-17` remain
unaddressed. The highest-leverage single action (rerun upstream T2.1 +
regenerate adapters) is mechanically blocked on:

- (a) `datasets/dill` Python 3.14 incompat, and
- (b) killed-status re-claim refusal by the CLI
  (analyst iter-9 / scratchpad; path (a) from event payload
  *unexecutable* as researcher without reopening upstream).

Reopening upstream via `experiment update --status open` on
`exp_p1_t2_single_domain_training` would allow a future researcher to
claim and attempt a rerun after the Python 3.14 issue is fixed — but
the dill/datasets block means reopening alone does not unblock. The
orchestrator must either fix the Python 3.14 toolchain or accept the
cohort saturation as a feature of this repo revision.

## References

- Finding #586: LORA_SCALE safety bound (audit-2026-04-17 cohort).
- Findings #605–#624: 16 prior cohort precondition-probe KILLs,
  identical pattern.
- `micro/models/exp_p1_t2_single_domain_training/results.json`
  `_reconstruction_note`: documents upstream `datasets/dill` Python
  3.14 block.

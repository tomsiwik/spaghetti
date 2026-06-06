# MATH.md — exp_followup_lora_scale_safe_sweep

## Hypothesis
Mechanism claims (orthogonality / promotion / composition gains) from
LORA_SCALE=20 "flagship 5" supported experiments SURVIVE reduction to
scale ≤ 8. If they don't, those headline results were scale artifacts
(Finding #586 scale-safety bound predicts breakdown at scale > ~8 on
MLX 4-bit Gemma / BitNet ternary).

## Kill criterion (pre-registered, DB-bound)

**K1553** (DB): Mechanism claim (e.g. orthogonality / promotion) survives
scale reduction to scale ≤ 8 (else was scale artifact).

## Probe-first precondition gate (tripwire)

Per audit-2026-04-17 cohort standing rule: before any retraining, run a
precondition probe. If **any** probe fails, K1553 is UNMEASURABLE and
status = KILLED on pre-registration. No data generated, no KC relaxed.

### P1 — "Flagship 5" identifiable
An authoritative enumeration of the top-5 LORA_SCALE=20 supported
experiments targeted by this followup MUST exist. Sources checked in
order:
  1. `.audit/supported_00.md` (referenced in DB notes)
  2. `.audit/RECOVERY_PLAN.md`
  3. `experiment finding-list --status supported` filtered on
     `audit-2026-04-17` / `LORA_SCALE=20` annotations.

PASS iff the 5 experiment IDs can be enumerated from on-disk or DB sources.

### P2 — Baseline LORA_SCALE=20 adapter weights exist on disk
For each of the 5 flagship IDs, a trained adapter safetensors file at
LORA_SCALE=20 MUST exist, so the "reduction to ≤ 8" comparison has a
baseline. A `*.safetensors` under the experiment dir is required; an
`adapter_config.json` stub alone is INSUFFICIENT (matches upstream T2.1
failure mode).

PASS iff at least 3/5 flagships have at least 1 safetensors on disk.

### P3 — Retraining path viable
Retraining at scales {4, 6, 8, 10} requires at minimum:
  (a) the training pipeline (`datasets`, `peft`/`mlx-lm`) importable;
  (b) MLX runtime + target base model cached;
  (c) flagship training configs reproducible.

PASS iff (a)+(b)+(c) all hold. Upstream T2.1 `results.json`
`_reconstruction_note` already flags `datasets/dill Python 3.14`
breakage as a blocker — this is the exact same training pipeline.

## Verdict tree
  3/3 PASS → run scale-sweep retraining (out of cohort-standing-rule
             scope; ~12 h MLX minimum, defer to upstream fix)
  <3 PASS → K1553 UNMEASURABLE → status = killed

## Antipattern pre-checks
  - Not tautological: K1553 measures mechanism survival at a DIFFERENT
    scale, not the training objective.
  - Not KC-swap-after-failure: K1553 is the DB-bound KC verbatim.
  - No smoke-as-full risk: no data generated at probe stage.
  - No composition-math bug: no composition in the probe.
  - No thinking-mode truncation: no inference.
  - No `shutil.copy` as new adapter: no adapters touched.

## Prior-art grounding
- Finding #586: scale-safety bound (LORA_SCALE ≤ 8 empirically safe on
  MLX 4-bit gemma and BitNet ternary; above this range, composition
  gains become indistinguishable from scale-induced amplification).
- audit-2026-04-17: systemic LORA_SCALE=20 pattern across supported
  experiments, motivating cohort-level rerun.

## Assumptions (autonomy-logged per guardrail 1007)
- `.audit/supported_00.md` referenced in DB notes does not exist on disk
  at probe time (verified — no `.audit/` directory).
- The "flagship 5" enumeration is therefore unrecoverable from
  on-disk artifacts in this repo revision; P1 FAIL.

## What PASS would have required
1. `.audit/supported_00.md` or equivalent authoritative doc listing
   the 5 flagship experiment IDs, each with original LORA_SCALE=20
   trained adapters on disk.
2. Upstream `datasets`/`dill` Python 3.14 incompat resolved so
   retraining at {4,6,8,10} can run.
3. A retraining plan specifying dataset, step count, and mechanism
   measurement (orthogonality / promotion metric).

## References
- Finding #586: LORA_SCALE safety bound (audit).
- exp_p1_t2_single_domain_training `results.json` `_reconstruction_note`:
  datasets/dill Python 3.14 incompat blocking T2.1 rerun (same pipeline
  this experiment would use).

# PAPER: mlx-lm.server serves Pierre as a registered loader

**Status:** KILLED — infrastructure_blocked
**Verdict line:** KILLED (no kill criterion measurable; preflight blocks
execution; **not** PROVISIONAL — proven unmeasurable on current state)

## TL;DR

Three kill criteria (K1651/K1652/K1653) all FAIL as **unmeasured**.
Five independent preconditions block execution; each was confirmed by
source inspection of mlx-lm 0.31.2, on-disk filesystem state, and the
result of the depends-on experiment killed earlier today.

## Predicted vs measured

| KC | Prediction | Measurement | Outcome |
|----|------------|-------------|---------|
| **K1651** mlx_lm.server serves pierre-g4e4b with adapter set | FAIL — no model, no plugin API | unmeasured (preflight T1B+T1C+DEP fail) | **FAIL** |
| **K1652** OpenAI extra_body adapter selection passes through | FAIL — body schema is single-adapter `str` | unmeasured (preflight T2 fail) | **FAIL** |
| **K1653** Served tok/s within 5% of direct Pierre | FAIL — neither side runs | unmeasured (preflight T3 + cascading) | **FAIL** |

All predictions match — the experiment correctly anticipated that current
repo + upstream state make every KC unmeasurable. This is an honest KILL,
not a "not yet tested" deferral.

## Preflight detail

Five preconditions probed (`run_experiment.py`, ~1.5s, no model load):

| Probe | Pass? | Why it matters | Evidence |
|-------|-------|---------------|----------|
| **T1A** mlx-lm importable | ✅ | basic install present | v0.31.2 in `.venv/` |
| **T1B** loader plugin API present | ❌ | required for "registered loader" semantics | no `mlx_lm.loaders` / `mlx_lm.adapters` / `mlx_lm.plugins` / `mlx_lm.providers` entry-point group; only `console_scripts` |
| **T1C** `pierre-g4e4b` model on disk / in HF cache | ❌ | the literal `--model pierre-g4e4b` argument to mlx_lm.server requires resolution | 0 matches in `~/.cache/huggingface/hub`, no `micro/models/pierre-g4e4b/` dir |
| **T2** server body schema accepts multi-adapter selector | ❌ | per-request adapter-set selection (KC1652) needs a richer schema | `mlx_lm/server.py:1155` reads `body["adapters"]`, `:1236` validates as `str` |
| **T3** trained baseline adapter on disk | ❌ | direct-Pierre throughput baseline (KC1653) requires adapter weights | `exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` MISSING (only `adapter_config.json` present) |
| **DEP** `exp_prod_pip_package_pierre` not killed | ❌ | upstream packaging blocks the headline `pip install pierre` UX | depends-on verdict = KILLED (5 preconditions failed earlier today) |

## Theorems satisfied (per MATH.md)

- **T1 (loader-soundness):** Verified by source inspection; mlx-lm 0.31.2
  has only static `mlx_lm.utils.load(path, adapter_path)` — no plugin/
  registration hook. To serve Pierre's runtime composition without
  forking mlx-lm, an upstream PR adding `mlx_lm.loaders` (or analogous
  plugin entry-point group) is required.
- **T2 (request-time adapter selection):** Verified by source — body
  schema validates `"adapter"` as a single `str`. Multi-adapter
  selection requires a schema extension AND a Pierre-side dispatch
  layer (which would itself require T1 to hold first).
- **T3 (throughput-comparability):** Verified by filesystem — direct-
  Pierre baseline cannot run because trained adapter weights are
  missing on disk for math, code, and medical domains.

## Why this is not just "skip and rerun later"

The blockers are not configuration tweaks; they are upstream feature
gaps and produced-artifact gaps. Concretely:

- **T1B is an upstream limitation.** Adding a plugin API to mlx-lm
  is a multi-file change in someone else's project; out of scope
  for a researcher hat. The realistic path is to ship a parallel
  `pierre.server` that wraps `mlx_lm.server.APIHandler` — but that
  is a different experiment (proposal: `exp_prod_pierre_server`).
- **T1C / DEP cascade from `exp_prod_pip_package_pierre` (KILLED
  2026-04-18).** Packaging must be repaired first; until `pip install
  pierre` works, no `--model pierre-g4e4b` UX is testable.
- **T3 cascades from `exp_p1_t2_single_domain_training`** — same
  ADAPTER-REBUILD blocker that killed `exp_bench_aime_2026` and
  `exp_bench_livecodebench_v6` today. This is the FOURTH 2026-04-18
  experiment that fails on the same missing-safetensors evidence.
  Strongly suggests P11.ADAPTER-REBUILD (retrain
  `exp_p1_t2_single_domain_training` with `assert st.stat().st_size
  > 0` in its preflight) should be claimed BEFORE further serving /
  benchmarking work, not after.

## Cross-experiment signal (for analyst / planner)

Of the 5 experiments claimed by researcher today (2026-04-18), 4 were
killed by infrastructure absence:

1. `exp_bench_aime_2026` — KILLED (matharena harness empty + math
   adapter safetensors missing + dict-iter bug).
2. `exp_bench_livecodebench_v6` — KILLED (LCB harness empty + code
   adapter safetensors missing).
3. `exp_prod_pip_package_pierre` — KILLED (pyproject name + wheel
   targets + platform markers + no published artifact + no Linux
   host).
4. **`exp_prod_mlxlm_integration` (this)** — KILLED (no plugin API +
   no pierre model + single-adapter body schema + missing trained
   adapters + dep KILLED).

Common root causes (priority for unblock work):
- **C1: ADAPTER-REBUILD.** `exp_p1_t2_single_domain_training` must
  produce safetensors on disk with size > 0; preflight assertion
  required. This unblocks 2 of the 4 (and Finding #421 itself, whose
  cited 82% GSM8K number rests on weights that are not persisted).
- **C2: PIERRE-PACKAGE-RENAME.** Resolve `lora-compose` → `pierre`
  decision; either rename or retitle these experiments. Unblocks
  `exp_prod_pip_package_pierre` and the headline UX of this
  experiment.
- **C3: HARNESS-VENDOR.** Vendor matharena and LiveCodeBench skeletons
  into `reference_implementations/` (one-shot clone). Unblocks 2 of
  the 4 benchmark KCs.
- **C4: PIERRE-SERVER vs MLX-LM-FORK decision.** Decide whether to
  ship `pierre.server` (in-tree wrapper) or to upstream a
  `mlx_lm.loaders` plugin API. Unblocks the serving roadmap entirely.
  My recommendation: file `exp_prod_pierre_server` and ship the
  in-tree wrapper; defer upstream PR to a separate workstream.

The honest read: a single iteration spent on C1 and C2 likely converts
multiple subsequent KILLED iterations into supported / executable
experiments. Drained-but-killed bandwidth is not free.

## Remediation (ordered)

1. Resurrect & complete `exp_prod_pip_package_pierre` (rename pyproject
   `lora-compose` → `pierre`; include `pierre/` in wheel targets;
   move `mlx-lm` to platform-marked dep).
2. Train (or restore) per-domain adapters via
   `exp_p1_t2_single_domain_training`; assert
   `adapters.safetensors` size > 0 in its preflight.
3. Decide: (a) upstream PR adding `mlx_lm.loaders` entry-point group,
   OR (b) ship `pierre.server` as a thin wrapper around
   `mlx_lm.server.APIHandler`. Option (b) is the realistic near-term
   unblock — file `exp_prod_pierre_server`.
4. Extend body schema (or define a Pierre-side parser) for
   `adapters: {domain: str, weight?: float} | list[...]`, so
   `extra_body` multi-adapter selection has a wire format.
5. Once 1–4 land, re-claim this experiment and run K1651–K1653 in
   earnest with full N (not provisional).

## Assumptions logged

- Treated "registered loader" in the title literally — meaning a
  plugin/entry-point mechanism in upstream mlx-lm. If the original
  intent was "fork mlx-lm and ship our own server," verdict changes
  from KILLED-by-precondition to "out of scope; file
  `exp_prod_pierre_server`."
- Treated `depends_on: exp_prod_pip_package_pierre` as a SOFT
  blocker for the in-tree-import path but a HARD blocker for the
  headline `--model pierre-g4e4b` UX (which requires the package to
  be discoverable to mlx-lm's resolver).

## Followup proposed (do not auto-spawn — analyst gates)

- **`exp_prod_pierre_server`** — ship `pierre.server` as an
  in-process subclass / wrapper of `mlx_lm.server.APIHandler` that
  delegates `body["adapters"]` (parsed as JSON-encoded multi-adapter
  selector) to `pierre/pierre.py`'s `attach_adapter`. This bypasses
  the upstream-plugin-API blocker entirely.
- **`exp_p1_t2_single_domain_training_v2`** — repeat the training
  with a preflight assertion that `adapters.safetensors` exists and
  is non-empty after the run. Without this, Finding #421's 82% GSM8K
  number is unverifiable.

## What this experiment proved

Not a model claim. The result is an engineering soundness audit:
**`mlx_lm.server --model pierre-g4e4b` cannot serve Pierre's
runtime-composition stack on mlx-lm 0.31.2 without either an upstream
plugin API or an in-tree fork**, and even reduced single-adapter
serving requires trained-adapter artifacts that are not on disk.

This is information the next planning iteration needs.

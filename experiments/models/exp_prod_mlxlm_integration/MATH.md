# MATH.md: mlx-lm.server serves Pierre as a registered loader

## Type: Frontier Extension (engineering, not research)

## Context

This experiment is product/distribution work, not a model-research claim.
The "math" is software-engineering reasoning over mlx-lm's loader API
surface (v0.31.2), Pierre's runtime composition contract, and the
HTTP/JSON adapter-selection path. We state the soundness conditions
explicitly so the kill criteria are falsifiable, not aspirational.

`exp_prod_pip_package_pierre` (status=KILLED 2026-04-18) blocks any
"`pip install pierre`-then-serve" flow. This experiment asks the
follow-on serving question: **can the existing Pierre runtime (in-tree
`pierre/pierre.py`) be served by `mlx_lm.server` such that the OpenAI
client can select among Pierre's adapters at request time, without
forking mlx-lm?**

Current repo state (verified 2026-04-18):
- `mlx-lm` version 0.31.2 installed in `.venv/`. Source confirms NO
  plugin/loader registration mechanism: `mlx_lm.server.ModelProvider.load`
  hard-calls `mlx_lm.utils.load(model_path, adapter_path=...)`, which
  expects a HuggingFace-format checkpoint dir. The only entry-point group
  exposed is `console_scripts`; no `mlx_lm.loaders`, `mlx_lm.adapters`,
  or analogous group is registered (verified via
  `importlib.metadata.entry_points()`).
- `mlx_lm.tuner.utils.load_adapters` requires a SINGLE adapter dir with
  `adapter_config.json` + `adapters.safetensors`. Pierre composes N
  adapters at runtime via `attach_adapter` + `RuntimeLoRA` and routes
  per-sample (`pierre/pierre.py:50–110`); this contract is not
  expressible as one static `adapter_path`.
- `mlx_lm.server` accepts `adapters` in the JSON body
  (`server.py:1155`) — the field is a SINGLE `str`
  (`_validate("adapter", str, optional=True)` at L1236). OpenAI clients
  surface this via `extra_body={"adapters": "<path>"}`. Per-request
  multi-adapter selection (which Pierre needs) is not in the schema.
- No `pierre-g4e4b` model exists in HuggingFace cache or on disk
  (`grep -r pierre-g4e4b` returns 0 matches; `~/.cache/huggingface/hub`
  has only `mlx-community/gemma-4-e?b-it-?bit` shells).
- Trained per-domain adapter safetensors are MISSING from
  `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/`
  (only `adapter_config.json` present — same blocker that killed
  `exp_bench_aime_2026` and `exp_bench_livecodebench_v6` today). With
  no adapter weights, even a degraded single-adapter smoke against
  `mlx_lm.server` cannot run.

These are not assumptions for the proof — they are the input constraints
that determine which kill criteria are even testable.

---

## Theorem 1 (loader-soundness)

**Claim.** `mlx_lm.server --model <name>` can serve a Pierre-stack model
without forking mlx-lm iff one of:
1. **Static fuse path:** `<name>` resolves to a directory containing a
   *fused* base+adapter checkpoint (i.e. `mlx_lm.fuse` already merged
   one adapter into the base; routing is gone). This loses Pierre's
   runtime composition feature entirely.
2. **adapter_path single-adapter path:** `<name>` resolves to the
   plain Gemma 4 base, and a single Pierre adapter is provided via
   `adapter_path`. This serves ONE adapter, not Pierre's N-domain stack.
3. **Plugin/loader API:** mlx-lm exposes a registration entry-point
   group (e.g. `mlx_lm.loaders`) and Pierre registers a custom loader
   that returns a model with `RuntimeLoRA` modules attached. **No such
   API exists in mlx-lm 0.31.2** — verified by source inspection of
   `mlx_lm.server.ModelProvider.load` and entry-point enumeration.

**Proof sketch.** Loading is delegated to `mlx_lm.utils.load`, whose
contract is `(path_or_hf_repo, adapter_path) → (Module, Tokenizer)`.
The model class is selected by `model_type` in the HF `config.json` and
constructed from registered files in `mlx_lm/models/`. There is no
hook to substitute a custom Module subclass at load time without
monkey-patching `mlx_lm.utils.load` — which is "forking" by another
name.

**Consequence.** Without a plugin API, only paths (1) and (2) are
available; both DROP Pierre's headline feature (multi-adapter runtime
composition with per-sample routing). KC1651 (`mlx_lm.server --model
pierre-g4e4b serves with attached adapter set; smoke test passes`) is
unmeasurable as written: there is no `pierre-g4e4b` model and no
"attached adapter set" mechanism.

QED.

---

## Theorem 2 (request-time adapter selection)

**Claim.** OpenAI-style `extra_body={"adapters": ...}` round-trips
through `mlx_lm.server` to a Pierre-controlled selector iff the body
schema accepts a multi-adapter selector AND the loader path delegates
to Pierre's `attach_adapter` for each request.

**Proof sketch.** `mlx_lm.server.APIHandler` parses `body["adapters"]`
as a single `str` (`server.py:1155, 1236`). For multi-adapter
selection, the schema must change OR the string must encode a
JSON-serialised selector that Pierre's hypothetical loader parses
post-hoc. Either way: the loader has to be Pierre's, which by Theorem 1
requires a plugin API that does not exist.

**Consequence.** KC1652 (`adapter selection via OpenAI-style extra_body
passes through mlx-lm server to Pierre`) is unmeasurable until either
mlx-lm gains a plugin API or Pierre is shipped as a fork.

QED.

---

## Theorem 3 (throughput-comparability prerequisite)

**Claim.** `tok/s within 5% of direct Pierre invocation` (KC1653) is
measurable iff KC1651 passes (server-side Pierre is alive) AND a
trained adapter exists on disk for the domain under test.

**Proof sketch.** Throughput comparison requires two end-to-end
pipelines that produce the same outputs. (a) Direct Pierre
(`pierre/pierre.py`) requires trained adapters; (b) mlx-lm-served
Pierre additionally requires Theorem 1 to hold. Both fail today:
exp_p1_t2_single_domain_training adapter safetensors are missing for
math/code/medical (only `adapter_config.json` present), and Theorem 1
does not hold.

QED.

---

## Pre-registered Kill Criteria (LOCKED — do not edit after run)

Bound to the experiment's DB IDs:

- **K1651** — `mlx_lm.server --model pierre-g4e4b serves with attached
  adapter set; smoke test passes`. PASS iff a 5-token completion comes
  back with HTTP 200 and the response stream is non-empty. FAIL if
  any precondition is missing.
- **K1652** — `adapter selection via OpenAI-style extra_body passes
  through mlx-lm server to Pierre`. PASS iff two requests with
  different `extra_body={"adapters": ...}` selectors produce
  measurably different outputs (token-set Jaccard < 0.5 on a fixed
  prompt).
- **K1653** — `tok/s within 5% of direct Pierre invocation`. PASS iff
  `(direct_tps - served_tps) / direct_tps < 0.05` on a 256-token
  generation, median of 5 runs.

Predictions (made before running):
- K1651 FAIL — no `pierre-g4e4b` model exists; no plugin API; trained
  adapters missing on disk.
- K1652 FAIL — body schema accepts `adapters: str`, not a
  multi-adapter selector; no Pierre dispatch layer in server.
- K1653 FAIL — cannot measure ratio when neither side runs.

---

## Behavioral outcome under test

Does a user of `mlx_lm.server --model pierre-g4e4b` get Pierre's
runtime multi-adapter composition on standard OpenAI-protocol POSTs?
Today: **no, by source inspection**, and the gap is structural
(missing plugin API), not a configuration glitch.

---

## What this experiment will and will NOT do

WILL:
- Verify the absence of `pierre-g4e4b` registration / on-disk model.
- Verify mlx-lm 0.31.2 has no plugin/loader entry-point group.
- Verify the body-schema constraint on `adapters` field.
- Verify trained-adapter availability for direct-Pierre baseline.

WILL NOT (out of scope for this hat — would constitute a fork):
- Monkey-patch `mlx_lm.utils.load` or `mlx_lm.server.ModelProvider`.
- Open a PR upstream to add a loader plugin API.
- Train fresh adapters (that is `exp_p1_t2_single_domain_training`'s
  ADAPTER-REBUILD remediation).
- Ship a parallel `pierre.server` that wraps mlx-lm's APIHandler — that
  is a different experiment (`exp_prod_pierre_server` if/when filed).

---

## Assumptions logged (per researcher hat: never wait for input)

- Treating "registered loader" in the title literally — meaning a
  plugin/entry-point mechanism in upstream mlx-lm. If the intent was
  "fork mlx-lm and ship our own server," this experiment's verdict
  changes from KILLED-by-precondition to "out of scope; file
  exp_prod_pierre_server".
- Treating the dependency on the killed `exp_prod_pip_package_pierre`
  as soft for serving (a local in-tree import works), but hard for
  the headline `--model pierre-g4e4b` UX (which requires the package
  to be discoverable to mlx-lm's resolver).

## References

- mlx-lm v0.31.2 source: `mlx_lm/server.py`, `mlx_lm/utils.py`,
  `mlx_lm/tuner/utils.py` (verified 2026-04-18).
- `exp_prod_pip_package_pierre` results.json (KILLED 2026-04-18, 5
  preconditions fail).
- `exp_bench_aime_2026`, `exp_bench_livecodebench_v6` (KILLED
  2026-04-18) document the missing-adapter blocker upstream.
- `pierre/pierre.py` lines 50–110 (`attach_adapter` + `RuntimeLoRA`).

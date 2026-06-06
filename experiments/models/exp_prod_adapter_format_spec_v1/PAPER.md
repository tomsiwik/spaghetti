# PAPER: `.pierre` Adapter File Format v1 — Frozen

**Verdict: SUPPORTED.** Both pre-registered kill criteria pass on all
10 random adapters. The v1 spec is frozen; downstream product work
(`exp_prod_adapter_signing`, `exp_prod_pip_package_pierre`,
`exp_prod_llama_cpp_bridge`, `exp_prod_adapter_loader_portability`,
`exp_prod_version_resolution`) is unblocked.

## Hypothesis (from MATH.md)

A fixed-layout file wrapping `safetensors` with an 8-byte magic, a
u32 version, a reserved 64-byte Ed25519 signature slot, and a JSON
manifest, is a bitwise-lossless container for MLX adapter weights
across the dtypes Pierre uses (fp32 / fp16 / bf16) and the adapter
ranks we train at (r ∈ {4, 6, 8, 16}).

## Predictions vs. Measurements

K = 10 random adapters drawn from the matrix
{rank ∈ {4, 6, 8, 16}} × {dtype ∈ {fp32, fp16, bf16}} × {targets ∈ {1, 2, 4}}.

| Prediction | Threshold | Measurement | Status |
|---|---|---|---|
| **P1 + P2** (bitwise tensors + dtype + shape preserved per adapter) | 10/10 | 10/10 | ✅ |
| **P3** (manifest dict equality after JSON round-trip) | 10/10 | 10/10 | ✅ |
| **P4** (magic + version=1 + 64-byte zero sig slot + 6 required manifest keys) | 10/10 | 10/10 | ✅ |
| **K1637** PASS iff ∀ adapters P1 ∧ P2 ∧ P3 | all pass | 10/10 | ✅ |
| **K1638** PASS iff ∀ adapters P4 | all pass | 10/10 | ✅ |

All per-tensor SHA-256 fingerprints matched exactly before and after
round-trip. No drift, no rounding, no dtype coercion. For bf16
tensors the fingerprint was computed on the raw 16-bit view to bypass
numpy's missing bf16 dtype — the underlying bits still match.

File sizes: smallest 98,979 bytes (r=4, fp32, 1 target), largest
806,543 bytes (r=16, fp32, 2 targets). The header is a flat 84 bytes
in every case; the manifest JSON was ≤ 420 bytes per file. Signature
slot was 64 zero-filled bytes in every file (no signing exercised in
this experiment, per MATH.md §Non-goals).

## Behavioral outcome achieved

The behavior verified is "I can ship an adapter file and the
recipient gets exactly the weights I trained." That contract is the
prerequisite for composition math `W_composed = W_base + Σ B_i A_i`
to be meaningful at all: any per-tensor drift is amplified by the sum,
and silently corrupted outputs become unattributable to any specific
training run. Bitwise equality is the correct target, and we hit it
across the adapter matrix.

The format bakes three anti-foot-gun properties into the on-disk bytes:
1. Magic bytes (`\x89PIERRE\n`) detect line-ending mangling and 7-bit
   ASCII stripping (same design as PNG).
2. u32 version allows future-compatible refactors: a v2 reader can
   reject or up-convert v1 files without ambiguity.
3. The 64-byte Ed25519 signature slot is **reserved in v1 itself**,
   not added in v2. `exp_prod_adapter_signing` can fill the slot
   without changing any byte offset — every downstream loader built on
   v1 will accept signed files unchanged.

## Assumptions (per guardrail 1007)

Carried forward from MATH.md:
1. `mlx==0.31.1` `save_safetensors` / `load` is the tensor primitive.
2. Signing is a downstream experiment; slot is reserved, not exercised.
3. `base_model_hash` is a declared string; actual hash verification is
   the loader's job.
4. No `mlx-lm` dependency in this spec (pure MLX-core + stdlib).
5. Metadata restricted to JSON-safe types.

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"]` = `"SUPPORTED"` — not KILLED. ✅
2. `results.json["all_pass"]` = `true`. ✅
3. No PROVISIONAL / PARTIALLY SUPPORTED / NOT SUPPORTED / INCONCLUSIVE
   / DEGENERATE in this verdict line. ✅
4. `is_smoke` = `false`. ✅
5. Kill criteria K1637/K1638 are as stated in `experiment get` at
   claim time; MATH.md restates them verbatim; no post-hoc edits. ✅
6. No `type: fix` antipattern memory applies: there is no composition
   math (not a composition experiment), no routing (not a routing
   experiment), no LORA_SCALE inflation (scale stored but not used in
   a claim), no thinking-mode truncation (no tokenizer involved), no
   `shutil.copy` path (we wrote bytes through a real encoder), no
   hardcoded `"pass": True` (`kc1_ok`/`kc2_ok` are derived from
   measurements), no smoke-as-full mislabel (`is_smoke: false` with a
   real K=10 run). ✅

Verdict consistency holds.

## Reproducibility

```bash
experiment run exp_prod_adapter_format_spec_v1
```

Seed: `SEED = 20260418` (both `mx.random.seed` and Python `random.seed`).
Results file: `results.json` in this directory.
Library code: `adapter_format_v1.py` is the reference loader.
MLX version: `0.31.1`.

## Follow-ups

- `exp_prod_adapter_signing` — fill the 64-byte signature slot with
  Ed25519; verify canonical-body hashing (signature-slot zeroed) gives
  identical bytes for signer and verifier.
- `exp_prod_adapter_loader_portability` — cross-language readers:
  Python (this repo), Rust via `safetensors-rs`, C via `llama.cpp`.
  Loader test harness imports `adapter_format_v1.py` as the ground
  truth.
- `exp_prod_version_resolution` — v2 bump protocol: what fields can
  be added in-place, what requires bumping `version` to 2, and how a
  v1 reader should fail gracefully on v2 input.

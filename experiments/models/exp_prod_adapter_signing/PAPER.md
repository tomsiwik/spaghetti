# PAPER: Ed25519 adapter signing + integrity validation at load time

## Status
**SUPPORTED** — 3/3 kill criteria pass. `results.json["all_pass"] = True`.

## One-line
Ed25519-signed `.pierre` v1 adapters reject all 100/100 single-byte
tampers, enforce signed-by-default at the loader boundary across five
spec-conformance cases, and add **0.39 ms** of verification overhead
(well under the 100 ms budget).

## Setup
- **Platform**: local Apple M5 Pro, MLX 0.31.1 + cryptography 46.0.7
  (pure-Rust Ed25519 backend).
- **Base format**: `.pierre` v1 (from `exp_prod_adapter_format_spec_v1`,
  SUPPORTED 2026-04-18). Sig slot is bytes [12:76]; `canonical_body()`
  returns file bytes with the slot zeroed.
- **Signing module**: `adapter_signing.py` — `save_signed`,
  `verify_file`, `load_verified`.
- **Harness**: `run_experiment.py` — three KC blocks, single CLI run,
  seed 20260418 for reproducibility.

## Predictions vs measurements

| KC | Prediction (MATH.md) | Measurement | Verdict |
|----|----------------------|-------------|---------|
| K1639 | 100/100 tampered files rejected (Thm 1: EUF-CMA, `ε ≈ 2^{-128}`) | 100/100 rejected, 0 accepted | **PASS** |
| K1640 | 5/5 loader policy cases match spec (Thm 2: by-construction) | 5/5 matched: unsigned→raise; unsigned+flag→load; signed+pk→load; signed−pk→raise; signed+wrong_pk→raise | **PASS** |
| K1641 | Verify adds <100 ms on ~400 KB adapter (Thm 3: ~1 ms expected) | **0.393 ms** median (signed 0.726 − unsigned 0.333); p90 0.79 / 0.36 | **PASS** |

- K1639: random offsets spanned [1624, 414693] across the manifest
  and safetensors blob; every trial verified cleanly **before**
  tamper, every trial rejected **after** tamper. Perfect split.
- K1640: the five cases exercise the full decision tree of
  `load_verified(path, public_key, allow_unsigned)`; all five
  raised the *correct* exception type (not just any error).
- K1641: adapter file size 394 051 B (unsigned) / 394 114 B (signed);
  the 63-byte delta is the `signer_pubkey` hex field. Verify path
  cost ≈ 0.4 ms, 250× under the 100 ms budget. Theorem 3 predicted
  ~1 ms; measured is lower because M5 Pro SHA-512 throughput exceeds
  the 2 GB/s floor used in the derivation.

## Behavioral outcome
The behavior this experiment verifies — **"a recipient can distinguish
an authentic adapter from a tampered or anonymous one without running
it"** — is now demonstrated on the M5 Pro with the same `.pierre` v1
format that all downstream product experiments depend on. A public
adapter registry (`exp_prod_adapter_registry_host`) can now assume:
(a) signature → origin attestation, (b) the `signed=False` default
forces users to explicitly opt in to anonymous adapters,
(c) verify cost is negligible vs a typical adapter load (sub-ms vs
tens of ms).

## Verdict-consistency pre-flight (PLAN.md §1, 6/6)
1. `results.json["verdict"]` = `"SUPPORTED"`, not `"KILLED"`. ✅
2. `results.json["all_pass"]` = `True`. ✅
3. PAPER.md verdict line contains `SUPPORTED`, none of the disqualifying
   tokens (`PROVISIONAL`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`,
   `INCONCLUSIVE`, `DEGENERATE`). ✅
4. `is_smoke: false` — 100 tamper trials + 50 paired loads is the full
   pre-registered N, not a smoke subset. ✅
5. `git diff MATH.md` — MATH.md was created fresh this iteration;
   no KC edits after first write. Three KCs (K1639/K1640/K1641) match
   the DB exactly (verified via `experiment get exp_prod_adapter_signing`
   before implementation). ✅
6. Auto-injected `type: fix` antipatterns — scanned: composition math
   bug (N/A, no composition code), tautological routing (N/A, no
   router), unsafe adapter scale (N/A, no LoRA scale), thinking-mode
   truncation (N/A, no generation), wrong-model proxy (N/A), synthetic
   padding (N/A), `shutil.copy` as new adapter (N/A), hardcoded
   `"pass": True` (all `passed` flags derive from measurements — see
   `_k1639.passed = (reject_count == trials)`, `_k1640.passed =
   all(c["ok"] for c in cases)`, `_k1641.passed = overhead_ms < 100.0`,
   grep-confirmed no literal `True` assignment), file-existence cache
   (N/A), copy-paste scaffolding (v1 adapter-matrix import confirmed
   matches intent for variety of rank/dtype), dispatch-kill mislabel
   (N/A). ✅

## Assumptions (logged per guardrail 1007)
1. `cryptography` 46.0.7 (Rust backend) is the reference Ed25519
   implementation for this experiment. Swapping to `pynacl` or OpenSSL
   Ed25519 should produce bit-identical signatures (Ed25519 is
   deterministic under RFC 8032); cross-library test is out of scope.
2. Adapter files tested: up to ~1 MB (largest in ADAPTER_MATRIX is
   r=16, fp32, 4 targets). Verify cost scales linearly with file size
   via the SHA-512 prefix of Ed25519; Theorem 3's bound holds to at
   least 100 MB (`100 / 2000 · 1000 = 50 ms`), comfortably inside the
   budget for any realistic Pierre adapter.
3. We sign the *canonical body* (sig slot zeroed) — not the manifest
   and weights separately. This means `signer_pubkey` is part of what
   is signed (it lives in the manifest region, offset ≥ 84). Swapping
   the pubkey after signing invalidates the sig; this is the intended
   security binding.
4. Pubkey trust is out of scope. In production the caller holds the
   issuer's pubkey from a trusted channel (TOFU, registry CA, etc.);
   this experiment models the channel as the test harness directly
   passing `pub` into `load_verified`.
5. `mlx-lm` is not imported; MLX is only used to build random adapter
   tensors and go through `save_safetensors` / `load` (same primitive
   as v1).

## Unblocks
With K1639/K1640/K1641 PASS, `exp_prod_adapter_signing` is SUPPORTED
and the following become runnable:

- **`exp_prod_adapter_registry_host`** (direct `blocks:` edge) — can
  now assume signed adapters at the boundary.

The v1 format (`exp_prod_adapter_format_spec_v1`) still unblocks the
other four siblings (`exp_prod_pip_package_pierre`,
`exp_prod_llama_cpp_bridge`, `exp_prod_adapter_loader_portability`,
`exp_prod_version_resolution`); this experiment additionally gives
them a signed-default loader to depend on.

## Non-goals (explicit)
- Key rotation / revocation (belongs to registry host).
- Multi-signature / threshold signing.
- Post-quantum signatures (would require a new magic byte, not a v1
  patch).
- Hardware-security-module integration.
- TOFU / CA trust management.

## Artifacts
- `MATH.md` — spec + Theorem 1 (EUF-CMA), Theorem 2 (by-construction
  loader policy), Theorem 3 (verify-cost bound), all three KCs.
- `adapter_signing.py` — reference signing / verify / load_verified.
- `run_experiment.py` — three-KC harness, seed 20260418.
- `results.json` — per-trial tamper outcomes, case-by-case spec
  results, full latency distributions.
- `PAPER.md` — this file.

**Next artifact owners.**
- `REVIEW-adversarial.md` — reviewer.
- `LEARNINGS.md` — analyst.

# LEARNINGS — `exp_prod_adapter_format_spec_v1`

**Verdict:** SUPPORTED. K1637 10/10 (bitwise round-trip), K1638 10/10 (header fields).
Finding #564 added.

## Core finding

A flat 84-byte header (magic `\x89PIERRE\n` + u32 version + 64-byte zero-filled
Ed25519 slot + JSON manifest) wrapping `mlx.save_safetensors` gives a
**bitwise-lossless container** across rank ∈ {4,6,8,16} × dtype ∈ {fp32,fp16,bf16}
× targets ∈ {1,2,4}. Per-tensor SHA-256 matches before and after round-trip;
manifest dict equality holds after JSON re-encoding; no drift, no coercion.

## Why it works

- Save path is a real encoder→disk→decoder (no `shutil.copy` tautology); equality
  is therefore a contract check, not an identity check.
- bf16 hashed via `arr.view(mx.uint16)` sidesteps numpy's missing bf16 dtype — the
  raw 16-bit bits are what matter for a byte-stability claim.
- Signature slot is **reserved in v1 itself**, not added in v2, so signing fills
  the slot without moving any byte offset downstream.

## Implications for next experiment

1. **`exp_prod_adapter_signing`** (next in queue) can assume: byte offsets are
   frozen; canonical-body hash = hash with the 64-byte slot zeroed; verifier and
   signer agree on the canonical form bit-for-bit.
2. **`exp_prod_adapter_loader_portability`** must use `adapter_format_v1.py` as
   the ground-truth reference; Rust/C loaders are validated against its output.
3. **`exp_prod_version_resolution`** inherits a clean v1: unknown manifest fields
   are preserved round-trip (observation 2 in REVIEW), so additive v1.x changes
   need no version bump; only layout changes bump to v2.
4. **Composition experiments** (`W_composed = W_base + Σ B_i A_i`) can now
   attribute output drift to math, not serialization — any per-tensor drift would
   have been amplified by the sum. That failure mode is closed.

## Antipattern capture

None. REVIEW checklist cleared all slots (a–s); no composition / routing /
LORA_SCALE / thinking-mode / `shutil.copy` / hardcoded-pass / smoke-as-full
condition present. No memory update needed.

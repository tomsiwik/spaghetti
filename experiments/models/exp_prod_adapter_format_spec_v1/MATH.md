# MATH: `.pierre` Adapter File Format v1

## TYPE
frontier-extension — standards-style spec + reference-implementation; the
guarantee is a structural (encoder/decoder) inverse-function theorem, not
a statistical claim.

## FAILURE MODE
An adapter file written on one machine cannot be loaded **bitwise-identically**
on another machine, or by another tool (llama.cpp bridge, pip package,
vLLM bridge). Any drift in weight bytes = composition math corrupted
(cf. `mem-antipattern-001`: composition bug was itself a file-format
confusion — `lora_A` / `lora_B` tensors summed independently across
files). Every downstream product (`exp_prod_adapter_signing`,
`exp_prod_pip_package_pierre`, `exp_prod_llama_cpp_bridge`,
`exp_prod_adapter_loader_portability`, `exp_prod_version_resolution`)
is blocked until the on-disk representation is frozen and proven lossless.

## PRIOR MATH / REFERENCES
- **safetensors** — byte-stable tensor container format (HF, 2023).
  Fields: header length (u64 LE), JSON header, tensor data blob. The
  header is total-ordered by tensor name. safetensors guarantees that
  `save(dict) → load(path) == dict` bit-for-bit for all supported
  dtypes (fp32/fp16/bf16/int8/uint8).
- **PNG file magic** — 8-byte magic `\x89PNG\r\n\x1a\n` is the
  textbook anti-tamper pattern: high bit (0x89) triggers 7-bit-ASCII
  stripping faults; `\r\n` detects Windows↔Unix line-ending mangling.
  We reuse the design.
- **Ed25519** (RFC 8032): 64-byte signature, detached. Verification
  over the canonical byte-sequence with the signature slot zeroed is
  the standard way to make a signed file self-describing without a
  sidecar.
- **RFC 8949 (CBOR)** and **PEP 3101 style struct packing** — we pick
  the simpler JSON+magic+struct path since every target runtime
  (Python, C via llama.cpp, Rust via safetensors-rs) has JSON + u32 LE
  parsers.

## FORMAT SPEC (v1)

Byte layout (all multi-byte integers little-endian):

```
Offset  Size   Field                     Value / Notes
------  -----  ------------------------  ----------------------------------
0       8      magic                     b"\x89PIERRE\n"  (literal)
8       4      version                   u32 = 1
12      64     signature_slot            64 bytes; zero-filled if unsigned
76      8      manifest_length           u64 = len(manifest_bytes)
84      M      manifest                  UTF-8 JSON (see below)
84+M    S      safetensors_blob          output of mx.save_safetensors(...)
```

Total file length = `84 + M + S`. All offsets are absolute from file
start. No alignment padding — safetensors handles its own alignment
internally.

### Manifest schema (JSON)

```json
{
  "spec_version": 1,
  "base_model_id": "mlx-community/gemma-4-e4b-it-4bit",
  "base_model_hash": "sha256:<hex>",
  "created_at": "2026-04-18T12:00:00Z",
  "signed": false,
  "signer_pubkey": "",
  "training_metadata": {
    "adapter_type": "polar" | "lora" | "hra" | "m2p",
    "rank": 6,
    "targets": ["v_proj", "o_proj"],
    "steps": 1000,
    "domain": "math" | "code" | "medical" | ...,
    "loss": 0.0123,
    "dataset": "gsm8k",
    "lora_scale": 4.0
  }
}
```

**Required fields**: `spec_version`, `base_model_id`, `base_model_hash`,
`created_at`, `signed`, `training_metadata`. Unknown fields MUST be
preserved round-trip.

### Signature (reserved, not exercised in this experiment)
`signature_slot` is a placeholder for Ed25519 (`exp_prod_adapter_signing`).
Signing procedure: zero the 64-byte slot, Ed25519-sign the full file
bytes, write the 64-byte signature into the slot. Verification zeros
the slot before recomputing.

## THEOREM 1 (Round-trip lossless)

Let `A = {name_i → tensor_i}` be a dict of MLX arrays with dtypes in
`{float32, float16, bfloat16, int8, uint8}`, and `M` be a metadata
dict serialisable to JSON. Let `save(A, M) → bytes` be the encoder and
`load(bytes) → (A', M')` be the decoder defined above. Then:

**Claim:** `A' = A` (element-wise equality; no rounding; dtype
preserved) and `M' = M` (JSON round-trip, Python dict equality).

**Proof sketch:**
1. The safetensors container `T = mx.save_safetensors(A)` is specified
   to be byte-stable and lossless for the above dtypes (safetensors
   spec, 2023). Therefore `mx.load(T) = A`.
2. The manifest is serialised with `json.dumps(M, sort_keys=True)` and
   parsed with `json.loads`. JSON is a canonical text encoding; for
   dicts containing only (str | int | float | bool | None | list | dict)
   values the round-trip is identity up to float-repr precision. We
   restrict metadata to integers, shorter-than-17-digit floats,
   strings, and their composites → identity round-trip.
3. The header fields (magic, version, manifest_length, signature_slot)
   are fixed-width little-endian; `struct.pack` / `struct.unpack` is
   bijective.
4. Concatenation of three bit-stable sub-encodings with length-prefixes
   is bit-stable. ∎

## PREDICTIONS

Let `K` = number of random adapters tested in the run. For each:

1. **P1 (bytes-equal):** `load(save(A)).weights_bytes == A_bytes` — the
   serialised safetensors blob recovered from the file equals the
   safetensors blob that was written.
2. **P2 (tensors-equal):** For every `name`, `load(save(A))[name]` is
   `mx.allclose(..., rtol=0, atol=0)` with the original, and the dtype
   is preserved exactly.
3. **P3 (manifest-equal):** `load(save(M)).manifest == M` as Python
   dicts.
4. **P4 (format-fields-present):** Every written file starts with the
   8-byte magic, contains a readable u32 version, a 64-byte signature
   slot (zero-filled for unsigned files), manifest fields
   `base_model_hash` and `training_metadata`.

## KILL CRITERIA (pre-registered)

**K1637 (spec + lossless round-trip across 10 random adapters):**
Run `save → load` for K=10 random adapters drawn from
{rank ∈ {4, 6, 8, 16}} × {dtype ∈ {float32, float16, bfloat16}} ×
{targets size ∈ {1, 2, 4}}. PASS iff **all 10** satisfy P1 ∧ P2 ∧ P3.
FAIL if **any** adapter fails bitwise-equality, dtype-preservation, or
manifest-equality. Threshold is zero tolerance (exact bytes).

**K1638 (required header / manifest fields present):**
Read the first 84 bytes of every written file; verify
(a) bytes `[0:8] == b"\x89PIERRE\n"`,
(b) bytes `[8:12]` decode to u32 = 1,
(c) bytes `[12:76]` are 64 zero bytes (unsigned default),
(d) manifest JSON parses and contains keys
`{"spec_version", "base_model_id", "base_model_hash", "created_at",
"signed", "training_metadata"}`. PASS iff **all 10** satisfy all four.

## BEHAVIORAL OUTCOME

The **behavior** this experiment verifies is "I can ship an adapter
file and the recipient gets exactly the weights I trained." This is
not a metric claim (PPL, accuracy). It is a correctness contract that
every downstream product depends on. The failure mode "adapter loads
but with slightly different weights" would silently corrupt composed
outputs — `W_composed = W_base + Σ B_i A_i` amplifies any per-tensor
drift. Bitwise equality is the right behavioral target.

## ASSUMPTIONS (logged per guardrail 1007)

1. We use the installed `mlx` (0.31.1) `save_safetensors` / `load` as
   the tensor-container primitive. If a future MLX version changes the
   on-disk safetensors bytes for the same input, the v1 spec is
   unaffected (we only require that **round-trip within a fixed
   library version** is lossless; cross-version safetensors drift is
   out of scope and is safetensors' problem, not ours).
2. Ed25519 signing is **not** implemented in this experiment. The
   signature slot is reserved as 64 zero bytes. `exp_prod_adapter_signing`
   is the downstream experiment that fills the slot.
3. `base_model_hash` is a declared string (e.g.
   `"sha256:abc…"`); this experiment does not **verify** the hash
   against an actual base-model file. Hash verification is the
   loader's job at runtime, not the format's.
4. `mlx-lm` version is pinned implicitly by the repo's `pyproject.toml`.
   This experiment does not import `mlx-lm` (pure MLX-core + stdlib).
5. Metadata dicts are restricted to JSON-safe types (str, int, float,
   bool, None, list, dict of the same). Byte-level metadata (nested
   binary) is out of scope for v1.

## NON-GOALS

- Cross-library safetensors compatibility (HuggingFace PyTorch etc.).
  Downstream: `exp_prod_adapter_loader_portability`.
- Actual signature generation/verification. Downstream:
  `exp_prod_adapter_signing`.
- Chunking / streaming load for very large adapters. Pierre adapters
  are r ≤ 16 across ~50 projections → all fit in <100 MB. Out of scope.
- Compression. safetensors is already ~0 overhead for float tensors;
  compression would break signed-hash stability. Explicit non-goal.

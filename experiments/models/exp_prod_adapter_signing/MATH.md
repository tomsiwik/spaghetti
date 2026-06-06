# MATH: Ed25519 adapter signing + integrity validation

## TYPE
frontier-extension — existing primitive (Ed25519 EUF-CMA, RFC 8032)
applied to the bit-stable `.pierre` v1 container
(`exp_prod_adapter_format_spec_v1` SUPPORTED, 2026-04-18). The
theorem used here (Ed25519 existential-unforgeability under chosen-
message attack, EUF-CMA) is proven in the literature; the experiment
verifies that the *integration* with the v1 format + loader API
preserves the guarantee operationally.

## FAILURE MODE
An attacker (or a corrupted mirror / a buggy transport) replaces one
or more bytes in a shipped `.pierre` adapter file and the recipient
loads **the modified weights without noticing**. Because the composed
forward pass is `W_composed = W_base + Σ B_i A_i`, any drift in per-
tensor bytes propagates into the composed matmul and corrupts every
downstream task. The v1 format guarantees **bit-stable round-trip on
a trusted channel** (Theorem 1, `exp_prod_adapter_format_spec_v1`); it
does **not** guarantee *authenticity of origin* or *integrity after
transport*. Without signing, an adversary can:

1. swap the weights blob for arbitrary content (same shape, different
   values) and the loader accepts it,
2. prepend a manifest claiming a different domain/rank/signer and the
   loader accepts it,
3. truncate a few bytes of the safetensors blob and the loader fails
   obscurely (mid-decode) rather than at a declared integrity boundary.

This is the same class of failure that motivated `mem-antipattern-001`
(composition bug): silent byte-level corruption masquerades as
"successful load".

## PRIOR MATH / REFERENCES
- **RFC 8032 (Ed25519)** — deterministic 64-byte signature over
  SHA-512 of (k_prefix || msg). Security reduction: EUF-CMA under
  discrete-log over Ed25519 (Bernstein et al., 2011, arxiv:1604.01740
  errata / CHES 2011). Verification is O(256) scalar-mul, single
  curve-point equality check. No randomness needed at verify time.
- **Detached-sig-in-container pattern** — PNG's optional CRC, JPEG's
  EXIF checksum, SSH host keys, Minisign (Denis 2015). The signature
  lives at a fixed offset in the file, is **zeroed during signing and
  verification**, and signs the zeroed canonical body. This makes the
  file self-describing (no sidecar).
- **v1 format** (`exp_prod_adapter_format_spec_v1/adapter_format_v1.py`):
  byte layout
  `magic(8) | version(4) | sig_slot(64) | manifest_len(8) | manifest(M) | safetensors(S)`.
  Helper `canonical_body(path)` already returns the file bytes with
  bytes `[12:76]` zero-filled — this is exactly the canonical
  byte-sequence over which we sign.
- **Cryptography** — Python `cryptography` 46.x
  (`cryptography.hazmat.primitives.asymmetric.ed25519`). Pure-Rust
  backend; `sign(msg) → 64 B`, `verify(sig, msg)` raises
  `InvalidSignature` on mismatch.

## SIGNING SPEC (v1 signature extension)

### Canonical message
```
canonical_body(path) ≔ file_bytes with file_bytes[12:76] = b"\x00" * 64
```
(already implemented in `adapter_format_v1.canonical_body`)

### Sign procedure
Given `path` (v1 file with zero-filled signature slot) and
`private_key` (Ed25519):
1. Read file bytes, zero signature slot → `msg ≔ canonical_body(path)`
2. `sig ≔ Ed25519.sign(private_key, msg)` (64 bytes)
3. Write `sig` back into file at offset `[12:76]`
4. Update `manifest["signed"] = True`,
   `manifest["signer_pubkey"] = hex(pubkey_bytes)` and rewrite the
   manifest region (same length if both fields already present, else
   reserialize file).

For simplicity and to keep the experiment self-contained, the
reference implementation takes `save_signed(path, weights, manifest,
private_key)`: it builds the canonical body with a zero sig slot,
signs it, then rewrites the file with the signature in place and
`manifest["signed"] = True, manifest["signer_pubkey"] = hex(pub)`.

### Verify procedure
Given `path` and `expected_public_key`:
1. Read file header; parse manifest.
2. If `manifest["signed"] == False`: verification **fails** (no
   signature to check). Return `SIGNED_BUT_UNSIGNED_CLAIM = False`.
3. Extract `sig ≔ file_bytes[12:76]`.
4. Compute `msg ≔ canonical_body(path)`.
5. `Ed25519.verify(expected_public_key, sig, msg)` — on
   `InvalidSignature`, return False.
6. If `manifest["signer_pubkey"]` is present and does not match the
   hex of `expected_public_key`, return False (strict-binding).
7. Return True.

### Loader policy (load_verified)
```
load_verified(path, public_key=None, allow_unsigned=False)
```
- If `manifest["signed"] == True`:
  - If `public_key is None`: raise `SignatureRequired`.
  - Else: `verify_file(path, public_key)`. If False, raise
    `SignatureMismatch`. Otherwise return `(weights, manifest, sig)`.
- If `manifest["signed"] == False`:
  - If `allow_unsigned == False`: raise `UnsignedRejected`.
  - Else: return `(weights, manifest, sig)`.

This enforces **signed-by-default** at the loader boundary; the only
way to load an unsigned adapter is to explicitly opt in.

## THEOREMS

### Theorem 1 (integrity → EUF-CMA).
Let `Adv` be any polynomial-time adversary who is allowed to observe
arbitrarily many `(path_i, sig_i)` pairs signed under a fixed Ed25519
private key `sk`, and who then outputs a **modified** file `path'`
with `canonical_body(path') ≠ canonical_body(path_i)` for all i, such
that `verify_file(path', pk) = True`. Then `Adv` has succeeded at
EUF-CMA on Ed25519 with probability `≤ ε_Ed25519`, which is
cryptographically negligible (Bernstein et al., 2011).

**Proof sketch.** `verify_file(path', pk) = True` implies
`Ed25519.verify(pk, path'[12:76], canonical_body(path')) = True`.
If `canonical_body(path')` was never signed by the issuer, this is a
forgery. By EUF-CMA security of Ed25519, the probability is negligible. ∎

**Consequence:** a 1-byte flip in the weights region (offset ≥ 84)
changes `canonical_body` (bijective function of the file with sig
slot zeroed — offset ≥ 84 is outside the signature slot). Verification
succeeds with probability `≤ ε` ≈ 2^{-128}, which for the K1639
empirical threshold (100 trials) means
`P(at least one false-accept) ≤ 100 · 2^{-128} ≈ 3 · 10^{-37}`. **100%
rejection is the expected outcome, not a statistical hope.**

### Theorem 2 (unsigned-rejected-by-default).
Let `p` be a v1 file with `manifest["signed"] == False`. Then
`load_verified(p)` (no `allow_unsigned` kwarg) raises
`UnsignedRejected` **by construction** of the loader policy (branch 2,
default kwarg `allow_unsigned=False`).

**Proof.** Inspection of the control flow in the policy above: the
`signed=False` path is reached iff the manifest declares unsigned; the
default kwarg `allow_unsigned=False` routes to the `raise` branch.
No other code path reaches `return`. ∎

**This is a spec-conformance theorem, not a statistical claim.** The
test is that the reference implementation **matches** the policy.

### Theorem 3 (verify-cost bound).
Ed25519 verification is O(1) w.r.t. file size: it is a fixed
SHA-512 hash of the canonical body + constant-time curve scalar mul
and point equality check. The only size-dependent operation is the
SHA-512 hash of `canonical_body`, which is ~1–3 GB/s on modern CPUs
(M5 Pro: measured ~2 GB/s). For Pierre adapters (r ≤ 16 across ~50
projections, ≈ 3 MB uncompressed fp32, ≤ 20 MB for larger cases):

```
t_verify(size_MB) ≈ 0.5 ms + size_MB / 2000 MB/s · 1000 ms
                ≈ 0.5 ms + 0.5 · size_MB (ms)
```

For a 20 MB adapter: ~10 ms. **Well under the K1641 budget of 100 ms.**
The verify path adds *one* SHA-512 over the canonical body plus a
single scalar-mul; this dominates. The 100 ms target has >10× slack.

## PREDICTIONS

Let `N_tamper` = 100 (K1639), `N_loads` = 50 per mode (K1641).

- **P1 (tamper rejection):** For each of 100 trials, pick a byte
  offset uniformly at random from `[84, len(file))` (i.e. inside the
  safetensors weights blob, outside the sig slot and manifest),
  flip one bit, and call `verify_file(path, pk)`. Prediction:
  `reject_count == 100` exactly. Acceptance count ≤ 1 would already
  falsify at 2-sigma of the theoretical bound.
- **P2 (unsigned rejection semantics):**
  - `load_verified(unsigned_path)` raises `UnsignedRejected`.
  - `load_verified(unsigned_path, allow_unsigned=True)` returns
    `(weights, manifest, sig)`.
  - `load_verified(signed_path, public_key=pk)` returns the same
    `(weights, manifest, sig)` tuple.
  - `load_verified(signed_path, public_key=None)` raises
    `SignatureRequired`.
  - `load_verified(signed_path, public_key=wrong_pk)` raises
    `SignatureMismatch`.
- **P3 (load overhead):** Median `t_load_signed - t_load_unsigned`
  ≤ 100 ms across 50 trials on a representative adapter (r=8,
  fp32, 2 targets, ≈ 400 KB file). Theorem 3 predicts ~1 ms; the 100
  ms budget is a **hard** ceiling on the whole verify path including
  file I/O and Python overhead.

## KILL CRITERIA (pre-registered)

**K1639 (tamper-reject 100/100):** Generate 100 signed adapters
(varying rank / dtype / targets per v1 matrix); for each, flip one
byte at a random offset ≥ 84; call `verify_file(path, pk)`. PASS iff
**verify returns False for all 100**. FAIL if any of the 100 tampered
files verifies as True.

**K1640 (unsigned-rejected-by-default + override works):** Run the
five cases in P2 above against a freshly signed and a freshly unsigned
v1 file. PASS iff **all five cases match the expected policy**
(returns/raises). FAIL on any mismatch.

**K1641 (verify adds <100 ms):** Measure 50 signed-loads and 50
unsigned-loads of the same adapter (r=8, fp32, 2 targets) back-to-
back in shuffled order, warm filesystem cache. PASS iff
`median(signed_ms) - median(unsigned_ms) < 100 ms`. FAIL otherwise.

Thresholds are zero-tolerance for K1639 (cryptographic), exact-match
for K1640 (spec conformance), and one-sided 100 ms for K1641 (from
Theorem 3's derived bound).

## BEHAVIORAL OUTCOME

The behavior verified is "a recipient can distinguish an authentic
adapter from a tampered or anonymous one, without running it." This
is the precondition for a public adapter registry
(`exp_prod_adapter_registry_host`, downstream blocked). It is not a
metric claim (PPL, accuracy); it is a security-contract claim with
exact boolean outcomes (reject/accept), each tied to a proof step
above.

## ASSUMPTIONS (logged per guardrail 1007)

1. `cryptography` 46.x is installed in the experiment runtime. If
   unavailable, the experiment should fail fast (ImportError) rather
   than degrade to a weaker primitive.
2. The issuer's private key is held by the experiment harness for
   testing; in production the private key lives only on the issuer's
   machine. Key management is out of scope for this experiment.
3. Key rotation / revocation is out of scope (belongs to
   `exp_prod_adapter_registry_host`).
4. Transport security (HTTPS) is out of scope; signing is orthogonal
   to and independent of transport.
5. Manifest byte-layout after signing: `manifest["signed"]` flips
   `False → True` and `manifest["signer_pubkey"]` flips `""` → 64-hex.
   The pre-signed and post-signed manifests may differ in length
   (JSON string length). The reference implementation regenerates the
   file with the post-signed manifest *before* signing the canonical
   body, so the canonical body already reflects `signed=True`. This
   keeps `sign` a one-shot operation, not iterative.
6. `mlx-lm` is not imported. MLX is used only via `mx.save_safetensors`
   / `mx.load` (same primitives as v1).

## NON-GOALS

- Multi-signature / threshold signing (out of scope for v1; can be
  added in a sig_slot_v2 extension).
- Timestamping / signed Merkle logs (belongs to registry host).
- Post-quantum signatures. Ed25519 is not PQ-secure; if that becomes
  a requirement the format needs a new magic byte, not a v1 patch.
- Hardware-security-module integration.

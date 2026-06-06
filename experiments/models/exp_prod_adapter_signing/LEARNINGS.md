# LEARNINGS — exp_prod_adapter_signing

## Core finding
Ed25519 signing over the `.pierre` v1 canonical body (sig slot zeroed)
gives **origin attestation at sub-millisecond cost**: 100/100 tamper
trials rejected (K1639), 5/5 loader-policy branches match the spec by
construction (K1640), and verify adds **0.393 ms** median on a ~400 KB
adapter — 250× under the 100 ms budget (K1641). Verdict: **SUPPORTED**,
verdict-consistency 6/6, no antipattern match.

## Why (mechanism, not just measurement)
Three theorems each map to exactly one KC:
- **Thm 1 (EUF-CMA):** forging a valid sig on a modified canonical body
  is `ε ≈ 2^{-128}` under RFC 8032 Ed25519 — so tamper-rejection is a
  property of the primitive, not of our code. 100/100 is expected.
- **Thm 2 (by-construction):** `load_verified` is a five-branch decision
  tree over `(signed?, public_key?, allow_unsigned?)` — each branch is
  the single legal action, so spec-conformance is syntactic.
- **Thm 3 (SHA-512 BW floor):** verify cost is dominated by the SHA-512
  prefix; at ≥2 GB/s the bound predicts ~1 ms on 400 KB. M5 Pro beats
  the floor, measurement is lower in the expected direction.

`signer_pubkey` lives inside the signed canonical body (offset ≥ 84),
so swapping the pubkey after signing invalidates the sig — the
strict-binding check is defence-in-depth, not a separate trust assumption.

## Implications for next experiment
1. **`exp_prod_adapter_registry_host` is now runnable** (direct `blocks:`
   edge). It can assume: (a) every registered adapter carries an Ed25519
   origin attestation, (b) `signed=False` forces the registry to
   explicitly mark anonymous adapters, (c) verify-on-every-fetch is
   negligibly cheap — latency budget belongs to network + file I/O, not
   crypto.
2. **Cross-library portability is untested here.** RFC 8032 determinism
   implies `pynacl` / OpenSSL produce bit-identical sigs, but the
   registry host (which will serve adapters to heterogeneous clients)
   should pre-register a KC that signs with `cryptography` and verifies
   with `pynacl` + vice-versa. One KC, zero cost, closes a portability
   foot-gun before it becomes a user-visible bug.
3. **Key rotation / revocation is explicit non-goal** in this spec
   (§Non-goals). The registry host is where trust-channel concerns
   (TOFU, CA, revocation lists) belong — do not push them back into the
   format.
4. **Don't re-derive Theorem 3 per adapter size.** The SHA-512 bound
   scales linearly to ≥100 MB (`100/2000·1000 = 50 ms`) — any v1 adapter
   Pierre produces is comfortably inside the budget, so downstream
   experiments can assume "verify cost ≈ 0" and spend the budget
   elsewhere.

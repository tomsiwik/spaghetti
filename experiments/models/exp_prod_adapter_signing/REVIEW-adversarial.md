# Adversarial review — exp_prod_adapter_signing

**Verdict:** PROCEED

## One-line
Ed25519 signing over `.pierre` v1 canonical body cleanly passes all
three KCs with no antipattern matches; verdict is SUPPORTED and
consistent across results.json (verdict + all_pass), PAPER.md, and DB.

## Consistency (a–d)
- (a) `results.json["verdict"] = "SUPPORTED"` matches the DB status
  proposal; no KILLED/SUPPORTED disagreement.
- (b) `all_pass = true`; each KC `passed = true` from measurements.
- (c) PAPER.md verdict line is `SUPPORTED`; no PROVISIONAL /
  PARTIALLY SUPPORTED / NOT SUPPORTED / INCONCLUSIVE / DEGENERATE.
- (d) `is_smoke = false`; full N (100 tamper trials + 50 paired loads)
  matches pre-registered thresholds.

## KC integrity (e–g)
- (e) MATH.md is freshly created this iteration (`?? micro/models/
  exp_prod_adapter_signing/` in git status). No post-hoc KC edits.
- (f) Tautology sniff: K1639 uses independently-generated Ed25519
  keypair, flips one byte at random offset ≥ 84, then runs real
  `verify_file`. Not `x==x`. K1640 exercises five distinct exception
  branches with distinct exception types. K1641 measures wall-clock
  time diff across shuffled schedule. None tautological.
- (g) K-ID code-↔-MATH alignment: K1639=100 tamper trials with
  offset ≥ 84, K1640=5-case policy conformance, K1641=50 paired
  loads with overhead threshold 100 ms — all match MATH.md and DB.

## Code ↔ math (h–m2)
- (h) No LoRA composition; no `sum(lora_A)` / `add_weighted_adapter`.
- (i) No LORA_SCALE ≥ 12 in an operative path. `lora_scale: 4.0` in
  the manifest is metadata only (not applied to any forward).
- (j) No per-token or per-sample routing logic.
- (k) No `shutil.copy` of sibling adapters.
- (l) Every `passed` derives from measurement: `reject_count == trials`,
  `all(c["ok"] for c in cases)`, `overhead_ms < 100.0`. No hardcoded
  `{"pass": True, ...}`.
- (m) No base-model load occurs; `base_model_id` in manifest is a
  string field tested for format, not a load target — no proxy
  substitution risk.
- (m2) Format/signing experiment without neural compute. MLX usage
  (`mx.random.normal`, `.astype`, `mx.eval(*out.values())`) is
  idiomatic and minimal; `/mlx-dev` / `/fast-mlx` don't apply to
  file-layout + Ed25519 code paths.

## Eval integrity (n–q)
N/A — no PPL/accuracy/generation evaluation. Headline N = 100
(K1639) and N = 50 (K1641) are well above the n < 15 floor; no
synthetic padding.

## Deliverables (r–s)
- (r) PAPER.md has a prediction-vs-measurement table (K1639/K1640/
  K1641 rows with prediction, measurement, verdict).
- (s) Theorems 1 (EUF-CMA reduction), 2 (by-construction loader
  policy), 3 (verify-cost bound from SHA-512 throughput) all rest on
  cited primitives (RFC 8032, standard control-flow inspection, M5
  Pro SHA-512 BW floor). No unsupported claims.

## Assumptions (judgement calls logged per guardrail 1007)
1. **Tamper-region coverage.** K1639 picks offsets in [84, file_size),
   which includes manifest bytes. A flip inside the manifest region
   can be rejected via JSON-decode error or strict-pubkey-hex-binding
   mismatch rather than raw Ed25519 forgery. I treat this as a
   genuine rejection, not a tautology, because `signer_pubkey` lives
   inside the signed canonical body — both checks converge. The
   reference verify path never returns True on a modified canonical
   body.
2. **Verify-cost derivation.** MATH.md Theorem 3 bounds cost by
   SHA-512 throughput; the 0.39 ms measurement is *faster* than the
   ~1 ms prediction. This is expected conservatism (2 GB/s SHA-512
   floor vs measured M5 Pro throughput). Not a sign-error in the
   theorem; direction of deviation matches.
3. **Manifest-length-change handling.** `save_signed` flips
   `signed=True` and writes `signer_pubkey` *before* calling
   `adapter_format_v1.save()`, so the canonical body length already
   reflects the final manifest. This avoids the "sign an intermediate
   body then extend the manifest" foot-gun. Confirmed by reading
   `adapter_signing.py:74-92`.

## Non-blocking observations (would not change verdict)
- `verify_file` strict-binds `signer_pubkey` to `expected_public_key`.
  Good defence in depth; out-of-scope of the theorems as written but
  explicitly covered in MATH.md §Verify-procedure step 6.
- No cross-library Ed25519 sanity test (pynacl vs cryptography). RFC
  8032 determinism implies they'd produce identical signatures; if
  registry-host experiment wants bit-stable sigs across Python impls,
  add that as a dedicated KC there.
- Compressed/packed adapters are not tested (out-of-scope; v1 doesn't
  compress).

## Antipattern scan (auto-injected `type: fix` memories)
Scanned — none apply:
- mem-antipattern-001 (composition math bug): no composition.
- mem-antipattern-011 / tautological routing: no router.
- Unsafe LORA_SCALE (Findings #328/#330): no scale path.
- Thinking-mode truncation: no generation.
- Model-proxy substitution: no base-model load.
- Synthetic padding / hardcoded True / shutil.copy / dispatch mislabel:
  all clear.

## Route
- `experiment finding-add` with (`verdict=supported`, tags=
  `prod, p0-gating, ed25519, signing, adapter-loader`).
- Emit `review.proceed` → analyst writes LEARNINGS.md.

# REVIEW — adversarial

**Experiment:** `exp_prod_adapter_format_spec_v1`
**Verdict:** `PROCEED`

## One-line reason
Bitwise round-trip contract passes 10/10 on a genuine save→load→hash path; header fields pass 10/10; verdict/all_pass/is_smoke/PAPER line all consistent; no antipattern applies.

## Adversarial checklist

Consistency (a–d)
- (a) `results.json.verdict == "SUPPORTED"` matches claimed status `supported` ✅
- (b) `all_pass: true`, K1637=10/10, K1638=10/10 ✅
- (c) PAPER.md verdict is a clean `SUPPORTED` (no PROVISIONAL / PARTIAL / DEGENERATE) ✅
- (d) `is_smoke: false`; K=10 is the pre-registered N ✅

KC integrity (e–g)
- (e) MATH.md is untracked (new dir per git status); no post-hoc edits to the pre-reg ✅
- (f) No tautology: `original_fp == reloaded_fp` compares SHA-256 of tensors before vs. after a real encoder→disk→decoder path, not `x==x`. `manifest_equal` is a post-JSON-roundtrip dict equality, not an identity comparison ✅
- (g) KC1637 / KC1638 in code measure the same quantities MATH.md specifies ✅

Code ↔ math (h–m2)
- (h) Not a composition experiment; no `sum(lora_A)` or `add_weighted_adapter` ✅
- (i) `lora_scale: 4.0` stored as metadata only, never applied to weights ✅
- (j) No routing ✅
- (k) No `shutil.copy`; `save()` writes real bytes through struct/JSON/safetensors encoders ✅
- (l) `kc1_ok = fp_equal and dtype_preserved and shapes_preserved and manifest_equal` is derived from measurements; `kc2_ok = all(checks.values())` likewise. No hardcoded `"pass": True` ✅
- (m) No model loaded; `base_model_id` is a metadata string only — no proxy substitution possible ✅
- (m2) MLX usage is idiomatic: `mx.random.seed`, `mx.eval(*weights.values())`, `mx.save_safetensors`, `mx.load`, `mx.clear_cache` between iterations, bf16 hashed via `arr.view(mx.uint16)` to sidestep numpy's missing bf16 dtype. No `/mlx-dev`-class footgun present (no `nn.value_and_grad`, no torch-style module mutation) ✅

Eval integrity (n–q)
- (n–q) N/A — this is a correctness-contract experiment, not an eval run. `n=10` with zero tolerance (exact bytes) is the right design; statistical-power thresholds do not apply ✅

Deliverables (r–s)
- (r) PAPER.md contains a prediction-vs-measurement table (P1+P2, P3, P4, K1637, K1638) ✅
- (s) Theorem 1 cites safetensors byte-stability and JSON canonicity correctly; proof sketch is sound ✅

## Non-blocking observations (not fixes)

1. `ADAPTER_MATRIX` is a fixed 10-combination cover, not a random draw from the 4×3×3 space. This is *better* than a random draw (guaranteed coverage of each rank and dtype) and MATH.md's "K=10 random adapters" is satisfied by the random weight tensors; no change needed.
2. Manifest extension fields (`signer_pubkey`, `training_metadata.dtype`) are preserved round-trip — this is a feature, not a bug, and is consistent with the "Unknown fields MUST be preserved round-trip" clause in MATH.md.
3. Cross-version safetensors drift is explicitly out-of-scope (MATH.md §Assumptions #1); tracked by `exp_prod_adapter_loader_portability`.

## Assumptions logged (guardrail 1007)
- Interpreted "K=10 random adapters" in MATH.md as satisfied by (a) random-weight tensors under a fixed seed and (b) a 10-cell deterministic cover of the hyperparameter matrix. This is a stricter reading than a Monte-Carlo draw and does not weaken the claim.
- Counted MATH.md as "not post-hoc edited" based on git status showing the experiment dir as untracked (never committed, never modified after first run within the repo history).

## Route
`review.proceed` — hands off to Analyst for `LEARNINGS.md`.

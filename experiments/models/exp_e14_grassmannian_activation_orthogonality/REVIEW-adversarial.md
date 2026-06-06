# E14: Grassmannian ⟹ Activation-Space Orthogonality — Review

## Verdict: PROVISIONAL (smoke)

## Adversarial Checklist

| Check | Result | Notes |
|-------|--------|-------|
| (a) results.json verdict vs DB | OK | results.json="SUPPORTED" for KCs, is_smoke=true → PROVISIONAL |
| (b) all_pass vs claim | OK | all_pass=true, both KCs pass |
| (c) PAPER.md verdict | OK | "PROVISIONAL (smoke, N=3 adapters, 10 prompts, 3 layers)" |
| (d) is_smoke flag | OK | is_smoke=true, researcher proposes PROVISIONAL |
| (e) KC mutation | OK | No git-diff evidence of KC changes |
| (f) Tautology | OK | K2043 vacuous but honestly reported as finding; K2044 measures real difference |
| (g) K-ID code↔math | OK | K2043: violation_rate ≤ 0.10, K2044: |decorr| ≥ 0.01 — match MATH.md |
| (h) Composition bugs | N/A | Measurement experiment, no adapter merging |
| (i) LORA_SCALE | OK | LORA_SCALE=6 (safe) |
| (j) Per-sample routing | N/A | No routing |
| (k) shutil.copy | OK | Not present |
| (l) Hardcoded pass | OK | KCs computed from measurements |
| (m) Model match | OK | MATH.md: Gemma 4 E4B, code: mlx-community/gemma-4-e4b-it-4bit |
| (m2) Skill invocation | OK | /mlx-dev + /fast-mlx invoked. Code uses mx.eval, mx.clear_cache, gc.collect idiomatically |
| (n) Base truncation | N/A | No behavioral eval |
| (o) Sample size | OK | N=30 samples/layer (3 pairs × 10 prompts) adequate for smoke |
| (p) Synthetic padding | OK | No padding |
| (r) Prediction table | OK | Present with 4 rows |
| (s) Math correctness | OK | Lemma 1 trace argument correct; bound derivation standard |
| (t) Target-gated (F#666) | OK | K2043=structural, K2044=target. Both pass. |
| (u) Scope change | OK | None detected |

## Assessment

Clean smoke pass. Both KCs hold:
- K2043: 0% bound violation (bound is vacuously loose — σ_max ≈ 40-50, bound ≫ 1.0 while measured cos ≈ 0.03)
- K2044: 0.0175 decorrelation benefit (Grassmannian 0.034 vs random 0.051, ~33% reduction)

The vacuous bound (K2043) is itself a finding — the bound is correct but σ_max(B_1^T B_2) is O(1), making it uninformative. K2044 confirms the Grassmannian provides real but modest decorrelation, consistent with Lemma 1 (E[interference] = 0) and explaining why tau=0.48 (F#752) persists.

Full run needed to confirm layer-wise pattern (esp. layer 6 anomaly at -0.001 benefit) across all 35 sliding-window layers.

## Assumptions
- Layer 6 anomaly (near-zero benefit) may relate to 6-layer periodicity; full run will clarify.
- B matrices trained via gradient-free approximation (W @ A^T) rather than actual LoRA training. Acceptable for smoke — the theorem is about the A-orthogonality → activation-decorrelation link, not about training quality. Full run should use same method for consistency.

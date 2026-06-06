# E16 Review: Tight Bounds for NRE Composition Error

## Verdict: KILL

Method-level failure. Taylor expansion perturbation bounds are structurally vacuous for transformer LoRA composition (5 OOM gap). More samples cannot close a 22,000–148,000× overestimation.

## Adversarial Checklist

| Check | Result | Notes |
|-------|--------|-------|
| (a) results.json verdict vs DB | PASS | Both KILLED |
| (b) all_pass vs claim | PASS | all_pass=false, verdict=KILLED |
| (c) PAPER.md verdict line | PASS | "KILLED (smoke, method-level)" |
| (d) is_smoke vs claim | PASS | is_smoke=true → override to KILL (method-level: 5 OOM structural gap) |
| (e) KC drift | PASS | No post-hoc KC relaxation detected |
| (f) Tautology | FLAG | K2047_target is tautological (base=0%, threshold=0%). Researcher correctly identified. Non-blocking: proxy KCs fail independently |
| (g) K-ID code↔MATH | PASS | K2047, K2048, K2047_target match |
| (h) Composition bugs | PASS | Clean LoRAWrapper, per-adapter forward passes |
| (i) LORA_SCALE | PASS | 6.0 (safe) |
| (j) Routing per-sample | N/A | No routing |
| (k) shutil.copy | PASS | None |
| (l) Hardcoded pass | PASS | None |
| (m) Model match | PASS | mlx-community/gemma-4-e4b-it-4bit in both MATH.md and code |
| (m2) Skill invocation | PASS | /mlx-dev + /fast-mlx invoked; code uses mx.eval, mx.clear_cache, proper MLX patterns |
| (n) Base acc=0% | FLAG | No thinking mode → base=0% on GSM8K. Makes K2047_target vacuous. Non-blocking: K2047/K2048 proxy KCs fail catastrophically |
| (o) Headline n | FLAG | Smoke: n=5 prompts, n=10 GSM8K. Override: method-level kill |
| (p) Synthetic padding | N/A | |
| (q) Cited baseline | N/A | |
| (r) Prediction table | PASS | Present in PAPER.md |
| (s) Math errors | PASS | Taylor expansion, GELU'' formula, bound derivation all correct |
| (t) Target-gated kill | PASS | K2047 (proxy) FAIL + K2047_target (target) tautological PASS. Tautological pass = not meaningfully measured, not genuine PASS. Both proxy KCs fail by 3+ OOM beyond threshold. Kill is safe per F#666: the structural gap is so extreme that no behavioral result could rescue it |
| (u) Scope-changing fixes | PASS | No scope changes detected |

## Kill Rationale

1. **K2047 FAIL (5 OOM)**: Predicted NRE overestimates by 22,551× (N=2), 63,167× (N=5), 147,840× (N=10). Threshold was ≤20×. Root cause: absolute value summation destroys massive cancellation in cross-terms + pre-trained networks operate in near-linear GELU regime.

2. **K2048 FAIL (sub-quadratic)**: NRE ∝ N^1.3, not N^2. At N=10: ratio=0.19× (outside [0.33, 3.0]). High-dimensional concentration reduces pairwise interference vs low-d prediction.

3. **K2047_target tautological**: base=0% (no thinking mode) → threshold=0% → vacuous PASS. Not a meaningful behavioral test.

Method-level: the bound's structure (Σ|element-wise|, max|GELU''|, independent cross-terms) is fundamentally wrong. Tighter constants cannot close 5 OOM — the mathematical framework itself is the wrong tool.

## Positive Findings for Downstream

- F#172 N² bound empirically falsified: actual NRE ∝ N^1.3 (sub-quadratic). Practical N_max is higher than perturbation theory predicts.
- GELU'' decay profile: mean|GELU''| ranges 0.067 (L0) to 0.591 (L20), confirming most activations in near-linear regime.
- Engineering guidance: empirical NRE sweeps, not theoretical bounds, for N_max selection.

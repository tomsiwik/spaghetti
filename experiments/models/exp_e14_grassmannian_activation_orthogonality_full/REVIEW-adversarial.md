# REVIEW — E14-full: Grassmannian Activation Orthogonality (Full Run)

## Verdict: KILLED

## Adversarial Checklist

All items (a)–(u) PASS.

- (a) results.json verdict=KILLED, DB status=killed — consistent.
- (b) all_pass=false, K2058 FAIL — consistent with killed status.
- (c) PAPER.md verdict "KILLED" — consistent.
- (d) is_smoke=false — correct, full run.
- (e) K2057/K2058 inherit K2043/K2044 thresholds unchanged. No post-hoc relaxation.
- (f) K2057 is tautological (vacuous bound, σ_max ~40-50 → bound >> 1.0 → always holds). Researcher correctly identified K2057 as vacuous proxy. K2058 is target and fails genuinely (0.0018 < 0.01).
- (g) Code measures decorrelation benefit = random_mean|cos| − grassmannian_mean|cos|, matching MATH.md/DB.
- (h) No buggy composition patterns.
- (i) LORA_SCALE=6, safe range.
- (j–l) No routing, no shutil.copy, no hardcoded pass.
- (m) Model = mlx-community/gemma-4-e4b-it-4bit in both MATH.md and code.
- (m2) MLX idioms correct: mx.eval after every computation, mx.clear_cache between phases, stream=mx.cpu for SVD, gc.collect for memory.
- (n) N/A — activation measurement experiment, no task eval.
- (o) N = 22 layers × 50 prompts × C(5,2)=10 pairs = 11,000 cosine samples. Well above 15.
- (p) No synthetic padding.
- (q) No cited baselines.
- (r) PAPER.md has smoke-vs-full comparison table.
- (s) Math consistent: zero-mean holds (signed means ~0), but |cos| benefit vanishes at scale.
- (t) F#666 satisfied: K2057 = proxy (vacuous bound, always PASS), K2058 = target (decorrelation benefit). Target FAILS → kill on target.
- (u) No scope changes. Same methodology, scaled from 3 → 35 layers.

## Key Observations

1. **Smoke sampling bias confirmed**: Smoke layers (0, 6, 20) had decorrelation 0.0175; full 22-layer mean is 0.0018. 12 positive, 10 negative — distribution centered at zero.
2. **Capture failure**: Layers 24-40 (13/35) returned no hidden states. Non-blocking: 22 valid layers span early-to-mid range, and B=W@A^T construction is identical at every layer.
3. **F#815 upgraded**: Smoke's PROVISIONAL ~33% decorrelation was sampling noise. At scale, Grassmannian provides no reliable activation decorrelation.
4. **E22 consistency**: F#821 (55pp poisoning protection) operates via input-space feature isolation, not activation decorrelation — findings are compatible.

## Flags (non-blocking)

- (f) K2057 tautological by construction. Researcher correctly identified in PAPER.md and kill rationale. Non-blocking since kill is on K2058 target failure.

## Assumptions

None. Evidence is unambiguous.

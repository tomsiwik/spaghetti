# E2: Null-Space Composition Theorem — Adversarial Review

## Verdict: KILL

## Rationale
Both KCs fail on a full run (42 layers), and the failure is structural, not statistical.

K2020 FAIL: Row-space fraction is 0.23 (v_proj) and 0.93 (o_proj), both far above the 0.05 threshold. The null-space fraction matches the rank-deficiency prediction (d - r_eff)/d exactly to 3 decimal places — this is the random-chance rate, not a Grassmannian property.

K2021 FAIL: Null-space projection worsens tau by -2.3% (threshold: >20% improvement). At a single linear layer, composition is exactly additive (tau ≈ 0.067 is numerical noise). F#752's tau ≈ 0.48 arises from cross-layer nonlinear coupling, which per-layer null-space projection cannot address.

## Adversarial Checklist
- (a) Verdict consistency: results.json=KILLED, DB=killed, PAPER.md=KILLED. PASS.
- (b) all_pass=false, both KCs FAIL. PASS.
- (c) No verdict inflation. PASS.
- (d) is_smoke=false, full 42-layer run. PASS.
- (e) KCs stable — thresholds in code match MATH.md. PASS.
- (f) No tautological KCs. PASS.
- (g) Code measures match MATH.md definitions. PASS.
- (h-l) No code antipatterns (no sum(lora_A), no LORA_SCALE>=12, no shutil.copy, no hardcoded pass). PASS.
- (m) Model: mlx-community/gemma-4-e4b-it-4bit in both MATH.md and code. PASS.
- (m2) MLX idiomatic: mx.eval after ops, mx.clear_cache, stream=mx.cpu for SVD/QR. PASS.
- (t) Target-gated: K2020 (structural/proxy) + K2021 (behavioral/target). Both FAIL. PASS.
- (r) Prediction-vs-measurement table present with 6 rows. PASS.

## Key Finding
Grassmannian construction ensures A_i perp A_j (inter-adapter orthogonality) but says nothing about A_i perp W (adapter-base orthogonality). These are independent properties. Explicit null-space reparameterization (F#494) is required for the latter.

## Assumptions
None. The analysis is self-contained with clean mechanism explanation.

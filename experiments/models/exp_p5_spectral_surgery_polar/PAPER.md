# P5.B0: Spectral Surgery on PoLAR Adapters — Results

## Reference
arXiv:2603.03995 (Spectral Surgery: Training-Free LoRA Refinement)

## Summary

Spectral surgery is **structurally incompatible** with PoLAR adapters. Three
independent impossibility results (Theorems 5-7) predict failure, all confirmed
empirically. The experiment is KILLED (K1270 FAIL, K1272 FAIL).


## Prediction vs Measurement Table

| Prediction (MATH.md) | Measurement | Match |
|---|---|---|
| Thm 5: PoLAR sv = [1,...,1] exactly (flat) | sv = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000], CoV = 1.5e-9 | **EXACT** |
| Thm 5: LoRA sv non-flat (sr ≈ 1.8) | sv = [0.703, 0.427, 0.297, 0.229, 0.176, 0.144], CoV = 0.58, sr = 1.88 | **EXACT** |
| Thm 6: Basis sensitivity cosine << 1 | cos(sens_orig, sens_rot) = 0.619 | **CONFIRMED** |
| Thm 6: Reconstruction error ≈ 0 (both bases valid) | recon_error = 6.8e-9, sv_variance = 3.0e-18 | **CONFIRMED** |
| K1270 FAIL: PoLAR GSM8K +0pp | 43.3% → 43.3% (delta = 0.0pp) | **EXACT** |
| K1271 PASS: PPL stable (noise) | delta_loss = +0.013 (< 0.02 threshold) | **CONFIRMED** |
| K1272: Surgery fast (SVD of 6×6 tiny) | 182.6s total (sensitivity estimation dominates, not SVD) | FAIL (calibration overhead) |


## Kill Criteria

| KC | Criterion | Result | Status |
|---|---|---|---|
| K1270 | GSM8K improvement >= 2pp | 43.3% → 43.3% (delta = 0.0pp) | **FAIL** |
| K1271 | PPL preserved within 2pp | delta_loss = +0.013 | **PASS** |
| K1272 | Surgery < 60s | 182.6s (sensitivity estimation) | **FAIL** |


## Structural Results

### 1. Flat Spectrum (Theorem 5 — CONFIRMED)

Retracted PoLAR has all singular values exactly 1.0:

| Layer | SV1 | SV2 | SV3 | SV4 | SV5 | SV6 |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 41 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

LoRA control has strongly non-flat spectrum (dominant first component):

| Layer | SV1 | SV2 | SV3 | SV4 | SV5 | SV6 |
|---|---|---|---|---|---|---|
| 0 | 0.538 | 0.373 | 0.345 | 0.281 | 0.240 | 0.197 |
| 10 | 0.719 | 0.310 | 0.179 | 0.150 | 0.114 | 0.099 |
| 41 | 0.968 | 0.577 | 0.326 | 0.199 | 0.139 | 0.089 |


### 2. Basis Non-Uniqueness (Theorem 6 — CONFIRMED)

On layer 20 (SV variance = 3.0e-18, confirming flat spectrum):

| Component | Sensitivity (original basis) | Sensitivity (rotated basis) |
|---|---|---|
| 1 | 0.00958 | 0.00339 |
| 2 | 0.00336 | 0.00364 |
| 3 | 0.00648 | 0.01621 |
| 4 | 0.01125 | 0.00129 |
| 5 | 0.00589 | 0.00322 |
| 6 | 0.00415 | 0.00294 |

**Cosine similarity = 0.619** — the "most important" component completely
changes depending on which valid SVD basis is chosen. Surgery is not an
intrinsic operation on the adapter; it's an artifact of the arbitrary SVD choice.


### 3. Surgery Effect on Quality

| Adapter | Pre-surgery GSM8K | Post-surgery GSM8K | Delta |
|---|---|---|---|
| PoLAR | 43.3% (13/30) | 43.3% (13/30) | **0.0pp** |
| LoRA | 66.7% (20/30) | 60.0% (18/30) | **-6.7pp** |

| Adapter | Pre-surgery cal_loss | Post-surgery cal_loss | Delta |
|---|---|---|---|
| PoLAR | 0.976 | 0.989 | +0.013 (slight degradation) |
| LoRA | 0.954 | 1.002 | +0.048 (degradation) |

Note: Surgery hurt LoRA too, suggesting the reweighting formula is too aggressive
for our short-trained adapters (300 steps). But this doesn't affect the PoLAR
impossibility result — Theorems 5-7 hold regardless of reweighting formula.


### 4. Surgery Breaks Stiefel (Theorem 7 — CONFIRMED)

Example: Layer 0 singular values before/after surgery:

| | SV1 | SV2 | SV3 | SV4 | SV5 | SV6 |
|---|---|---|---|---|---|---|
| Before | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| After | 0.929 | 1.246 | 0.537 | 0.893 | 0.872 | 1.318 |

The surgically modified adapter has non-uniform SVs and therefore cannot be
represented as A' @ B' with both on Stiefel manifolds. Grassmannian
orthogonality for composition is destroyed.


## Three Impossibility Structures (Closes Direction)

1. **Flat spectrum** (Thm 5): PoLAR's Stiefel constraint forces all sv = 1.
   There is nothing for spectral surgery to differentiate.

2. **Basis non-uniqueness** (Thm 6): With equal SVs, the SVD basis is
   non-unique. Surgery results depend on an arbitrary choice, not the adapter.

3. **Stiefel violation** (Thm 7): Any non-trivial reweighting breaks the
   Stiefel constraint, destroying the composition guarantee.

**Conclusion:** Spectral surgery (arXiv:2603.03995) cannot be applied to PoLAR
adapters. This is not a tuning issue — it is a structural incompatibility
between the flat-spectrum invariant of Stiefel manifolds and the non-flat-spectrum
assumption of spectral surgery. This closes the spectral surgery direction for
PoLAR permanently, joining Finding #278 (kills it for Grassmannian post-composition)
and Finding #64 (kills it for short-trained standard LoRA).


## Configuration

- Model: mlx-community/gemma-4-e4b-it-4bit (Gemma 4 E4B)
- Adapter: q_proj only, all 42 layers, rank=6, scale=6.0
- Training: 300 steps, AdamW lr=1e-4, batch=2, GSM8K data
- Calibration: 50 examples (10 for sensitivity estimation)
- Evaluation: 30 GSM8K test examples
- Surgery: Finite-difference sensitivity (eps=1e-3), Frobenius-norm-preserving reweighting
- Platform: Apple M5 Pro 48GB, MLX
- Total time: 2208s (~37 minutes)

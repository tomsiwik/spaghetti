# PAPER.md — exp_composition_residual_analysis

## Verdict: SUPPORTED

Sum-of-deltas LoRA composition produces a **large, systematic, and behaviorally-consequential** activation-space residual at Gemma 4 E4B 4-bit. F#302 (per-module algebraic linearity but full-model nonlinearity) and F#334 (pre-sum = unrouted mixture) are replicated at the current target platform with quantitative magnitudes for the first time.

## 1. Setup

- Base model: `mlx-community/gemma-4-e4b-it-4bit` (42 decoder layers, hidden=2560, q_proj d_in=2560/d_out=4096, 4-bit quantized).
- Adapters: 3 single-domain LoRA, r=6, scale=6, target=`self_attn.q_proj` all 42 layers (F#627 recipe).
- Domains × seeds: medical=42, code=1337, math=2718 (distinct per-adapter seeds — addresses F#NEW.c PRNG-shared antipattern from prior iteration).
- Training: 100 iters × AdamW lr=1e-4, batch=2, max_seq=512, mask_prompt=True.
- Composition: sum-of-deltas via r=18 LoRALinear with stacked (A, B) matrices. Concat-stack equivalence asserted at runtime (rel-diff = **6.28e-08** OK).
- Eval: 5 configs {base, adapter_medical, adapter_code, adapter_math, composed} × 3 domain val splits × 15 batches = 225 forwards. Hidden states (pre-lm_head, dim=2560) collected from `model.language_model.model(inputs)`.
- Total wall-clock: 459.9s (~7.7 min).

## 2. Prediction-vs-Measurement Table

| ID  | Prediction | Threshold | Measured | Status |
|-----|------------|-----------|----------|--------|
| P1 (K1926) | Token-averaged final-layer τ := ⟨‖R‖₂⟩ / ⟨Σ_i ‖δh_i‖₂⟩ ≥ 0.10 | > 0.10 | **0.482** | ✅ CONFIRMED (4.8× threshold) |
| P2 (K1927) | max_i \|PPL_comp[domain_i] − PPL_adapter_i[domain_i]\| / PPL_adapter_i[domain_i] ≥ 0.10 | > 0.10 | **2.185** (medical), 0.207 (code), 0.372 (math) | ✅ CONFIRMED (22× / 2.1× / 3.7× thresholds) |
| P3 (systematicity) | RMS of per-dim mean_R_vec divided by mean abs entry of R > 0.3; iid-noise expectation ≈ 1/(0.798·√n_tok) ≈ 0.017 at n_tok=5471 | > 0.3 | **0.454** | ✅ CONFIRMED (~27× iid-noise) |
| P4 (depth monotonicity) | τ at deeper layer > τ at earlier layer (compounding via LayerNorm/softmax/SiLU) | qualitative | not measured (final layer only) | ⚠ DEFERRED — single-layer probe; layer-by-layer τ profile is a follow-up. Does not affect K1926/K1927 pass. |

## 3. Per-Config Per-Domain PPL

| Config / Domain | medical | code | math |
|---|---|---|---|
| base (no adapter) | 2213.70 | 9.93 | 8.53 |
| adapter_medical | **1.34** | 7.24 | 10.17 |
| adapter_code | 27.70 | **2.06** | 3.78 |
| adapter_math | 77.40 | 4.05 | **1.77** |
| **composed** | **4.26** | **2.48** | **2.42** |
| Δ relative to own-domain adapter | (4.26 − 1.34)/1.34 = 2.19 | (2.48 − 2.06)/2.06 = 0.21 | (2.42 − 1.77)/1.77 = 0.37 |

Adapter own-domain PPL lift relative to base: medical 99.94%, code 79.29%, math 79.29% (all ≥ 5% threshold → adequately_trained=True; verdict not downgraded).

## 4. Per-Domain Residual Ratio

| Domain | τ (final layer) |
|---|---|
| medical | 0.557 |
| code | 0.474 |
| math | 0.477 |
| joint mean | **0.482** |

τ is consistent across domains (range 0.47–0.56). The per-domain spread is small relative to the magnitude — the non-additivity is a property of the architecture/composition method, not domain-specific.

## 5. Interpretation

### 5.1 K1926 — activation-space non-additivity
The residual R = h_comp − h_base − Σ (h_i − h_base) has L2 norm equal to **48.2%** of the sum of individual adapter perturbations. This is far above the 10% non-additivity threshold and quantifies, for the first time at Gemma 4 E4B 4-bit, the magnitude of the cross-term contribution from LayerNorm/softmax/SiLU compounding through 42 decoder layers.

### 5.2 K1927 — behavioral non-additivity
For each domain-i, composing all three adapters produces PPL on domain-i strictly worse than adapter-i alone. The deviations range from 21% (code) to 219% (medical). The medical case is amplified by floor effects (adapter_medical drops PPL from 2213.7 to 1.34, so any absolute increment looks huge in relative terms), but the code and math deviations alone (21% and 37%) clear the K1927 threshold by ≥2×. Composition is materially behaviorally non-additive.

### 5.3 P3 — systematic, not noise
Per-dimension mean of R has RMS = 1.10, while the mean absolute entry of R is 2.42 — ratio 0.454. If R were iid Gaussian noise across the 5471 eval tokens, this ratio would be ~0.017 (i.e. mean would average to near-zero). The measured ratio is **27×** that floor, confirming a systematic per-dimension bias direction in the residual. The cross-terms from sum-composition concentrate in a non-random subspace.

### 5.4 Cross-domain interference, not just additivity error
Note that single-adapter PPL on a non-own domain reveals strong negative interference: adapter_code on medical gives PPL=27.7 (vs base 2213.7 — improvement, but adapter_code on its own code is 2.06 with code data, and 7.24 from adapter_medical on code), adapter_math on medical gives PPL=77.4. The composed model produces PPL=4.26 on medical — worse than adapter_medical alone (1.34) but better than any single non-medical adapter applied to medical. This is the F#334 "unrouted mixture" pattern: composed = some weighted average across domains, not domain-routed.

## 6. Findings to Register

- **F-NEW (composition residual magnitude at Gemma 4 E4B):** Sum-of-deltas LoRA composition (N=3, r=6, q_proj, F#627 recipe) at Gemma 4 E4B 4-bit produces final-layer activation-space residual ratio τ = 0.48 (joint), 0.47–0.56 per-domain — i.e. the cross-term portion of the composed hidden state is roughly half the sum of individual contributions. F#302 confirmed at current platform with first numeric magnitude.
- **F-NEW (behavioral non-additivity at Gemma 4 E4B):** Composing 3 distinct-domain adapters yields domain-i PPL strictly worse than adapter-i alone, with relative deviations 0.21 (code) to 2.19 (medical). F#334 confirmed at current platform; prediction "pre-sum is unrouted mixture" is now numerically lower-bounded at this scale.
- **F-NEW (residual systematicity):** R is not noise — per-dim mean RMS / mean entry-abs ratio = 0.45 vs 0.017 iid-noise floor (27× over). The cross-term bias has structure that could be investigated as a routing signal.
- **Antipattern continued:** distinct per-adapter seeds applied throughout. No PRNG-key sharing.

## 7. Limitations / Assumptions Logged

- **L1 (single-layer τ):** measured τ at final hidden state only. P4 (depth-monotonic τ) is not directly verified; layer-by-layer profile is a clean follow-up that does not affect K1926/K1927 verdicts.
- **L2 (single seed per adapter):** variance bound on τ across seed triplets is not measured. Within-batch consistency is implicit (τ is consistent across 45 batches × 3 domains, range 0.47–0.56), but a 3-seed-triplet replication would tighten the claim.
- **L3 (medical Δ amplification):** the 219% medical Δ is amplified by floor effects (adapter_medical drives PPL from 2213.7 → 1.34, a 1655× reduction; any absolute drift in composed dwarfs the tiny adapter_medical PPL). Code (21%) and math (37%) Δ values are not floor-amplified and independently exceed the 10% threshold.
- **L4 (under-training):** 100 iters per adapter is reduced from F#627 canonical 1000. All three adapters meaningfully trained (own-domain lift ≥79% — adequately_trained=True), but a longer-train follow-up could test whether τ grows or shrinks with adapter magnitude.
- **L5 (q_proj only):** F#627-canonical target is q_proj. Whether τ at v_proj+o_proj (also F#627 supported target combo) differs is a follow-up.

## 8. Conclusion

Both pre-registered kill criteria pass with large margins. Sum-of-deltas LoRA composition at the F#627-canonical Gemma 4 E4B 4-bit recipe is **structurally non-additive (τ ≈ 0.48)** and **behaviorally non-additive (composed PPL deviates from single-adapter PPL on its own domain by 0.21–2.19 relative)**. F#302 and F#334 are replicated at the current target with quantitative magnitudes. Pre-sum composition cannot be a drop-in replacement for a routed composer at this scale; the cross-term residual is too large to ignore.

**Verdict: SUPPORTED.**

# PAPER — exp_spark_quant_residual_repair

**Verdict: KILLED** (K2301 fail). The domain LoRA delta does **not** align with the 4-bit
quantization residual of its own q_proj. Adapters are not measurably "dequant repair kits."

## Claim under test

Hypothesis (Gemini "quant repair kit" hunch, invert-assumption perturbation): a domain LoRA's
weight-space delta `ΔW = (A@B)ᵀ` on q_proj partly *repairs the 4-bit quantizer's residual*
`R = W_fp16 − dequant(W_int4)`. If true, the real per-layer mean cosine `cos̄` between `ΔW` and `R`
should exceed `2× p95` of a norm-matched random-rank6 null, with predicted real `cos̄` in the
**0.05–0.30** band (orders of magnitude above the ~1e-3 null floor).

## Method (real, not mock)

- Both models loaded from disk as real quantized snapshots: 4-bit `mlx-community/gemma-4-e4b-it-4bit`
  (the base the adapters were trained on) and 8-bit `mlx-community/gemma-4-e4b-it-8bit` as the fp proxy.
- Residual reference `R_ℓ = dequant(W_int8_ℓ) − dequant(W_int4_ℓ)`. The 8-bit step is 16× finer than
  4-bit, so this captures the 4-bit residual direction to within ~6% — far inside the 2× null margin,
  so it cannot manufacture a pass.
- Per layer ℓ (42 q_proj layers), `cos(flatten(ΔW_ℓ), flatten(R_ℓ))`; pooled norm-matched random-rank6
  null with n=200/layer (8400 pooled), seed=42. Pure weight space — no decode, no training.
- `is_smoke: false`. Run wall clock 1539.7 s. mlx_lm 0.31.2.

## Prediction vs. measurement

| Adapter (q_proj) | Predicted `cos̄` | Measured `cos̄` (signed) | `\|cos̄\|` | 2×p95 threshold | ratio cos̄/p95 | Result |
|---|---|---|---|---|---|---|
| math (primary) | 0.05–0.30 | **−4.66e-5** | 3.45e-4 | 1.628e-3 | −0.057 | fail |
| medical | 0.05–0.30 | −6.14e-5 | 2.98e-4 | 1.628e-3 | −0.075 | fail |
| python | 0.05–0.30 | +2.95e-5 | 3.31e-4 | 1.665e-3 | +0.035 | fail |

The signed real mean is statistically indistinguishable from zero (≈ ±5e-5, i.e. `O(1/√D)` noise),
and even `|cos̄|` (~3e-4) sits **below** the null's own p50 (~2.7e-4) and well below 2×p95 (~1.6e-3).
The predicted alignment is absent by **2–3 orders of magnitude**. All three adapters agree.

## Refutation threshold and how it was crossed

Pre-registered kill K2301: KILL if `cos̄_real < 2 · p95_null` for the math q_proj adapter.
Measured `cos̄_real = −4.66e-5 < 1.628e-3`. The ratio `cos̄/p95 = −0.057` — the real delta is
*less* aligned with the quant residual than a random rank-6 matrix of matched norm. **Threshold
crossed decisively; the hypothesis is killed**, and the corroborating adapters (medical, python)
fail identically.

## Interpretation

The LoRA delta is essentially orthogonal to the quantizer's residual direction. Whatever a domain
adapter encodes, it is **not** a correction of its own base's int4 rounding error on q_proj — there
is no measurable overlap. This refutes the "adapter = dequant repair kit" reframing and does not
rescue the off-domain-interference picture (F#827): off-domain effects cannot be re-explained as
competing quant-error corrections, because there is no quant-error correction component to compete.

## Threats to validity (addressed)

- **8-bit proxy, not true fp16.** The ~6% direction error is far smaller than the 2× margin and the
  observed gap is ~100×; a true-fp16 reference could not lift a −0.057 ratio to ≥ 2.0.
- **Orientation.** `ΔW = (A@B)ᵀ` matches `W` (out=2048, in=2560); cosine and the null are scale-invariant,
  so LORA_SCALE is irrelevant.
- **Smoke.** `is_smoke:false`; both quantized models loaded, all 42 layers dequantized and compared.

**Verdict line: KILLED — math q_proj cos̄ = −4.66e-5 vs 2×p95 = 1.63e-3 (ratio −0.057); predicted
0.05–0.30 band missed by 2–3 orders of magnitude; medical/python corroborate.**

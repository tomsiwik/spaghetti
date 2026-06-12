# MATH — exp_spark_quant_residual_repair

## Hypothesis (the failure mode being probed)

The 862-finding program treats a domain LoRA delta `ΔW = B@A` as *task knowledge* and treats
off-domain composition effects as *interference* (F#827). This experiment probes an alternative:
the frozen base is **4-bit quantized**, and part of what a LoRA learns when fine-tuned on top of a
4-bit base is to *repair the quantizer's own residual* — i.e. the fp16→int4 error on that exact
projection. If true, then `ΔW` should align with the quantization residual
`R = W_fp16 − dequant(W_int4)` of the same q_proj far more than a random rank-6 matrix of matched norm.

## Setup

- Base: `mlx-community/gemma-4-e4b-it-4bit` (frozen, the model the adapters were trained on).
- Adapters: r=6 q_proj LoRA on `/data/adapters/{math,medical,python}` (84 keys each, 42 q_proj layers).
- Per layer ℓ and adapter d, the weight-space delta is `ΔW_ℓ = (A_ℓ @ B_ℓ).ᵀ` with shape (out=2048, in=2560),
  matching `W` orientation. (Forward is `x@A@B`; as a weight delta on `y=Wx` this is `(A@B)ᵀ`.)
  LORA_SCALE is irrelevant: cosine and the norm-matched null are both scale-invariant.

### Quantization residual reference (real, local, no download)

The true fp16 reference `google/gemma-4-e4b-it` is a 16 GB **gated** download not present locally. We use
the **8-bit** local model `mlx-community/gemma-4-e4b-it-8bit` as a high-fidelity fp proxy:

    R_ℓ = dequant(W_int8_ℓ) − dequant(W_int4_ℓ)

Justification (quantitative): affine quant residual scales ≈ `Δ/√12` with step `Δ = range/2^bits`.
The 8-bit step is `2^4 = 16×` finer than 4-bit, so `‖W_fp16 − W_int8‖ ≈ ‖W_fp16 − W_int4‖ / 16`.
Therefore `dequant(W_int8)` recovers the fp16 reference to within ~6% of the 4-bit residual norm, and
`R_8→4 = W_int8 − W_int4` captures the 4-bit quantization residual direction to within that ~6%. This is
a real measured weight-space quantity (both models are quantized snapshots on disk), never a mock. The
≤6% reference error is far below the 2× null-margin we pre-register, so it cannot manufacture a pass.

## Theorem (alignment prediction)

For a rank-6 matrix `ΔW` drawn with **no relation** to `R` (the null), in dimension
`D = 2048·2560 ≈ 5.2M`, the JL / random-projection concentration gives
`E[cos(flatten(ΔW_rand), flatten(R))] = 0` with std `≈ 1/√D ≈ 4.4e-4`. A norm-matched random *rank-6*
matrix has its energy in a 6-dim row/col subspace but flattened cosine against a fixed vector still
concentrates near 0 with std `O(1/√D)`; the 95th percentile of the null is `≈ 1.6e-3` to a few e-3.

**Prediction.** If LoRA partly repairs the 4-bit residual, the *real* per-layer mean
`cos̄ = mean_ℓ cos(flatten(ΔW_ℓ), flatten(R_ℓ))` for q_proj will exceed **2× the 95th-percentile of the
norm-matched random-rank6 null**, i.e. `cos̄ ≥ 2 · p95_null`. Predicted real cos̄ in the **0.05–0.30**
band if the repair effect is real (orders of magnitude above the ~1e-3 null).

## Pre-registered kill criterion (K2301)

> **KILL** if, for q_proj across all 42 layers, the mean cosine of the real LoRA delta against the 4-bit
> quant residual does **NOT** exceed `2 ×` the 95th-percentile of a norm-matched random-rank6 null
> (i.e. `cos̄_real < 2 · p95_null`, equivalently the effect is `< 2×` null or below the 95th pct).

Decision rule, evaluated **per adapter** (math is the primary, registered in the spec; medical/python
are corroborating):
- `supported`  ⇔ `cos̄_real ≥ 2 · p95_null` for the **math** q_proj adapter.
- `killed`     ⇔ `cos̄_real <  2 · p95_null` for the math q_proj adapter.

Null construction: for each layer, draw `n_null = 200` random rank-6 matrices `Ã@B̃` with iid Gaussian
factors, rescale each to `‖·‖_F = ‖ΔW_ℓ‖_F` (norm-matched), flatten, take `|cos|` against `flatten(R_ℓ)`;
pool across layers; `p95_null` is the 95th percentile of the pooled `|cos|` distribution. Seed=42.
We compare the *signed* real mean against `2·p95` and also report `|cos̄_real|` for robustness.

## What would make this provisional, not a verdict

Only smoke / inability to load the real models. `is_smoke:false` requires both 4-bit and 8-bit Gemma-4
loaded and all 42 q_proj layers dequantized and compared. No decode, no training — pure weight space.

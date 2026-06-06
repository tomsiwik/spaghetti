# MATH.md — P11.K0: CLoQ Calibrated LoRA Initialization

## Experiment Type
**Verification** — Proof complete; experiment confirms quantitative predictions.

## Failure Mode
Standard LoRA initialization ignores quantization error. The adapter starts at W_Q (4-bit
reconstructed) and must learn both the quantization compensation AND the task-specific
adaptation, wasting training capacity on a known, computable correction.

## Prior Math
- **Eckart-Young-Mirsky theorem** (Schmidt 1907, Eckart-Young 1936): The best rank-r
  approximation of a matrix E in Frobenius norm is given by its truncated SVD:
  E_r = U_r Σ_r V_r^T, with ||E - E_r||_F = sqrt(σ_{r+1}² + ... + σ_n²).
- **CLoQ** (arXiv:2501.18475): Calibrated LoRA for Quantized LLMs — computes optimal
  LoRA initialization that compensates for quantization error using calibration data.
- **Group-wise quantization error structure**: For per-group quantization, error matrix E
  has structured correlation across output rows (each row uses the same per-group scales),
  giving E low effective rank. Top-r SVD captures dominant error directions.

## Theorem

**Theorem (CLoQ Initialization)**: Let W_4 ∈ R^{m×n} be the 4-bit dequantized weights
and W_8 ∈ R^{m×n} be the 8-bit dequantized weights of a target layer. Define the
quantization error approximation:

    E = W_8 - W_4 ≈ W_float - W_4     [8-bit error is O(1/2^8), 4-bit is O(1/2^4)]

Let E_r = U_r Σ_r V_r^T be the rank-r SVD of E (Eckart-Young optimal).

In the MLX LoRALinear forward pass:

    output = W_4 x + scale · (x @ lora_a) @ lora_b

The weight-space delta is:

    ΔW = scale · lora_b^T @ lora_a^T ∈ R^{m×n}

The CLoQ initialization that achieves ΔW = E_r is:

    lora_a = V_r · diag(√(Σ_r / scale))          ∈ R^{n×r}
    lora_b = diag(√(Σ_r / scale)) · U_r^T        ∈ R^{r×m}

## Proof

    scale · lora_b^T @ lora_a^T
    = scale · [U_r · diag(√(Σ_r/scale))] @ [diag(√(Σ_r/scale)) · V_r^T]
    = scale · U_r · diag(Σ_r/scale) · V_r^T
    = U_r · diag(Σ_r) · V_r^T
    = E_r  ✓

**Initial forward pass error** with CLoQ:

    ||W_4 + ΔW - W_float||_F ≈ ||E - E_r||_F = sqrt(Σ_{i>r} σ_i(E)²)

vs. standard LoRA (lora_b=0 at init):

    ||W_4 - W_float||_F = ||E||_F

**Reduction ratio**: ||E - E_r||_F / ||E||_F = sqrt(1 - Σ_{i≤r} σ_i(E)² / ||E||_F²)

If top-r singular values capture fraction f of total energy:
    reduction = sqrt(1 - f)

**QED**

## Why E Has Low Effective Rank

For per-group 4-bit quantization (group_size=64):
- Each group of 64 input values shares one scale and one bias
- All output rows in the same group use the same quantization parameters
- The quantization error for row i: E[i, :] = W_float[i, :] - round(W_float[i, :] / s) · s
- Rows share scale structure → column-group covariance is structured → E has low effective rank
- Prediction: **top-8 singular values capture ≥ 70% of ||E||_F²** for v_proj/o_proj layers

## Quantitative Predictions

| Quantity | Predicted Value | Kill Criterion |
|----------|----------------|----------------|
| SVD energy capture (top-8) | ≥ 70% of ||E||_F² | Empirical verification |
| CLoQ calibration time | < 10 min | K1536 |
| CLoQ vs standard init, MMLU-Pro Δ | +2pp (≥ 65% → ≥ 67%) | K1535 |
| CLoQ + s1K reasoning SFT | ≥ 66% MMLU-Pro (thinking=True) | K1537 |

## Behavioral Prediction

CLoQ starts the adapter closer to W_float, so gradient updates from s1K data go toward
improving reasoning rather than compensating quantization artifacts. This should produce
responses with cleaner factual reasoning and fewer hallucinations induced by quantization
noise in v_proj (values that determine what gets written to residual stream).

## Approximation Quality: 8-bit vs Float

The 8-bit quantization error is O(max_scale / 2^8) while 4-bit is O(max_scale / 2^4).
The ratio is 1/16, so ||W_8 - W_float||_F ≤ (1/16) ||W_4 - W_float||_F.

Therefore: E_approx = W_8 - W_4 satisfies:
    ||E_approx - (W_float - W_4)||_F ≤ (1/16) ||W_4 - W_float||_F

The 8-bit proxy captures ≥ 15/16 of the true quantization error signal.

## Connection to Architecture (Pierre P1)

v_proj projects keys to value space — quantization noise here directly corrupts what
information gets written into the residual stream. o_proj merges attention heads —
quantization here corrupts the composition of multi-head outputs.

CLoQ init ensures that from step 0, the adapter begins correcting the most damaging
quantization directions in these critical projection matrices.

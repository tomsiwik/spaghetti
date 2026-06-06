# PAPER — exp_rdt_lti_injection_mlx

**Verdict: SUPPORTED** — MLX LTI-injection primitive is numerically faithful to
OpenMythos PyTorch reference and bounded ρ(A_d) across typical training dynamics.

## Goal

Port the RDT LTI-injection primitive (Parcae, Prairie et al., 2026) from PyTorch
(OpenMythos, `open_mythos/main.py:643-686`) to MLX. This primitive guarantees
ρ(A) < 1 by construction for the recurrent update h_{t+1} = A·h_t + B·e + T_out
and is a prerequisite for `exp_rdt_loop_lora_gemma4`.

## Method

Construction (see MATH.md Theorem 1):

    A_d = exp(-exp(clamp(log_dt + log_A, -20, 20)))

Three pre-registered kill criteria:

- **K1737** — 1000 Adam training steps on synthetic MSE; record ρ(A_d) per step.
- **K1738** — MLX↔PyTorch parity on 100 random (h, e, transformer_out) triples, identical init.
- **K1739** — NaN-freedom at 4 corners of (log_A, log_dt) ∈ {-20, +20}².

Platform: MLX 0.31.1 float32 GPU / PyTorch 2.10.0 CPU. seed=42, dim=128,
batch=4, seqlen=16.

## Prediction-vs-Measurement

| KC | Prediction (from MATH.md) | Measured | Pass |
|----|---------------------------|----------|------|
| K1737 | max_rho < 1.0 across 1000 steps, 0 violations | max_rho=0.3656, min_rho=0.0295, 0 violations | ✓ |
| K1738 | min_cos > 0.9999 across 100 samples | min_cos=0.99999988, mean=1.0, max_abs_diff=4.77e-7 | ✓ |
| K1739 | 0 NaN/Inf in forward+backward at 4 corners | 0 NaN/Inf across all 4 corners | ✓ |

Trajectory diagnostics for K1737 (sanity for "log_A free in [-20,20]" assumption):

- max(|log_A|) over trajectory: 0.641 (far from the ±20 clamp)
- max(|log_dt|) over trajectory: 0.731
- Confirms the optimizer stayed in the central regime where exact-arithmetic
  ρ < 1 holds without float32 boundary degradation.

K1739 observation: at (log_A=-20, log_dt=-20) the float32 boundary yields
ρ=1.0 exactly (Lemma 1 confirmed empirically). This is the *extreme-corner*
test and measures NaN-freedom only; no NaN or Inf observed. The Theorem's
exact-arithmetic guarantee degrades to ρ ≤ 1 under float32, which is the
documented failure mode of the construction but does not occur during normal
training (K1737 trajectory max_rho = 0.366).

## Findings

- **F-new (candidate)**: The OpenMythos LTI-injection construction ports to
  MLX with cosine similarity 0.999999 (5 "9s") vs PyTorch reference, no NaN
  at ±20 extremes, and ρ bounded well below 1 during 1000 Adam steps of MSE
  training. The primitive is ready for use in `exp_rdt_loop_lora_gemma4`.
- The float32 boundary (Lemma 1) is documented: at s=-20, A_d rounds to 1
  exactly. Downstream experiments should avoid driving log_A+log_dt to -20
  simultaneously via init or gradient clipping; central-regime training does
  not hit this boundary.

## Assumptions

- Target hardware is Apple Silicon (MLX float32 GPU). A float64 path is
  CPU-only in MLX; not tested since downstream loops run on GPU.
- "log_A free in [-20,20]" interpreted as expected functional range, not
  adversarial corner testing. K1739 explicitly tests the corners separately.
- Adam(lr=1e-3) with Gaussian synthetic data is a proxy for the downstream
  RDT training dynamics. Trajectory stays central (max |log_A| < 1).

## Caveats

- K1738 parity uses CPU PyTorch vs GPU MLX — both float32; any kernel
  differences are absorbed in the 4.77e-7 max absolute diff (still < 1e-6).
- Only element-wise primitive tested. Downstream composition with
  TransformerBlock and LoRA adapters is deferred to `exp_rdt_loop_lora_gemma4`.
- K1737 trajectory ran 1000 steps but optimizer barely moved (log_A rms < 0.7).
  A more aggressive optimizer (higher lr, adversarial targets) could push
  log_A toward clamp boundaries; out of scope for this primitive test.

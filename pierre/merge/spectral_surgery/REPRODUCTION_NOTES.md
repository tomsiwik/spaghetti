# Reproduction Notes

Paper: arxiv 2603.03995 – "Spectral Surgery: Training-Free Refinement of LoRA via Gradient-Guided Singular Value Reweighting" (Tian et al., 2026)

## What is implemented

- **SVD decomposition** of trained LoRA deltas (DeltaW = B @ A -> U Sigma V^T) via `mx.linalg.svd`
- **Per-component sensitivity estimation** via gradient projections: g_k = u_k^T G v_k (Sec 3.3, Eq. 4)
- **Four editing policies** (Sec 3.4):
  - `abs_select`: hard three-level gating (core/noise/middle)
  - `smooth_abs`: continuous sigmoid-gated reweighting with adaptive temperature
  - `grad_direction`: signed multiplicative update with asymmetric step sizes
  - `random_index`: matched-random control baseline
- **Energy preservation**: L1 (nuclear-norm) renormalization, sigma clamping
- **LoRA factor reconstruction** from edited SVD (B', A' = sqrt-split of U diag(sigma') V^T)
- **End-to-end pipeline** with configurable calibration loop

## Differences from paper

1. **Framework**: MLX (Apple Silicon) instead of PyTorch/CUDA. All ops use `mlx.core` and `mlx.nn`.
2. **SVD computation**: `mx.linalg.svd` runs on CPU stream (Apple Accelerate) -- this is correct for MLX as SVD is CPU-bound.
3. **Gradient approximation**: The paper computes dL/d(DeltaW) directly. Since DeltaW = B @ A and we only have dL/dB and dL/dA from autograd, we approximate G = dL/dB @ A. This is exact when A has orthonormal rows (which is approximately true for small-rank LoRA with typical initialization).
4. **No evaluation harness**: The paper uses lm-evaluation-harness for benchmarking. This implementation provides only the surgery procedure, not the evaluation pipeline.
5. **Quantile computation**: Simplified to index-based rather than interpolated quantiles; sufficient for rank-16 vectors.

## Ambiguities in the paper

- **Exact normalization in smooth_abs**: The paper describes "within-module normalization (default: mean-absolute normalization)" but does not specify the exact sigmoid output range. We use a conservative spread of 0.4 centered on mid_factor.
- **Gradient accumulation across calibration samples**: The paper says "aggregate the scalar sensitivity magnitude s_k = |g_k| over calibration examples" but does not specify whether this is per-batch or per-sample. We accumulate per-batch and then take mean_abs.
- **align_mid shifting**: The paper mentions optionally shifting mu so gate at center = gamma_mid. We implement this as a linear rescaling of the sigmoid output.

## Key hyperparameters (from paper)

| Parameter | Default | Source |
|-----------|---------|--------|
| Calibration samples | 128 | Table 4 |
| LoRA rank | 16 | Sec A.1 |
| LoRA alpha | 32 | Sec A.1 |
| Edit modules | o_proj, down_proj | Sec 3.2, Table 5 |
| Energy preservation | L1 | Table 3 |
| smooth_abs center_q | 0.5 (median) | Sec 3.4 |
| smooth_abs temperature | 1.0 | Sec 3.4 |

## Relevance to composable ternary experts project

Spectral Surgery is directly relevant to adapter composition:
- It provides a principled way to refine LoRA adapters post-training
- The subspace-spectrum dichotomy (stable directions, inefficient spectrum) may apply to composed adapter stacks
- Energy-preserving reweighting could help maintain interference bounds when composing multiple adapters
- The sensitivity estimation framework could guide which singular components to preserve during adapter merging

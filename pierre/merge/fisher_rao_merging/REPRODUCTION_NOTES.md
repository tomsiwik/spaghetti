# Reproduction Notes: Fisher-Rao Karcher Mean Merging

Paper: "Functionality-Oriented LLM Merging on the Fisher-Rao Manifold" (arXiv:2603.04972)
Authors: Jiayu Wang, Zuojun Ye, Wenpeng Yin (2025)

## What this implementation covers

- **Section 3.3**: Fixed-point Karcher mean iteration on S^(d-1) via closed-form spherical log/exp maps.
- **Section 3.4**: Spherical proxy -- blockwise normalization, Karcher mean, norm rescaling.
- **Section 3.3 special case**: For N=2, the algorithm reduces to SLERP (verified analytically).
- **Collapse diagnostics** (Section 4.2, Q4): activation variance, effective rank, norm shrinkage ratio.

## What this implementation does NOT cover

- **Evaluation harness**: The paper uses lm-evaluation-harness with HellaSwag, BBH, MMLU-Pro, MuSR, GPQA-Diamond. Those benchmarks are external.
- **Model loading/saving**: This is a pure merging algorithm. Loading MLX model weights is left to the caller (e.g., via `mlx-lm` or `mlx.nn.Module.load_weights`).
- **Optional Fisher preconditioning** (Section 3.4, "Connection to Fisher geometry"): The config flag exists but diagonal Fisher estimation is not implemented. The paper describes this as optional and complementary.
- **MergeKit integration**: The paper's official code is a PR to mergekit (PyTorch). This is an independent MLX reimplementation.

## Unspecified choices made in this implementation

| Choice | Value | Rationale |
|--------|-------|-----------|
| Convergence tolerance | 1e-7 radians | Not stated in paper; standard for spherical means |
| Max iterations | 100 | Paper does not specify; convergence is typically <20 on S^(d-1) |
| Karcher init | Weighted Euclidean mean projected to sphere | Not stated; faster convergence than arbitrary init |
| Block granularity | Per-tensor (each named parameter) | Paper says "e.g., layer or tensor group"; per-tensor is finer-grained |
| Near-zero norm guard | 1e-8 | Numerical safety for skip-merging degenerate blocks |
| SLERP fallback for near-parallel vectors | Linear interpolation when sin(omega) < 1e-8 | Standard numerical practice |

## Official code reference

The authors' implementation is a MergeKit PR (PyTorch):
https://github.com/arcee-ai/mergekit/commit/09bbb0ae282c6356567f05fe15a28055b9dc9390

This MLX implementation follows the same mathematical algorithm but uses `mlx.core` throughout.

## Key equations

1. **Karcher objective** (Eq. 3): `theta* = argmin sum_i alpha^(i) * d_FR(theta, theta^(i))^2`
2. **Fixed-point update** (Eq. 5): `theta_{t+1} = Exp_{theta_t}(eta * sum_i alpha^(i) * Log_{theta_t}(theta^(i)))`
3. **Spherical log map**: `Log_p(q) = (theta/sin(theta)) * (q - cos(theta) * p)` where `theta = arccos(<p,q>)`
4. **Spherical exp map**: `Exp_p(v) = cos(||v||) * p + sin(||v||) * v/||v||`
5. **Norm rescaling**: `merged_block = mean_norm * karcher_direction`

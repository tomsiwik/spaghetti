# OrthoMerge: Karcher mean on Stiefel for Pierre's shared-A — KILLED

## Result

**KILLED** after 3 fix attempts. OrthoMerge (arxiv 2602.05943) cannot run on Pierre's shared-A architecture in the no-base path (B).

## Measurements

| Method | gsm8k | humaneval | medqa | avg |
|--------|-------|-----------|-------|-----|
| Single-best | 66.0% | 78.0% | 42.0% | 62.0% |
| Fisher-Rao | 68.0% | 68.0% | 58.0% | 64.7% |
| Full-delta DARE | — | — | — | 71.3% (prior) |
| OrthoMerge | CRASH | CRASH | CRASH | — |

## Kill Criteria

| KC | Verdict | Evidence |
|----|---------|----------|
| K1: OrthoMerge avg ≥ Fisher-Rao + 3pp | FAIL | Never produced results |
| K2: DARE avg − OrthoMerge avg ≤ 4pp | FAIL | Never produced results |
| K3: Pipeline ≤ 60s | FAIL | Crashed during preprocessing |
| K4: K=1 sanity | FAIL | Never reached |

## Root Cause

Three distinct failures across 3 fix attempts:

1. **Shape mismatch (d_in > d_out)**: ΔW is (2560, 2048) but rotation matrices are (2048, 2048). The original padding logic assumed d_in ≤ d_out.

2. **Shape mismatch (d_in < d_out)**: On up-projection layers, ΔW is (2560, 4096). The Procrustes target construction broadcast failed.

3. **SVD numerical instability**: After fixing both shape bugs, LAPACK `sgesvdx_` failed with code 1 (QR non-convergence). The no-base path computes `target = I + scale·A·B[:d_sq, :d_sq]` where scale=6.0, producing ill-conditioned matrices that crash SVD.

## Conclusion

OrthoMerge's no-base fallback (path B, W_0=I) is both **mathematically degenerate** (acknowledged in MATH.md) and **numerically unstable** for Pierre's adapter magnitudes. Path A (with-base) would require reading quantized base weights per layer — non-trivial in MLX and outside the scope of a single experiment.

Finding: Riemannian averaging methods that need full W_0 are incompatible with Pierre's adapter-only composition design. Simple Euclidean methods (Fisher-Rao, DARE) remain the correct approach.

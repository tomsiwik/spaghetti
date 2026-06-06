# DC-Merge Reproduction Notes

Paper: "DC-Merge: Improving Model Merging with Directional Consistency"
arXiv: 2603.06242
Official code: https://github.com/Tobeginwith/DC-Merge

## What This Implementation Covers

This is an MLX-native implementation of the DC-Merge algorithm (Algorithm 1) for
Apple Silicon. It implements the full pipeline:

1. **Task vector extraction**: tau_i = W_finetuned_i - W_base
2. **Truncated SVD** of each task vector
3. **Energy smoothing** of singular values (average and linear strategies)
4. **Cover space construction** via whitening of concatenated per-task bases
5. **Projection** of smoothed task vectors onto shared orthonormal cover space
6. **Element-wise merging** in cover space (Task Arithmetic or TIES-Merging)
7. **Back-projection** with structural block-diagonal mask
8. **Rescaled merge**: W_merged = W_base + alpha * Delta_merged

## Key Equations Mapped to Code

| Paper Reference | Code Location |
|---|---|
| Task vector (Sec 2.1) | `merge.py: dc_merge()` — computed as `ts[key] - base_w` |
| Truncated SVD | `model.py: truncated_svd()` |
| Energy smoothing (Alg 1, line 6; Appendix E.4) | `model.py: energy_smoothing()` |
| Average smoothing (Eq 12) | `energy_smoothing(strategy="average")` |
| Linear smoothing (Eq 13-14) | `energy_smoothing(strategy="linear")` |
| Cover basis construction (Eq 9) | `model.py: construct_cover_basis()` |
| Whitening (Alg 1, line 10) | `model.py: _whiten()` |
| Projection to cover space (Eq 10) | `model.py: project_to_cover_space()` |
| Structural mask (Alg 1, line 15) | `model.py: _build_block_diag_mask()` |
| Back-projection (Eq 11) | `model.py: project_to_param_space()` |
| TIES-Merging in cover space | `merge.py: _merge_ties()` |
| DirSim metric (Eq 3) | `model.py: dir_sim()` |

## Ambiguities and Decisions

1. **SVD stream**: We use `stream=mx.cpu` for `mx.linalg.svd` because MLX SVD
   is CPU-only as of v0.22. This is a known MLX limitation.

2. **Whitening implementation**: The paper states whitening is used as a
   computationally efficient near-optimal solution to Eq. (8). We implement it
   via SVD: X_white = U @ V^T, which yields orthonormal columns spanning the
   same column space as X.

3. **TIES top-k threshold**: We use a sort-based threshold rather than
   `mx.topk` (which MLX does not provide). This is functionally equivalent.

4. **Structural mask**: Built as explicit block-diagonal matrix multiplication
   rather than in-place indexing, for MLX compatibility.

5. **1D parameters**: Biases and layer norm params are merged via simple
   averaging, following the paper's Section E.2.

## Hyperparameter Defaults (from paper)

- `rank`: 16 (= LoRA rank for LoRA settings)
- `smoothing`: "average" for LoRA, "none" (truncation only) for FFT
- `rho`: 5.0 for linear smoothing
- `merge_method`: "ties" with `top_k=0.1`
- `alpha`: tuned on validation set; paper uses 2.0 for MM-MergeBench
- `use_mask`: True (especially important for FFT with many tasks)

## Differences from Official PyTorch Implementation

- All ops use `mlx.core` instead of `torch`
- `mx.linalg.svd` runs on CPU (MLX limitation)
- No `torch.topk` — we sort instead
- Memory managed via explicit `mx.eval()` at loop boundaries
- No CUDA — runs natively on Apple Silicon unified memory

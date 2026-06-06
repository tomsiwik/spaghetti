"""TIES-Merging adapted to Pierre's shared-A architecture.

Per arxiv 2306.01708 (Yadav et al., 2023). Three-step algorithm operating
on stacked task-vector deltas:

  1. Trim:        keep top-K% by magnitude per adapter (paper: K%=20-30)
  2. Elect Sign:  γ_m = sign(Σ_t T_t)  (majority-vote sign per cell)
  3. Disjoint:    average only the t's that agree with γ_m

For Pierre's shared-A: materialize ΔW_t = scale · A · B_t at compose time,
run TIES on the materialized deltas, install via _FusedDeltaLinear.

Reference: prateeky2806/ties-merging (`src/ties_minimal.ipynb`).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def _topk_mask(M: mx.array, keep_frac: float) -> mx.array:
    """Per-row TopK-by-magnitude mask. M shape (T, D), returns mask (T, D)."""
    abs_M = mx.abs(M)
    T, D = M.shape
    k = max(1, int(D * keep_frac))
    # Find threshold per row: kth largest magnitude
    sorted_abs = mx.sort(abs_M, axis=1)  # ascending
    threshold = sorted_abs[:, -k:-k+1] if k < D else sorted_abs[:, :1]
    # threshold has shape (T, 1)
    return abs_M >= threshold


def compose_ties_full_delta(
    adapter_states, A_dict, scale,
    *, keep_frac: float = 0.3,
):
    """TIES-Merging on shared-A materialized deltas.

    Args:
        adapter_states: list of K state dicts.
        A_dict: shared-A dict (used to materialize ΔW).
        scale: PoLAR scale factor.
        keep_frac: fraction of largest-magnitude entries to keep per adapter
            (paper recommends K%=20-30%, here defaulting to 30%).
    """
    K = len(adapter_states)
    if K == 0:
        return {}

    all_keys: set[str] = set()
    for st in adapter_states:
        all_keys.update(st.keys())

    fused: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        per_adapter = [st[key] for st in adapter_states if key in st]
        T = len(per_adapter)
        if T == 0:
            continue
        if T == 1:
            entry = per_adapter[0]
            A = entry["a"].astype(mx.float32)
            B = entry["b"].astype(mx.float32)
            fused[key] = (scale * (A @ B)).astype(mx.bfloat16)
            continue

        # Materialize per-adapter ΔW_t
        deltas = []
        for entry in per_adapter:
            A_t = entry["a"].astype(mx.float32)
            B_t = entry["b"].astype(mx.float32)
            s_t = float(entry.get("scale", scale))
            dW = s_t * (A_t @ B_t)
            mx.eval(dW)
            deltas.append(dW)

        # Stack flattened for trim/sign/merge
        d_in, d_out = deltas[0].shape
        D = d_in * d_out
        T_flat = mx.stack([dW.reshape(-1) for dW in deltas])  # (T, D)

        # 1. Trim — TopK-by-magnitude per row
        keep = _topk_mask(T_flat, keep_frac)
        T_trim = T_flat * keep.astype(T_flat.dtype)

        # 2. Elect sign: γ = sign(Σ_t T_trim_t)
        sign_sum = mx.sum(T_trim, axis=0)
        gamma = mx.sign(sign_sum)  # (D,)

        # 3. Disjoint merge: keep only entries whose sign matches γ
        # For each cell, mean over t such that sign(T_trim_t) == γ_cell.
        agree = (mx.sign(T_trim) == gamma[None, :]) & (T_trim != 0)
        agree_f = agree.astype(T_flat.dtype)
        masked_vals = T_trim * agree_f
        n_agree = mx.sum(agree_f, axis=0)  # (D,)
        merged_flat = mx.sum(masked_vals, axis=0) / mx.maximum(n_agree, 1.0)

        merged = merged_flat.reshape(d_in, d_out)
        mx.eval(merged)
        fused[key] = merged.astype(mx.bfloat16)
    return fused

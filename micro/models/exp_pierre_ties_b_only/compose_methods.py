"""TIES-Merging applied to B-matrices only (no full-delta materialization).

Per the research agent's note: "TopK-by-magnitude on B alone wouldn't carry
semantic meaning the way it does on ΔW entries — the magnitudes of B's
entries don't directly correspond to output influence (A scales them)."

This experiment tests that claim empirically. If TIES-on-B works anyway,
the claim is false and we have a Pierre-architecture-honest TIES variant.
If it fails (likely), it's confirmation that full-delta is necessary for
TIES's trim+sign-elect to be meaningful.

Algorithm: same TIES three-step but operating on B-matrices directly.
Returns a B-dict (B-only architecture preserved).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def _topk_mask(M: mx.array, keep_frac: float) -> mx.array:
    abs_M = mx.abs(M)
    T, D = M.shape
    k = max(1, int(D * keep_frac))
    sorted_abs = mx.sort(abs_M, axis=1)
    threshold = sorted_abs[:, -k:-k+1] if k < D else sorted_abs[:, :1]
    return abs_M >= threshold


def compose_ties_b_only(B_lists, A_dict, *, keep_frac: float = 0.3,
                        rescale_to_mean_norm: bool = True):
    """TIES three-step applied directly to B-matrices.

    Returns a B-dict (Pierre's compose API preserved). Optionally rescales
    output to mean source Frobenius norm (matching Fisher-Rao convention).
    """
    if len(B_lists) == 1:
        return B_lists[0]

    all_keys: set[str] = set()
    for ab in B_lists:
        all_keys.update(ab.keys())

    composed: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        Bs = [ab[key].astype(mx.float32) for ab in B_lists if key in ab]
        T = len(Bs)
        if T == 0:
            continue
        if T == 1:
            composed[key] = Bs[0].astype(mx.bfloat16)
            continue

        rank, d_out = Bs[0].shape
        D = rank * d_out
        T_flat = mx.stack([b.reshape(-1) for b in Bs])  # (T, D)

        # 1. Trim — TopK by magnitude per row
        keep = _topk_mask(T_flat, keep_frac)
        T_trim = T_flat * keep.astype(T_flat.dtype)

        # 2. Elect sign
        gamma = mx.sign(mx.sum(T_trim, axis=0))

        # 3. Disjoint merge
        agree = (mx.sign(T_trim) == gamma[None, :]) & (T_trim != 0)
        agree_f = agree.astype(T_flat.dtype)
        masked = T_trim * agree_f
        n_agree = mx.sum(agree_f, axis=0)
        merged_flat = mx.sum(masked, axis=0) / mx.maximum(n_agree, 1.0)

        merged = merged_flat.reshape(rank, d_out)

        # Optional norm rescaling (consistent with Fisher-Rao)
        if rescale_to_mean_norm:
            orig_norms = mx.stack([mx.linalg.norm(b.reshape(-1)) for b in Bs])
            mean_source_norm = mx.mean(orig_norms)
            mean_norm = mx.linalg.norm(merged.reshape(-1))
            mx.eval(mean_source_norm, mean_norm)
            if mean_norm.item() > 1e-8:
                merged = merged * (mean_source_norm / mean_norm)

        composed[key] = merged.astype(mx.bfloat16)
    return composed

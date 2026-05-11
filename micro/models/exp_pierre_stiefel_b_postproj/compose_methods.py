"""Stiefel post-hoc projection on existing PoLAR B-matrices.

Joint-row-orthonormalization across K adapters per layer:
  B_all = [B_1; ...; B_K] ∈ ℝ^(K·r × d_out)
  Project so that B_all B_all^T = I (rows orthonormal)
  Slice back into per-adapter B-dicts

Two variants:
  - strict: pure QR projection, scaling info from R discarded
  - rescaled: project + rescale each adapter's B to its original Frobenius norm
"""
from __future__ import annotations
from typing import Literal

import mlx.core as mx


StiefelVariant = Literal["strict", "rescaled"]


def stiefel_project_b_dicts(
    B_dicts: list[dict[str, mx.array]],
    *,
    variant: StiefelVariant = "rescaled",
) -> list[dict[str, mx.array]]:
    """Joint-Stiefel projection on K B-matrices, per layer.

    Args:
        B_dicts: list of K dicts, each {layer_key: B[r, d_out]}.
        variant: "strict" → exact Stiefel (rows orthonormal, magnitude=1);
                 "rescaled" → Stiefel-then-rescale to original Frobenius norm.

    Returns:
        List of K B-dicts with projected B's. Same shape/keys as input.
    """
    K = len(B_dicts)
    if K == 0:
        return []

    # Collect all keys present in any adapter
    all_keys: set[str] = set()
    for d in B_dicts:
        all_keys.update(d.keys())

    out: list[dict[str, mx.array]] = [{} for _ in range(K)]

    for key in sorted(all_keys):
        # Only adapters that have this key contribute
        present = [(k, B_dicts[k][key]) for k in range(K) if key in B_dicts[k]]
        if not present:
            continue
        if len(present) == 1:
            k, B = present[0]
            out[k][key] = B
            continue

        # Stack rows
        Bs_f32 = [B.astype(mx.float32) for _, B in present]
        rs = [B.shape[0] for B in Bs_f32]
        d_out = Bs_f32[0].shape[1]
        B_all = mx.concatenate(Bs_f32, axis=0)  # (sum_r, d_out)

        # Joint Stiefel: rows orthonormal. QR on transpose.
        # B_all^T = Q · R, Q ∈ ℝ^(d_out × sum_r) orthonormal columns
        Q, R = mx.linalg.qr(B_all.T, stream=mx.cpu)
        B_all_stiefel = Q.T  # (sum_r, d_out), rows orthonormal

        # Slice back per adapter
        offset = 0
        for idx, (k, B_orig) in enumerate(present):
            r = rs[idx]
            B_proj = B_all_stiefel[offset:offset + r, :]
            offset += r

            if variant == "rescaled":
                orig_norm = mx.linalg.norm(B_orig.astype(mx.float32).reshape(-1))
                proj_norm = mx.linalg.norm(B_proj.reshape(-1))
                mx.eval(orig_norm, proj_norm)
                if proj_norm.item() > 1e-8:
                    B_proj = B_proj * (orig_norm / proj_norm)

            out[k][key] = B_proj.astype(mx.bfloat16)

    return out


def compose_simple_mean(
    B_lists: list[dict[str, mx.array]],
    A_dict=None,
    weights=None,
):
    """Uniform 1/K mean of B-matrices, no rescaling.

    This is the composition method that *should* work if adapters are
    already on Stiefel — no Fisher-Rao norm correction needed, because
    orthogonal rows preserve mean magnitudes by Pythagoras.
    """
    if len(B_lists) == 1:
        return B_lists[0]
    if weights is None:
        K = len(B_lists)
        weights = [1.0 / K] * K
    w_sum = sum(weights)

    all_keys: set[str] = set()
    for ab in B_lists:
        all_keys.update(ab.keys())

    composed: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        tensors = [ab[key].astype(mx.float32) for ab in B_lists if key in ab]
        ws = weights[: len(tensors)]
        merged = mx.zeros_like(tensors[0])
        for t, w in zip(tensors, ws):
            merged = merged + (t * (w / w_sum))
        composed[key] = merged.astype(mx.bfloat16)
    return composed

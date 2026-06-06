"""Pico calibration on stacked B-matrices, applied as a Fisher-Rao pre-stage.

Per arxiv 2604.16826 ("Crowded in B-Space"). Pico diagnoses that the merged
adapter's interference comes from over-shared output-side directions in B,
and calibrates by SVD'ing stacked B's, computing per-direction sharing
scores, and dampening over-shared directions before merging.

Implementation notes from paper (algorithm spec from research agent
acbe00274a1a6eb9c):

  1. Column-stack:  B_all = [B_1 | B_2 | ... | B_T]  shape (d_out, T·r)
  2. SVD:           B_all = U Σ V^T  with σ_1, ..., σ_m, m = min(d_out, T·r)
  3. Sharing:       s_j = σ_j² / Σ_k σ_k²
  4. Damping:       α_j = 1 / (1 + (T-1) s_j)
  5. Calibration:   S = I + U · diag(α - 1) · U^T
  6. Apply:         B_t_calib = S · B_t  for each t
  7. Merge:         B_merged = Fisher-Rao(B_t_calib for t in 1..T)
  8. Rescale:       γ = (1/||B_merged||_F) · (1/T) Σ_t ||B_t||_F  (mean source norm)
                    B_out = γ · B_merged

K=1 → α=1 → S=I (no-op), reproduces single-B identity.
α-axis (sanity setting α=1 manually) → S=I → reduces to plain Fisher-Rao.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def compose_pico_then_fisher_rao(
    B_lists, A_dict,
    *, alpha_override: float | None = None,
    rescale_to_mean_norm: bool = True,
):
    """Pico calibration on B-stack, then Fisher-Rao.

    Args:
        B_lists: list of K B-dicts (each: layer_key → B[r, d_out]).
        A_dict:  shared-A dict (unused in B-only Pico; kept for signature uniformity).
        alpha_override: if not None, sets all α_j to this value. Used for K4
            sanity (α=1 → S=I → must reproduce Fisher-Rao within 1pp).
        rescale_to_mean_norm: if True, apply Pico's γ rescaling (`γ · merged`).
            If False, return un-rescaled merged B (ablation: does γ matter?).
    """
    if len(B_lists) == 1:
        return B_lists[0]
    K = len(B_lists)
    all_keys: set[str] = set()
    for ab in B_lists:
        all_keys.update(ab.keys())

    composed: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        B_per = [ab[key].astype(mx.float32) for ab in B_lists if key in ab]
        T = len(B_per)
        if T == 0:
            continue
        if T == 1:
            composed[key] = B_per[0].astype(mx.bfloat16)
            continue

        # B shapes: (r, d_out). Pico stacks in (d_out, T·r) form, so we transpose.
        B_T_per = [b.T for b in B_per]                       # (d_out, r) each
        B_all = mx.concatenate(B_T_per, axis=1)              # (d_out, T·r)

        # SVD on B_all → U: (d_out, m), Σ: (m,), V^T: (m, T·r)
        U, sigma, _ = mx.linalg.svd(B_all, stream=mx.cpu)
        sigma2 = sigma ** 2
        total = mx.sum(sigma2) + 1e-12
        s = sigma2 / total
        alpha = 1.0 / (1.0 + (T - 1) * s)

        if alpha_override is not None:
            alpha = mx.full(alpha.shape, float(alpha_override), dtype=alpha.dtype)

        # Calibration: S = I + U · diag(α - 1) · U^T
        # Compute U_diag = U * (α - 1)  (broadcast across rank dim)
        m = sigma.shape[0]
        U_m = U[:, :m]  # (d_out, m) — top-m singular directions
        alpha_m1 = (alpha - 1.0)
        U_scaled = U_m * alpha_m1[None, :]  # (d_out, m)
        # S = I + U_scaled @ U_m^T  — (d_out, d_out)
        d_out = B_all.shape[0]
        S_calib = mx.eye(d_out, dtype=mx.float32) + (U_scaled @ U_m.T)
        mx.eval(S_calib)

        # Apply: B_t_calib = S · B_t  (operating on (r, d_out): B_t @ S^T)
        B_calib = [b @ S_calib.T for b in B_per]

        # Fisher-Rao mean (norm-rescaled to mean source norm of CALIBRATED B's)
        mean = mx.zeros_like(B_calib[0])
        for b in B_calib:
            mean = mean + b
        mean = mean / T

        # Rescale per Pico's γ = (1/||mean||_F) · mean source norm of ORIGINAL B's
        orig_norms = mx.stack([mx.linalg.norm(b.reshape(-1)) for b in B_per])
        mean_source_norm = mx.mean(orig_norms)
        mean_norm = mx.linalg.norm(mean.reshape(-1))
        mx.eval(mean_source_norm, mean_norm)
        if rescale_to_mean_norm and mean_norm.item() > 1e-8:
            B_out = mean * (mean_source_norm / mean_norm)
        else:
            B_out = mean
        composed[key] = B_out.astype(mx.bfloat16)
    return composed

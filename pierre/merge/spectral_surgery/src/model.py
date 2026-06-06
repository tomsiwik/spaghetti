"""
Spectral Surgery – core spectral primitives (MLX).

Paper: arxiv 2603.03995 – "Spectral Surgery: Training-Free Refinement of
LoRA via Gradient-Guided Singular Value Reweighting" (Tian et al., 2026)

Implements:
  - SVD decomposition of a trained LoRA delta  DeltaW = B @ A  ->  U Sigma V^T
  - Per-component sensitivity estimation:  g_k = u_k^T G v_k
  - Reconstruction of edited delta from (U, sigma', V)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Data container for a decomposed LoRA update
# ---------------------------------------------------------------------------

@dataclass
class SpectralDecomposition:
    """Thin SVD of a LoRA delta:  DeltaW = U @ diag(sigma) @ V^T

    Shapes (rank r, d_out, d_in):
        U:     (d_out, r)   – left singular vectors
        sigma: (r,)         – singular values  (>= 0)
        Vt:    (r, d_in)    – right singular vectors transposed
    """
    U: mx.array
    sigma: mx.array
    Vt: mx.array

    @property
    def rank(self) -> int:
        return self.sigma.shape[0]

    def reconstruct(self, sigma_override: Optional[mx.array] = None) -> mx.array:
        """Reconstruct  DeltaW = U @ diag(sigma) @ V^T."""
        s = sigma_override if sigma_override is not None else self.sigma
        # (d_out, r) * (r,) -> broadcast then @ (r, d_in)
        return (self.U * s[None, :]) @ self.Vt


# ---------------------------------------------------------------------------
# SVD of a LoRA delta
# ---------------------------------------------------------------------------

def decompose_lora_delta(
    B: mx.array,
    A: mx.array,
) -> SpectralDecomposition:
    """Compute thin SVD of the LoRA product  DeltaW = B @ A.

    Args:
        B: (d_out, r)  – LoRA "up" factor
        A: (r, d_in)   – LoRA "down" factor

    Returns:
        SpectralDecomposition with U (d_out, r), sigma (r,), Vt (r, d_in).
    """
    delta_w = B @ A  # (d_out, d_in)
    # mx.linalg.svd returns full matrices; we slice to rank-r thin SVD.
    U_full, S_full, Vt_full = mx.linalg.svd(delta_w, stream=mx.cpu)
    r = min(B.shape[1], A.shape[0])
    mx.eval(U_full, S_full, Vt_full)
    return SpectralDecomposition(
        U=U_full[:, :r],
        sigma=S_full[:r],
        Vt=Vt_full[:r, :],
    )


# ---------------------------------------------------------------------------
# Per-component sensitivity via gradient projections  (Sec 3.3)
# ---------------------------------------------------------------------------

def compute_component_sensitivity(
    decomp: SpectralDecomposition,
    grad_delta_w: mx.array,
) -> mx.array:
    """Sensitivity of each singular component to the loss.

    g_k = u_k^T  G  v_k     (Eq. 4 in paper)

    where G = dL/d(DeltaW) accumulated over calibration examples.

    Args:
        decomp:       SpectralDecomposition of the LoRA delta.
        grad_delta_w: (d_out, d_in) – gradient of the loss w.r.t. DeltaW.

    Returns:
        g: (r,) – signed sensitivity per component.
    """
    # u_k^T G v_k  =  diag(U^T @ G @ V)
    # U^T: (r, d_out),  G: (d_out, d_in),  V: (d_in, r)
    V = decomp.Vt.T  # (d_in, r)
    GV = grad_delta_w @ V             # (d_out, r)
    UtGV = decomp.U.T @ GV           # (r, r)
    g = mx.diag(UtGV)                # (r,)
    return g


def aggregate_sensitivities(
    g_list: list[mx.array],
    method: str = "mean_abs",
) -> mx.array:
    """Aggregate per-sample sensitivities into a single score vector.

    Args:
        g_list: list of (r,) sensitivity vectors, one per calibration batch.
        method: "mean_abs" (default) – mean of |g_k| across samples.

    Returns:
        s: (r,) – aggregated sensitivity magnitude.
    """
    stacked = mx.stack(g_list, axis=0)  # (N, r)
    if method == "mean_abs":
        return mx.mean(mx.abs(stacked), axis=0)
    elif method == "mean_signed":
        return mx.mean(stacked, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Reconstruct LoRA-compatible factors from edited SVD  (Sec 3.4 end)
# ---------------------------------------------------------------------------

def svd_to_lora_factors(
    decomp: SpectralDecomposition,
    sigma_new: mx.array,
) -> tuple[mx.array, mx.array]:
    """Convert edited SVD back to LoRA-compatible (B', A') factors.

    We set:
        B' = U @ diag(sqrt(sigma'))          (d_out, r)
        A' = diag(sqrt(sigma')) @ V^T         (r, d_in)

    so that  B' @ A' = U @ diag(sigma') @ V^T = DeltaW'.

    Args:
        decomp:    original decomposition (for U, V^T).
        sigma_new: (r,) edited singular values.

    Returns:
        (B_new, A_new) tuple.
    """
    sqrt_s = mx.sqrt(mx.maximum(sigma_new, 0.0))
    B_new = decomp.U * sqrt_s[None, :]       # (d_out, r)
    A_new = sqrt_s[:, None] * decomp.Vt       # (r, d_in)
    return B_new, A_new

"""OrthoMerge ported to Pierre's shared-A architecture.

Per arxiv 2602.05943 ("Orthogonal Model Merging"). The 5-step algorithm
operates on full ΔW for non-OFT models like LoRA. We adapt it to Pierre's
shared-A by materializing per-adapter ΔW_t = scale · A · B_t, running
OrthoMerge, and installing via _FusedDeltaLinear.

Algorithm:
  1. Procrustes SVD: solve min_R ‖W_target_t − R · W_0‖_F s.t. R orthogonal.
       M_t = (W_0 + ΔW_t) · W_0^T
       U, _, V^T = SVD(M_t)
       R_t = U @ V^T
  2. Lie-algebra (skew-symmetric) via INVERSE Cayley (paper's choice — not log):
       Q_t = (R_t − I) · (R_t + I)^{-1}
  3. Magnitude-corrected tangent-mean:
       Q_mean = (1/T) · Σ_t Q_t
       c      = Σ_t ‖Q_t‖_F / ‖Σ_t Q_t‖_F
       Q_merged = c · Q_mean
  4. Cayley back to O(n):
       R_merged = (I + Q_merged) · (I − Q_merged)^{-1}
  5. Linear residual merge in Euclidean space:
       ρ_t        = (W_0 + ΔW_t) − R_t · W_0
       ρ_merged   = mean_t(ρ_t)        (or any merger)
       W_final    = R_merged · W_0 + ρ_merged
       ΔW_final   = W_final − W_0      (return as fused delta)

Caveat: Cayley fails if (R+I) is singular (R has eigenvalue −1, 180° rotation).
Paper does not specify a guard — we add `eps · I` regularization in matrix
inversions. Flagged as a porting risk.

For PoLAR's d=2560 (q_proj input dim) × d=2048 (output dim), W_0 is
(2048, 2560). Procrustes target is square (d_out, d_out) = (2048, 2048),
so all the orthogonal-group operations live in d_out × d_out.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def _cayley_inverse(R: mx.array, eps: float = 1e-6) -> mx.array:
    """Inverse Cayley: Q = (R - I) · (R + I)^{-1}."""
    n = R.shape[0]
    I_n = mx.eye(n, dtype=mx.float32)
    R_plus = R + I_n + eps * I_n
    R_minus = R - I_n
    R_plus_inv = mx.linalg.inv(R_plus, stream=mx.cpu)
    return R_minus @ R_plus_inv


def _cayley_forward(Q: mx.array, eps: float = 1e-6) -> mx.array:
    """Cayley: R = (I + Q) · (I - Q)^{-1}."""
    n = Q.shape[0]
    I_n = mx.eye(n, dtype=mx.float32)
    I_minus_Q = I_n - Q + eps * I_n
    I_plus_Q = I_n + Q
    inv_term = mx.linalg.inv(I_minus_Q, stream=mx.cpu)
    return I_plus_Q @ inv_term


def _procrustes_R(W_target: mx.array, W_0: mx.array) -> mx.array:
    """Find orthogonal R minimizing ‖W_target − R · W_0‖_F.

    Both have shape (d_out, d_in). M = W_target · W_0^T is (d_out, d_out).
    SVD: M = U Σ V^T, R = U V^T.
    """
    M = W_target @ W_0.T
    U, _, Vt = mx.linalg.svd(M, stream=mx.cpu)
    return U @ Vt


def compose_orthomerge_karcher(
    adapter_states, A_dict, scale,
    *, base_W0_per_layer: dict | None = None,
    eps_cayley: float = 1e-6,
):
    """OrthoMerge for Pierre's shared-A — returns fused deltas.

    Args:
        adapter_states: list of K state dicts {layer_key: {a, b, scale}}.
        A_dict: shared-A dict (used to materialize ΔW_t).
        scale: PoLAR scale factor.
        base_W0_per_layer: optional dict of layer_key → W_0 base linear weight.
            If None, OrthoMerge falls back to a DEGENERATE form using
            ΔW directly (Procrustes target = ΔW, with W_0 = I_{d_out}). This
            is mathematically a different operation; documented as the
            no-base path.
        eps_cayley: regularization on Cayley inversions to avoid singularity.
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

        # Materialize per-adapter deltas
        deltas = []
        for entry in per_adapter:
            A_t = entry["a"].astype(mx.float32)
            B_t = entry["b"].astype(mx.float32)
            s_t = float(entry.get("scale", scale))
            dW = s_t * (A_t @ B_t)  # shape (d_in, d_out)
            mx.eval(dW)
            deltas.append(dW)

        # We need W_0 to extract rotations. Without it, fall back to using
        # I as the base (degenerate path) — Procrustes finds R such that
        # R approximates ΔW_t as a rotation. This is the documented
        # no-base-W0 fallback.
        if base_W0_per_layer is None or key not in base_W0_per_layer:
            d_in = deltas[0].shape[0]
            d_out = deltas[0].shape[1]
            d_sq = min(d_in, d_out)
            W_0_sq = mx.eye(d_sq, dtype=mx.float32)
            R_per = []
            for dW in deltas:
                target = mx.eye(d_sq, dtype=mx.float32) + dW[:d_sq, :d_sq]
                R_t = _procrustes_R(target, W_0_sq)
                R_per.append(R_t)
        else:
            W_0 = base_W0_per_layer[key].astype(mx.float32)  # (d_out, d_in)
            R_per = []
            for dW in deltas:
                W_target = W_0 + dW.T  # (d_out, d_in)
                R_t = _procrustes_R(W_target, W_0)
                R_per.append(R_t)

        # Lie-algebra mapping via inverse Cayley
        Q_per = [_cayley_inverse(R, eps=eps_cayley) for R in R_per]

        # Magnitude-corrected tangent-mean
        Q_sum = mx.zeros_like(Q_per[0])
        for Q in Q_per:
            Q_sum = Q_sum + Q
        Q_mean = Q_sum / T
        norms_per = mx.stack([mx.linalg.norm(Q.reshape(-1)) for Q in Q_per])
        sum_norms = mx.sum(norms_per)
        norm_of_sum = mx.linalg.norm(Q_sum.reshape(-1))
        mx.eval(sum_norms, norm_of_sum)
        denom = float(norm_of_sum.item()) + 1e-12
        c = float(sum_norms.item()) / denom
        Q_merged = c * Q_mean

        # Cayley back to O(n)
        R_merged = _cayley_forward(Q_merged, eps=eps_cayley)

        # Linear residual: ρ_t = (W_0 + ΔW_t) − R_t · W_0
        # In the no-W0 fallback, W_0 = I and ΔW lives outside the rotation,
        # so residual = ΔW_t − (R_t − I)
        if base_W0_per_layer is None or key not in base_W0_per_layer:
            d_in = deltas[0].shape[0]
            d_out = deltas[0].shape[1]
            d_sq = R_merged.shape[0]  # min(d_in, d_out)
            I_sq = mx.eye(d_sq, dtype=mx.float32)
            residuals = []
            for dW, R_t in zip(deltas, R_per):
                R_diff = R_t - I_sq
                residuals.append(dW[:d_sq, :d_sq] - R_diff)
            rho_sq = mx.zeros_like(residuals[0])
            for r in residuals:
                rho_sq = rho_sq + r
            rho_sq = rho_sq / T
            dW_sq = (R_merged - I_sq) + rho_sq  # (d_sq, d_sq)
            dW_mean = mx.zeros_like(deltas[0])
            for dW in deltas:
                dW_mean = dW_mean + dW
            dW_mean = dW_mean / T
            if d_in == d_sq and d_out == d_sq:
                dW_final = dW_sq
            elif d_in <= d_out:
                dW_final = mx.concatenate([dW_sq, dW_mean[:, d_sq:]], axis=1)
            else:
                dW_final = mx.concatenate([dW_sq, dW_mean[d_sq:, :]], axis=0)
        else:
            W_0 = base_W0_per_layer[key].astype(mx.float32)
            residuals = []
            for dW, R_t in zip(deltas, R_per):
                rho = (W_0 + dW.T) - (R_t @ W_0)  # (d_out, d_in)
                residuals.append(rho)
            rho_merged = mx.zeros_like(residuals[0])
            for r in residuals:
                rho_merged = rho_merged + r
            rho_merged = rho_merged / T
            W_final = R_merged @ W_0 + rho_merged
            dW_final = (W_final - W_0).T  # back to (d_in, d_out)

        mx.eval(dW_final)
        fused[key] = dW_final.astype(mx.bfloat16)
    return fused

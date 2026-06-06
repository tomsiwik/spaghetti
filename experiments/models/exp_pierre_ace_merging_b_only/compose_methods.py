"""ACE-Merging adapted to Pierre's shared-A B-only architecture.

Per arxiv 2603.02945 + reference impl unravel-xu/ACE-Merging
(`src/merge/strategy.py`). The closed-form merge:

    W̄ = (Σ_t W̃_t · Σ̂_{t,reg}) · (Σ_t Σ̂_{t,reg} + C_agg)^{-1}

where Σ̂_t is per-task input covariance derived from ΔW alone (no calibration
data; Theorem 1 in the paper). For Pierre's shared-A regime: ΔW_t = scale · A · B_t,
so the input covariance reduces to a B-only operation:

    W̃_t        = ΔW_t  − col_mean(ΔW_t)        (centered task vector)
    Σ̂_t (raw)  = W̃_t^T · W̃_t                   shape (d_in, d_in)
    Σ̂_t (reg)  = Σ̂_t / tr(Σ̂_t) + ε·I            (heterogeneity branch)
                 OR  Σ̂_t + ε·I                  (homogeneous branch)

    γ-flag    = Var_t[log ‖ΔW_t‖²] / E_t[log ‖ΔW_t‖²]² > τ=0.3
    C_agg     = ε·I  (simplified per released code; paper has rank-1 broadcast)

We can NOT collapse to a B-only fused-B output because the closed-form merge
operates on full ΔW. So this experiment uses the **fused_delta** kind:
materialize ΔW_t = scale · A · B_t per layer, run ACE merge, install the
fused result via `_FusedDeltaLinear`. The shared-A constraint is preserved
because we still load only B from disk; the fusion is a runtime operation.

For the optional spectral refinement when γ > τ:
    U, S, V^T   = SVD(ΔW_fused)
    k           = int(len(S) * 0.3)
    σ_iso       = mean(S[:k])
    refinement  = σ_iso · (U[:, :k] @ V[:k, :])
    ΔW_fused   += refinement
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def compose_ace_merging_b_only(
    adapter_states, A_dict, scale,
    *, eps: float = 1e-2, tau: float = 0.3, k_frac: float = 0.3,
    force_disable_spectral: bool = False,
):
    """ACE merge for Pierre's shared-A architecture, returning fused deltas.

    Args:
        adapter_states: list of K state dicts {layer_key: {a, b, scale}}.
        A_dict: shared-A dict {layer_key: A}. (used to materialize per-task ΔW)
        scale: PoLAR scale factor (LORA_SCALE in polar_train).
        eps: regularization on the inverse — `Σ̂_{reg} = Σ̂ + eps·I` in the
             homogeneous branch, or scaled by trace in heterogeneous branch.
        tau: heterogeneity threshold for spectral refinement trigger.
        k_frac: fraction of top singular values to isotropize during refinement.
        force_disable_spectral: testing knob — set True to force-skip spectral.
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

        # Materialize per-adapter ΔW_t and centered W̃_t
        deltas = []
        log_norms_sq = []
        for entry in per_adapter:
            A_t = entry["a"].astype(mx.float32)
            B_t = entry["b"].astype(mx.float32)
            s_t = float(entry.get("scale", scale))
            dW = s_t * (A_t @ B_t)
            mx.eval(dW)
            deltas.append(dW)
            n2 = mx.sum(dW * dW)
            mx.eval(n2)
            v = float(n2.item()) + 1e-12
            log_norms_sq.append(np.log(v))

        # Heterogeneity flag
        ln = np.array(log_norms_sq, dtype=np.float64)
        gamma_num = float(np.var(ln))
        gamma_den = float(np.mean(ln) ** 2 + 1e-12)
        gamma = gamma_num / gamma_den
        flag = (gamma > tau)

        # Per-task centered task vector W̃_t and covariance Σ̂_t
        Sigmas = []
        for dW in deltas:
            col_mean = mx.mean(dW, axis=0, keepdims=True)  # (1, d_out)
            W_tilde = dW - col_mean                         # (d_in, d_out)
            Sigma_raw = W_tilde.T @ W_tilde                 # (d_out, d_out)
            mx.eval(Sigma_raw)
            tr = mx.sum(mx.diagonal(Sigma_raw)) + 1e-12
            mx.eval(tr)
            tr_v = float(tr.item())
            d_out = Sigma_raw.shape[0]
            if flag:
                Sigma_t = Sigma_raw / tr_v
                eps_t = eps / tr_v
            else:
                Sigma_t = Sigma_raw
                eps_t = eps
            Sigma_reg = Sigma_t + eps_t * mx.eye(d_out, dtype=mx.float32)
            Sigmas.append(Sigma_reg)

        # Closed-form: W̄ = (Σ_t W̃_t · Σ̂_{t,reg}) · (Σ_t Σ̂_{t,reg} + C_agg)^{-1}
        # We use the centered W̃_t (not dW) to remain faithful to ACE's derivation.
        numerator = mx.zeros_like(deltas[0])  # (d_in, d_out)
        denominator = mx.zeros_like(Sigmas[0])  # (d_out, d_out)
        for dW, Sigma_reg in zip(deltas, Sigmas):
            col_mean = mx.mean(dW, axis=0, keepdims=True)
            W_tilde = dW - col_mean
            numerator = numerator + (W_tilde @ Sigma_reg)
            denominator = denominator + Sigma_reg
        # C_agg = eps·I (released code's simplification)
        denominator = denominator + (eps * mx.eye(denominator.shape[0], dtype=mx.float32))
        mx.eval(numerator, denominator)

        # Solve via Cholesky-style inverse on CPU (numerically safer for small d_out)
        denom_inv = mx.linalg.inv(denominator, stream=mx.cpu)
        merged = numerator @ denom_inv
        mx.eval(merged)

        # Restore the column-mean contribution (uniform across t)
        col_mean_fused = mx.mean(mx.stack([mx.mean(dW, axis=0) for dW in deltas]), axis=0)
        merged = merged + col_mean_fused[None, :]

        # Optional spectral refinement when heterogeneity is high
        if flag and not force_disable_spectral:
            U, S, Vt = mx.linalg.svd(merged, stream=mx.cpu)
            k = max(1, int(S.shape[0] * k_frac))
            sigma_iso = mx.mean(S[:k])
            refinement = sigma_iso * (U[:, :k] @ Vt[:k, :])
            mx.eval(refinement)
            merged = merged + refinement

        mx.eval(merged)
        fused[key] = merged.astype(mx.bfloat16)
    return fused

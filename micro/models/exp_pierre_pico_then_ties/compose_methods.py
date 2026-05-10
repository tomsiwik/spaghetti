"""Pico calibration on B-stack, then TIES on materialized full deltas.

Combinatorial test: do the Pico (B-space SVD calibration) and TIES
(full-delta sign-aware merge) operations compose? Or does one absorb the
other (redundant)?

Pipeline:
  1. Apply Pico calibration to B's: B_t_calib = S · B_t
  2. Materialize ΔW_t = scale · A · B_t_calib
  3. Run TIES (Trim + Sign-Elect + Disjoint mean) on materialized deltas
  4. Install via _FusedDeltaLinear

If pico_then_ties beats both Pico+Fisher-Rao and TIES alone, the operations
are orthogonal and additive. If it matches the better of the two, they're
redundant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Import Pico calibration (without rescale; we don't want double rescale through TIES)
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

# Avoid the parent's __init__.py — direct file import to grab the function
from importlib.util import spec_from_file_location, module_from_spec  # noqa: E402

_pico_spec = spec_from_file_location(
    "_pico_compose",
    str(EXP_DIR.parent / "exp_pierre_pico_calibration" / "compose_methods.py"),
)
_pico_mod = module_from_spec(_pico_spec)
_pico_spec.loader.exec_module(_pico_mod)
compose_pico_then_fisher_rao = _pico_mod.compose_pico_then_fisher_rao  # noqa: E402


def _topk_mask(M: mx.array, keep_frac: float) -> mx.array:
    abs_M = mx.abs(M)
    T, D = M.shape
    k = max(1, int(D * keep_frac))
    sorted_abs = mx.sort(abs_M, axis=1)
    threshold = sorted_abs[:, -k:-k+1] if k < D else sorted_abs[:, :1]
    return abs_M >= threshold


def compose_pico_then_ties(
    adapter_states, A_dict, scale,
    *, keep_frac: float = 0.3, alpha_override: float | None = None,
):
    """Pico calibration → TIES on materialized deltas.

    The Pico calibration step operates on B-matrices; we then materialize
    full deltas using the calibrated B's and apply TIES.

    Args:
        adapter_states: list of K state dicts.
        A_dict: shared-A dict (used to build the per-adapter B's and materialize deltas).
        scale: PoLAR scale factor.
        keep_frac: TIES keep fraction (default 0.3 per paper).
        alpha_override: testing knob — if not None, sets all Pico α_j to this value.
    """
    K = len(adapter_states)
    if K == 0:
        return {}

    # Build per-adapter B-dicts (Pico operates on these)
    B_lists_per_adapter: list[dict] = []
    for st in adapter_states:
        B_lists_per_adapter.append({k: v["b"] for k, v in st.items() if "b" in v})

    # === Step 1: Pico calibration ===
    # We invoke Pico WITHOUT post-rescale, because TIES will do its own merge.
    # But our Pico API only returns the merged B (already rescaled+merged).
    # We need Pico's *per-adapter calibrated B's*, not the merged result.
    # So we re-run the calibration step inline here.

    # Replicate Pico's per-key calibration step, returning per-adapter calibrated B's.
    all_keys: set[str] = set()
    for ab in B_lists_per_adapter:
        all_keys.update(ab.keys())

    calibrated_B_per_adapter: list[dict] = [{} for _ in range(K)]

    for key in sorted(all_keys):
        Bs = [ab[key].astype(mx.float32) for ab in B_lists_per_adapter if key in ab]
        T = len(Bs)
        if T == 0:
            continue
        if T == 1:
            calibrated_B_per_adapter[0][key] = Bs[0]
            continue

        # Pico calibration matrix S
        B_T_per = [b.T for b in Bs]
        B_all = mx.concatenate(B_T_per, axis=1)  # (d_out, T·r)
        U, sigma, _ = mx.linalg.svd(B_all, stream=mx.cpu)
        sigma2 = sigma ** 2
        total = mx.sum(sigma2) + 1e-12
        s = sigma2 / total
        alpha = 1.0 / (1.0 + (T - 1) * s)
        if alpha_override is not None:
            alpha = mx.full(alpha.shape, float(alpha_override), dtype=alpha.dtype)
        m = sigma.shape[0]
        U_m = U[:, :m]
        U_scaled = U_m * (alpha - 1.0)[None, :]
        d_out = B_all.shape[0]
        S_calib = mx.eye(d_out, dtype=mx.float32) + (U_scaled @ U_m.T)
        mx.eval(S_calib)

        for t_idx, b in enumerate(Bs):
            calibrated_B_per_adapter[t_idx][key] = b @ S_calib.T

    # === Step 2: Materialize calibrated deltas + TIES merge ===
    fused: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        per_adapter_calib = [
            calibrated_B_per_adapter[i].get(key)
            for i in range(K) if key in calibrated_B_per_adapter[i]
        ]
        T = len(per_adapter_calib)
        if T == 0:
            continue

        # Look up per-adapter A and scale
        deltas = []
        for t_idx in range(K):
            if key not in calibrated_B_per_adapter[t_idx]:
                continue
            entry = adapter_states[t_idx][key]
            A_t = entry["a"].astype(mx.float32)
            s_t = float(entry.get("scale", scale))
            B_calib = calibrated_B_per_adapter[t_idx][key]
            dW = s_t * (A_t @ B_calib)
            mx.eval(dW)
            deltas.append(dW)

        if len(deltas) == 1:
            fused[key] = deltas[0].astype(mx.bfloat16)
            continue

        d_in, d_out = deltas[0].shape
        D = d_in * d_out
        T_flat = mx.stack([dW.reshape(-1) for dW in deltas])  # (T, D)
        keep = _topk_mask(T_flat, keep_frac)
        T_trim = T_flat * keep.astype(T_flat.dtype)
        gamma = mx.sign(mx.sum(T_trim, axis=0))
        agree = (mx.sign(T_trim) == gamma[None, :]) & (T_trim != 0)
        agree_f = agree.astype(T_flat.dtype)
        masked = T_trim * agree_f
        n_agree = mx.sum(agree_f, axis=0)
        merged = (mx.sum(masked, axis=0) / mx.maximum(n_agree, 1.0)).reshape(d_in, d_out)
        mx.eval(merged)
        fused[key] = merged.astype(mx.bfloat16)
    return fused

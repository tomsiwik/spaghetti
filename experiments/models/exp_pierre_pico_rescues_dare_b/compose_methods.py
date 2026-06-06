"""Test whether Pico calibration rescues the failed B-space DARE.

In `exp_pierre_dare_b_vs_fisher_rao`, B-space DARE collapsed to 55.3%
(vs Fisher-Rao 64.7%). Diagnosis: dropping 90% of B-entries randomly
hollowed out concentrated B-matrices (especially medical: 30% MedQA).

Pico (arxiv 2604.16826) explicitly diagnoses that "merge interference is
concentrated in B" and dampens over-shared B-directions. Question:
**does Pico calibration condition B's so that B-space DARE works?**

Pipeline:
  1. Apply Pico calibration: B_t_calib = S · B_t (no rescale yet)
  2. Apply DARE on calibrated B: drop 90% randomly, rescale 1/(1-0.9)
  3. Mean over t
  4. Norm-rescale to mean source norm of ORIGINAL (pre-calib) B's
  5. Returns B-dict (Pierre's compose API preserved)
"""
from __future__ import annotations

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import mlx.core as mx
import numpy as np

EXP_DIR = Path(__file__).resolve().parent

# Reuse Pico's calibration logic by importing
_pico_spec = spec_from_file_location(
    "_pico_compose",
    str(EXP_DIR.parent / "exp_pierre_pico_calibration" / "compose_methods.py"),
)
_pico_mod = module_from_spec(_pico_spec)
_pico_spec.loader.exec_module(_pico_mod)


def compose_pico_rescues_dare_b(
    B_lists, A_dict,
    *, drop_rate: float = 0.9, seed: int = 42,
):
    """Pico calibration → B-space DARE → mean → norm-rescale.

    Args:
        B_lists: list of K B-dicts.
        A_dict: shared-A dict (unused).
        drop_rate: DARE drop rate (default 0.9 matches B-space DARE prior).
        seed: RNG seed for the drop masks.
    """
    if len(B_lists) == 1:
        return B_lists[0]
    K = len(B_lists)
    keep = 1.0 - drop_rate

    all_keys: set[str] = set()
    for ab in B_lists:
        all_keys.update(ab.keys())

    rng = np.random.default_rng(seed)
    composed: dict[str, mx.array] = {}

    for key in sorted(all_keys):
        Bs = [ab[key].astype(mx.float32) for ab in B_lists if key in ab]
        T = len(Bs)
        if T == 0:
            continue
        if T == 1:
            composed[key] = Bs[0].astype(mx.bfloat16)
            continue

        # Pico calibration matrix
        B_T_per = [b.T for b in Bs]
        B_all = mx.concatenate(B_T_per, axis=1)  # (d_out, T·r)
        U, sigma, _ = mx.linalg.svd(B_all, stream=mx.cpu)
        sigma2 = sigma ** 2
        total = mx.sum(sigma2) + 1e-12
        s = sigma2 / total
        alpha = 1.0 / (1.0 + (T - 1) * s)
        m = sigma.shape[0]
        U_m = U[:, :m]
        U_scaled = U_m * (alpha - 1.0)[None, :]
        d_out = B_all.shape[0]
        S_calib = mx.eye(d_out, dtype=mx.float32) + (U_scaled @ U_m.T)
        mx.eval(S_calib)

        # Apply Pico calibration per adapter
        Bs_calib = [b @ S_calib.T for b in Bs]

        # Apply B-space DARE on calibrated B's
        Bs_dare = []
        for b in Bs_calib:
            mask_np = (rng.random(b.shape) < keep).astype(np.float32)
            mask = mx.array(mask_np)
            b_dare = (b * mask) / keep
            Bs_dare.append(b_dare)

        # Mean
        merged = mx.zeros_like(Bs_dare[0])
        for b in Bs_dare:
            merged = merged + b
        merged = merged / T

        # Norm rescale to mean source norm of ORIGINAL B's
        orig_norms = mx.stack([mx.linalg.norm(b.reshape(-1)) for b in Bs])
        mean_source_norm = mx.mean(orig_norms)
        mean_norm = mx.linalg.norm(merged.reshape(-1))
        mx.eval(mean_source_norm, mean_norm)
        if mean_norm.item() > 1e-8:
            merged = merged * (mean_source_norm / mean_norm)

        composed[key] = merged.astype(mx.bfloat16)
    return composed

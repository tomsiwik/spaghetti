"""Composition methods under test.

Faithful copy of Pierre's `pierre_model/compose.py` (Fisher-Rao) plus the
new `dare_b` candidate. The Fisher-Rao block is verbatim from Pierre at
the time of writing (2026-05-05); if Pierre's compose.py changes, update
this file and re-run.

This experiment must NOT import from pierre — it must copy. That keeps
the measurement reproducible against a frozen snapshot of pierre's logic
even if pierre evolves.
"""
from __future__ import annotations
from typing import Optional, Literal

import mlx.core as mx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# M1: Fisher-Rao (Pierre's current default — VERBATIM from compose.py)
# ─────────────────────────────────────────────────────────────────────────

def compose_fisher_rao(
    adapter_Bs: list[dict[str, mx.array]],
    weights: Optional[list[float]] = None,
) -> dict[str, mx.array]:
    """Norm-rescaled weighted average of adapter B-matrices."""
    if len(adapter_Bs) == 1:
        return adapter_Bs[0]
    if weights is None:
        weights = [1.0 / len(adapter_Bs)] * len(adapter_Bs)

    all_keys: set[str] = set()
    for ab in adapter_Bs:
        all_keys.update(ab.keys())

    composed: dict[str, mx.array] = {}
    for key in all_keys:
        tensors = [ab[key] for ab in adapter_Bs if key in ab]
        w = weights[: len(tensors)]
        composed[key] = _norm_rescaled_average(tensors, w)
    return composed


def _norm_rescaled_average(
    tensors: list[mx.array], weights: list[float],
) -> mx.array:
    """Karcher-mean closed form: weighted mean rescaled to mean-source-norm."""
    if len(tensors) == 1:
        return tensors[0]

    w_sum = sum(weights)
    mean = sum(
        t.astype(mx.float32) * (w / w_sum)
        for t, w in zip(tensors, weights)
    )

    norms = mx.stack([
        mx.linalg.norm(t.reshape(-1).astype(mx.float32))
        for t in tensors
    ])
    source_norm = mx.mean(norms)
    mean_norm = mx.linalg.norm(mean.reshape(-1))
    mx.eval(source_norm, mean_norm)

    if mean_norm.item() > 1e-8:
        return (mean * (source_norm / mean_norm)).astype(mx.bfloat16)
    return mean.astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────
# M2: DARE-on-B (NEW candidate — B-space drop+rescale then mean)
# ─────────────────────────────────────────────────────────────────────────

def compose_dare_b(
    adapter_Bs: list[dict[str, mx.array]],
    weights: Optional[list[float]] = None,
    drop_rate: float = 0.9,
    seed: int = 42,
) -> dict[str, mx.array]:
    """B-space DARE: per-adapter drop+rescale on B-entries, then weighted mean.

    Element-wise Bernoulli mask with `drop_rate` zeros, surviving entries
    rescaled by 1/(1-drop_rate). Per the DARE paper (arxiv 2311.03099),
    `E[DARE(B)] = B` — but with reduced variance for the mean operator
    when applied to a sum across multiple adapters.

    Args:
        adapter_Bs: list of K B-dicts, same shape as `compose_fisher_rao`.
        weights: per-adapter mixing weights. Default uniform 1/K.
        drop_rate: fraction of B-entries zeroed per adapter (default 0.9
            matches research's measured optimum).
        seed: RNG seed for reproducible drop masks across runs.
    """
    if len(adapter_Bs) == 1:
        return adapter_Bs[0]
    if weights is None:
        weights = [1.0 / len(adapter_Bs)] * len(adapter_Bs)

    all_keys: set[str] = set()
    for ab in adapter_Bs:
        all_keys.update(ab.keys())

    rng = np.random.default_rng(seed)
    keep_rate = 1.0 - drop_rate
    composed: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        tensors = [ab[key] for ab in adapter_Bs if key in ab]
        w = weights[: len(tensors)]
        composed[key] = _dare_b_average(tensors, w, drop_rate, rng)
    return composed


def _dare_b_average(
    tensors: list[mx.array],
    weights: list[float],
    drop_rate: float,
    rng: np.random.Generator,
) -> mx.array:
    """Per-tensor DARE drop+rescale, then weighted mean."""
    keep = 1.0 - drop_rate
    w_sum = sum(weights)
    out = None
    for t, w in zip(tensors, weights):
        t_f32 = t.astype(mx.float32)
        # Drop mask in numpy for reproducibility, transfer to MLX
        mask_np = (rng.random(t.shape) < keep).astype(np.float32)
        mask = mx.array(mask_np)
        # DARE: zero (1 - keep) fraction, rescale survivors by 1/keep
        t_dare = (t_f32 * mask) / keep
        contrib = t_dare * (w / w_sum)
        out = contrib if out is None else out + contrib
    return out.astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────
# M3: Full-delta DARE (research upper bound — per-adapter A, fused delta)
# ─────────────────────────────────────────────────────────────────────────

def compose_full_delta_dare(
    adapter_states: list[dict[str, dict[str, mx.array]]],
    weights: Optional[list[float]] = None,
    drop_rate: float = 0.9,
    seed: int = 42,
) -> dict[str, mx.array]:
    """Per-adapter full-delta DARE — the research path.

    Inputs are (A, B, scale) per adapter per layer key. Computes
    ΔW_k = scale_k · A_k @ B_k, applies element-wise DARE, then weighted mean.
    Returns a fused delta dict (shape d_in × d_out per key) — NOT a B-dict.

    Args:
        adapter_states: list of K dicts, each mapping
            layer_key → {"a": [d_in, r], "b": [r, d_out], "scale": float}.
        weights: per-adapter mixing weights. Default uniform 1/K.
        drop_rate: DARE drop rate (default 0.9).
        seed: RNG seed.
    """
    K = len(adapter_states)
    if K == 0:
        return {}
    if weights is None:
        weights = [1.0 / K] * K

    all_keys: set[str] = set()
    for st in adapter_states:
        all_keys.update(st.keys())

    rng = np.random.default_rng(seed)
    keep = 1.0 - drop_rate
    w_sum = sum(weights)

    fused: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        contributions = []
        per_adapter = [st[key] for st in adapter_states if key in st]
        for entry, w in zip(per_adapter, weights[: len(per_adapter)]):
            A = entry["a"].astype(mx.float32)
            B = entry["b"].astype(mx.float32)
            scale = float(entry["scale"])
            delta = scale * (A @ B)
            mx.eval(delta)
            mask_np = (rng.random(delta.shape) < keep).astype(np.float32)
            mask = mx.array(mask_np)
            delta_dare = (delta * mask) / keep
            contributions.append(delta_dare * (w / w_sum))
        fused_delta = contributions[0]
        for c in contributions[1:]:
            fused_delta = fused_delta + c
        mx.eval(fused_delta)
        fused[key] = fused_delta.astype(mx.bfloat16)
    return fused


# ─────────────────────────────────────────────────────────────────────────
# Method dispatcher
# ─────────────────────────────────────────────────────────────────────────

ComposeMethod = Literal["fisher_rao", "dare_b", "dare_full_delta", "single_best"]

__all__ = [
    "compose_fisher_rao",
    "compose_dare_b",
    "compose_full_delta_dare",
    "ComposeMethod",
]

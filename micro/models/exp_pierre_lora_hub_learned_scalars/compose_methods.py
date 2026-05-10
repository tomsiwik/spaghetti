"""LoRA Hub-style learned scalar weights for shared-A B-only composition.

Per arxiv 2307.13269 ("LoRA Hub"). Under shared-A, LoRA Hub's formula
    ΔW_merged = (Σ w_i A_i)(Σ w_i B_i)
collapses to
    ΔW_merged = (Σ w_i) · A_shared · (Σ w_i B_i)
which means we only need to learn K scalar weights `w_i` for `Σ w_i B_i`.
This is a strict superset of Fisher-Rao (Fisher-Rao computes `w_i = 1/K`
analytically; LoRA Hub *learns* them via gradient-free search).

Functions as the **architectural ceiling** test for shared-A B-only:
    if even *learned* scalar weights can't beat Fisher-Rao, the shared-A
    architecture has fundamentally hit its expressive capacity ceiling.

We use scipy's `differential_evolution` instead of Nevergrad NGOpt — same
gradient-free black-box class, no extra dependency. Budget = 40 evaluations
matching LoRA Hub paper.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import mlx.core as mx
import numpy as np


def compose_weighted_mean_b(
    B_lists: list[dict],
    A_dict: dict,
    *, weights: Optional[list[float]] = None,
    rescale_to_mean_norm: bool = True,
):
    """Σ w_i B_i with optional norm rescaling.

    Internal helper used during black-box optimization. Returns a B-dict
    matching Pierre's compose API.
    """
    if len(B_lists) == 1:
        return B_lists[0]
    if weights is None:
        weights = [1.0 / len(B_lists)] * len(B_lists)

    all_keys: set[str] = set()
    for ab in B_lists:
        all_keys.update(ab.keys())

    composed: dict[str, mx.array] = {}
    for key in sorted(all_keys):
        tensors = [ab[key].astype(mx.float32) for ab in B_lists if key in ab]
        K_used = len(tensors)
        ws = weights[:K_used]
        merged = mx.zeros_like(tensors[0])
        for t, w in zip(tensors, ws):
            merged = merged + (t * float(w))
        if rescale_to_mean_norm:
            orig_norms = mx.stack([mx.linalg.norm(t.reshape(-1)) for t in tensors])
            mean_source_norm = mx.mean(orig_norms)
            mean_norm = mx.linalg.norm(merged.reshape(-1))
            mx.eval(mean_source_norm, mean_norm)
            if mean_norm.item() > 1e-8:
                merged = merged * (mean_source_norm / mean_norm)
        composed[key] = merged.astype(mx.bfloat16)
    return composed


def compose_lora_hub_learned_scalars(
    B_lists: list[dict],
    A_dict: dict,
    *,
    objective_fn: Callable[[list[float]], float],
    bounds: tuple[float, float] = (-1.5, 1.5),
    budget: int = 40,
    seed: int = 42,
    rescale_to_mean_norm: bool = True,
):
    """Learn K weights via scipy differential_evolution, return composed B.

    Args:
        B_lists: list of K B-dicts.
        A_dict: shared-A dict (passed through; not used in computation).
        objective_fn: callable taking list of K floats, returning a SCALAR
            cost to minimize (e.g. negative validation accuracy). Provided
            by the experiment runner — black-boxes the model + tokenizer
            + a small validation set.
        bounds: per-coefficient bounds (paper uses (-1.5, 1.5)).
        budget: max number of evaluations (paper uses 40).
        seed: RNG seed for reproducibility.
        rescale_to_mean_norm: whether to apply Pierre's norm rescaling.

    Returns:
        (composed_B_dict, learned_weights, optimization_history)
    """
    from scipy.optimize import differential_evolution

    K = len(B_lists)
    bounds_list = [bounds] * K

    history = []

    def wrapped_obj(w):
        cost = objective_fn(list(w))
        history.append({"weights": list(w), "cost": cost})
        return cost

    # tiny budget — popsize × maxiter ≈ 40
    result = differential_evolution(
        wrapped_obj, bounds_list,
        maxiter=8, popsize=5,  # 8 × 5 = 40 evals
        seed=seed, polish=False, tol=1e-3, init="sobol",
    )

    learned_weights = [float(w) for w in result.x]
    composed = compose_weighted_mean_b(
        B_lists, A_dict,
        weights=learned_weights,
        rescale_to_mean_norm=rescale_to_mean_norm,
    )
    return composed, learned_weights, history

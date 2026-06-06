"""Gram-Schmidt orthogonalization for LoRA delta merging.

Given N LoRA adapters with deltas D_1, ..., D_N, some pairs may have significant
cosine similarity (e.g., math-medical at cos=0.70). Naive additive merging
causes interference in the overlapping subspace.

Gram-Schmidt projects out overlapping components sequentially:
  D_1' = D_1  (first expert keeps full signal)
  D_k' = D_k - sum_{i<k} proj(D_k, D_i')  for k = 2, ..., N

where proj(u, v) = (u . v / v . v) * v is the vector projection.

Each D_k' contains ONLY the novel contribution of expert k that is
orthogonal to all previous experts. The merged model is:
  W_merged = W_base + D_1' + D_2' + ... + D_N'

This guarantees zero interference between merged expert deltas in weight space.

The cost is signal loss: if expert k overlaps heavily with earlier experts,
most of its delta is projected out. The kill criterion is: if the signal
retained (||D_k'|| / ||D_k||) drops below 50%, the projection is too aggressive.
"""

import mlx.core as mx
import numpy as np


def flatten_delta_dict(delta_dict: dict) -> np.ndarray:
    """Flatten a delta dict {(layer, sublayer): mx.array} to a single numpy vector."""
    parts = []
    for key in sorted(delta_dict.keys()):
        arr = np.array(delta_dict[key])
        parts.append(arr.flatten())
    return np.concatenate(parts)


def unflatten_delta_dict(flat: np.ndarray, template_dict: dict) -> dict:
    """Reshape a flat numpy vector back to a delta dict matching template shapes."""
    result = {}
    offset = 0
    for key in sorted(template_dict.keys()):
        shape = np.array(template_dict[key]).shape
        size = int(np.prod(shape))
        result[key] = mx.array(flat[offset:offset + size].reshape(shape))
        offset += size
    return result


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def gram_schmidt_orthogonalize(delta_dicts: list[dict], names: list[str] | None = None
                                ) -> tuple[list[dict], dict]:
    """Apply Gram-Schmidt orthogonalization to a list of LoRA delta dicts.

    Args:
        delta_dicts: list of N delta dicts {(layer, sublayer): mx.array}
        names: optional list of N expert names for reporting

    Returns:
        orthogonalized: list of N orthogonalized delta dicts
        report: dict with diagnostics (signal retention, pairwise cosines, etc.)
    """
    N = len(delta_dicts)
    if names is None:
        names = [f"expert_{i}" for i in range(N)]

    template = delta_dicts[0]

    # Flatten all deltas to vectors
    flat_originals = [flatten_delta_dict(d) for d in delta_dicts]

    # Compute pairwise cosine similarities BEFORE orthogonalization
    pre_cosines = {}
    for i in range(N):
        for j in range(i + 1, N):
            cos = cosine_sim(flat_originals[i], flat_originals[j])
            pre_cosines[(names[i], names[j])] = cos

    # Gram-Schmidt process
    flat_ortho = []
    for k in range(N):
        v = flat_originals[k].copy()
        for i in range(len(flat_ortho)):
            # Project out component along orthogonalized vector i
            e_i = flat_ortho[i]
            dot_ve = np.dot(v, e_i)
            dot_ee = np.dot(e_i, e_i)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e_i
        flat_ortho.append(v)

    # Compute pairwise cosine similarities AFTER orthogonalization
    post_cosines = {}
    for i in range(N):
        for j in range(i + 1, N):
            cos = cosine_sim(flat_ortho[i], flat_ortho[j])
            post_cosines[(names[i], names[j])] = cos

    # Signal retention: ||D_k'|| / ||D_k||
    signal_retention = {}
    for k in range(N):
        orig_norm = np.linalg.norm(flat_originals[k])
        ortho_norm = np.linalg.norm(flat_ortho[k])
        if orig_norm > 1e-12:
            signal_retention[names[k]] = float(ortho_norm / orig_norm)
        else:
            signal_retention[names[k]] = 0.0

    # Unflatten back to delta dicts
    orthogonalized = [unflatten_delta_dict(flat_ortho[k], template) for k in range(N)]

    report = {
        "n_experts": N,
        "names": names,
        "pre_cosines": {f"{a} vs {b}": v for (a, b), v in pre_cosines.items()},
        "post_cosines": {f"{a} vs {b}": v for (a, b), v in post_cosines.items()},
        "signal_retention": signal_retention,
        "signal_retention_min": min(signal_retention.values()),
        "max_pre_cosine": max(abs(v) for v in pre_cosines.values()) if pre_cosines else 0.0,
        "max_post_cosine": max(abs(v) for v in post_cosines.values()) if post_cosines else 0.0,
    }

    return orthogonalized, report


def merge_with_gram_schmidt(delta_dicts: list[dict], names: list[str] | None = None
                             ) -> tuple[dict, dict]:
    """Gram-Schmidt orthogonalize then sum all deltas.

    Returns:
        merged: single delta dict (sum of orthogonalized deltas)
        report: diagnostics from gram_schmidt_orthogonalize
    """
    ortho_dicts, report = gram_schmidt_orthogonalize(delta_dicts, names)

    # Sum all orthogonalized deltas
    keys = sorted(ortho_dicts[0].keys())
    merged = {}
    for k in keys:
        merged[k] = sum(d[k] for d in ortho_dicts)

    return merged, report


def merge_gs_average(delta_dicts: list[dict], names: list[str] | None = None
                      ) -> tuple[dict, dict]:
    """Gram-Schmidt orthogonalize then AVERAGE (1/N) all deltas.

    This is the correct merge strategy: orthogonalize to remove interference,
    then scale by 1/N to keep the perturbation magnitude appropriate.

    Returns:
        merged: single delta dict (average of orthogonalized deltas)
        report: diagnostics from gram_schmidt_orthogonalize
    """
    ortho_dicts, report = gram_schmidt_orthogonalize(delta_dicts, names)
    N = len(ortho_dicts)

    keys = sorted(ortho_dicts[0].keys())
    merged = {}
    for k in keys:
        merged[k] = sum(d[k] for d in ortho_dicts) / N

    return merged, report


def merge_naive_sum(delta_dicts: list[dict]) -> dict:
    """Naive additive merge: sum all deltas without orthogonalization.

    This is the baseline: W_merged = W_base + D_1 + D_2 + ... + D_N.
    With orthogonal deltas (cos~0), this is equivalent to GS merge.
    With non-orthogonal deltas (cos>0), overlapping components are double-counted.
    """
    keys = sorted(delta_dicts[0].keys())
    merged = {}
    for k in keys:
        merged[k] = sum(d[k] for d in delta_dicts)
    return merged

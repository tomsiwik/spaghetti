"""DC-Merge: the full merging pipeline.

Implements Algorithm 1 from "DC-Merge: Improving Model Merging with Directional Consistency"
(arXiv 2603.06242, Zhang et al. 2025).

All operations use mlx.core. No PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from pierre.merge.dc_merge.src.model import (
    truncated_svd,
    energy_smoothing,
    construct_cover_basis,
    project_to_cover_space,
    project_to_param_space,
)


# ---------------------------------------------------------------------------
# Element-wise merging methods (applied in cover space)
# ---------------------------------------------------------------------------

def _merge_ta(projections: list[mx.array]) -> mx.array:
    """Task Arithmetic: simple element-wise average of projections.

    Reference: Ilharco et al. "Editing Models with Task Arithmetic" (ICLR 2023).
    """
    stacked = mx.stack(projections, axis=0)  # (T, k, k)
    return mx.mean(stacked, axis=0)


def _merge_ties(projections: list[mx.array], top_k: float = 0.1) -> mx.array:
    """TIES-Merging: Trim, Elect Sign, Disjoint Merge.

    Reference: Yadav et al. "Resolving Interference When Merging Models" (NeurIPS 2023).

    Steps:
        1. Trim: zero out the smallest (1 - top_k) fraction of values per task.
        2. Elect sign: for each element, pick the sign with greatest total magnitude.
        3. Disjoint merge: average only values whose sign matches the elected sign.

    Args:
        projections: List of T projection matrices, each (k, k).
        top_k: Fraction of elements to retain per task (paper uses 0.1).

    Returns:
        Merged matrix (k, k).
    """
    T = len(projections)

    # Step 1: Trim — keep only top_k fraction by magnitude per task
    trimmed = []
    for M in projections:
        flat = mx.reshape(M, (-1,))
        n_keep = max(1, int(flat.shape[0] * top_k))
        # Get threshold: the n_keep-th largest absolute value
        abs_flat = mx.abs(flat)
        # Sort descending, take the n_keep-th value as threshold
        sorted_vals = mx.sort(abs_flat)
        threshold = sorted_vals[-(n_keep)]
        mask = abs_flat >= threshold
        trimmed.append(M * mask.reshape(M.shape))
    mx.eval(*trimmed)

    stacked = mx.stack(trimmed, axis=0)  # (T, k, k)

    # Step 2: Elect sign — majority sign weighted by magnitude
    sum_vals = mx.sum(stacked, axis=0)  # (k, k)
    elected_sign = mx.sign(sum_vals)  # +1, 0, or -1

    # Step 3: Disjoint merge — average values matching elected sign
    # For each element, only include tasks whose sign matches
    signs = mx.sign(stacked)  # (T, k, k)
    elected_expanded = mx.expand_dims(elected_sign, axis=0)  # (1, k, k)
    agreement = (signs == elected_expanded)  # (T, k, k) bool

    # Masked sum / count
    masked = stacked * agreement
    count = mx.sum(agreement.astype(mx.float32), axis=0)
    count = mx.maximum(count, mx.array(1.0))  # avoid div by zero

    return mx.sum(masked, axis=0) / count


# ---------------------------------------------------------------------------
# Full DC-Merge pipeline
# ---------------------------------------------------------------------------

def dc_merge(
    task_vectors: list[mx.array],
    rank: int = 16,
    smoothing: str = "average",
    rho: float = 5.0,
    merge_method: str = "ties",
    ties_top_k: float = 0.1,
    use_mask: bool = True,
    alpha: float = 1.0,
    base_weights: mx.array | None = None,
) -> mx.array:
    """Run the full DC-Merge algorithm (Algorithm 1).

    Steps:
        1. For each task vector, compute truncated SVD, smooth singular values,
           and reconstruct the energy-balanced task vector.
        2. Construct shared orthonormal cover basis via whitening.
        3. Project each smoothed task vector onto cover space.
        4. Merge projections using TA or TIES in cover space.
        5. Project merged result back to parameter space with structural mask.
        6. Optionally add to base weights with rescaling coefficient alpha.

    Args:
        task_vectors: List of T task vectors, each (m, n).
                      tau_i = theta_finetuned_i - theta_base
        rank: SVD truncation rank r. For LoRA, use LoRA rank.
        smoothing: Energy smoothing strategy ("average", "linear", "none").
        rho: Max/min ratio for linear smoothing.
        merge_method: "ta" (Task Arithmetic) or "ties" (TIES-Merging).
        ties_top_k: TIES top-k fraction (paper uses 0.1).
        use_mask: Apply block-diagonal structural mask.
        alpha: Rescaling coefficient for final merge.
        base_weights: If provided, returns W_base + alpha * Delta_merged.
                      If None, returns alpha * Delta_merged.

    Returns:
        Merged weights (m, n). Either full weights or scaled merged task vector.
    """
    T = len(task_vectors)
    if T == 0:
        raise ValueError("Need at least one task vector.")

    # --- Step 1: SVD + energy smoothing + reconstruction ---
    U_list = []
    V_list = []
    smoothed_deltas = []

    for i, delta in enumerate(task_vectors):
        U_r, S_r, V_r = truncated_svd(delta, rank)
        mx.eval(U_r, S_r, V_r)

        # Smooth singular values (Algorithm 1, line 6)
        S_smooth = energy_smoothing(S_r, strategy=smoothing, rho=rho)
        mx.eval(S_smooth)

        # Reconstruct energy-balanced task vector (Algorithm 1, line 7)
        # Delta_i = U_r @ diag(S_smooth) @ V_r^T
        delta_smooth = U_r * S_smooth.reshape(1, -1) @ V_r.T
        mx.eval(delta_smooth)

        U_list.append(U_r)
        V_list.append(V_r)
        smoothed_deltas.append(delta_smooth)

    # --- Step 2: Construct cover basis (Algorithm 1, lines 9-10) ---
    U_star, V_star = construct_cover_basis(U_list, V_list)
    mx.eval(U_star, V_star)

    # --- Step 3: Project each task vector onto cover space (line 12) ---
    projections = []
    for delta_smooth in smoothed_deltas:
        M_i = project_to_cover_space(delta_smooth, U_star, V_star)
        mx.eval(M_i)
        projections.append(M_i)

    # --- Step 4: Merge in cover space (line 13) ---
    if merge_method == "ta":
        M_merged = _merge_ta(projections)
    elif merge_method == "ties":
        M_merged = _merge_ties(projections, top_k=ties_top_k)
    else:
        raise ValueError(f"Unknown merge method: {merge_method!r}. Use 'ta' or 'ties'.")
    mx.eval(M_merged)

    # --- Step 5: Project back to parameter space (lines 15-16) ---
    delta_merged = project_to_param_space(
        M_merged, U_star, V_star,
        use_mask=use_mask,
        rank=rank,
        num_tasks=T,
    )
    mx.eval(delta_merged)

    # --- Step 6: Construct final weights ---
    result = alpha * delta_merged
    if base_weights is not None:
        result = base_weights + result
    mx.eval(result)

    return result


def dc_merge_from_config(
    task_vectors: list[mx.array],
    config_path: str | Path = "dc_merge/configs/base.yaml",
    base_weights: mx.array | None = None,
) -> mx.array:
    """Run DC-Merge with parameters loaded from a YAML config file.

    Args:
        task_vectors: List of T task vectors, each (m, n).
        config_path: Path to YAML config.
        base_weights: Optional base weight matrix.

    Returns:
        Merged weights (m, n).
    """
    import yaml

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return dc_merge(
        task_vectors=task_vectors,
        rank=cfg.get("rank", 16),
        smoothing=cfg.get("smoothing", "average"),
        rho=cfg.get("rho", 5.0),
        merge_method=cfg.get("merge_method", "ties"),
        ties_top_k=cfg.get("ties_top_k", 0.1),
        use_mask=cfg.get("use_mask", True),
        alpha=cfg.get("alpha", 1.0),
        base_weights=base_weights,
    )


# ---------------------------------------------------------------------------
# Utility: merge full model state dicts
# ---------------------------------------------------------------------------

def dc_merge_state_dicts(
    base_state: dict[str, mx.array],
    task_states: list[dict[str, mx.array]],
    rank: int = 16,
    smoothing: str = "average",
    rho: float = 5.0,
    merge_method: str = "ties",
    ties_top_k: float = 0.1,
    use_mask: bool = True,
    alpha: float = 1.0,
) -> dict[str, mx.array]:
    """Merge full model state dicts using DC-Merge.

    For each parameter key present in all task states:
        - 2D matrices: apply full DC-Merge algorithm.
        - 1D vectors (biases, layer norm): simple averaging (per paper Section E.2).

    Args:
        base_state: Base/pretrained model state dict.
        task_states: List of T fine-tuned model state dicts.
        rank, smoothing, rho, merge_method, ties_top_k, use_mask, alpha:
            DC-Merge hyperparameters (see dc_merge() docstring).

    Returns:
        Merged model state dict.
    """
    T = len(task_states)
    merged_state = {}

    # Find common keys
    all_keys = set(base_state.keys())
    for ts in task_states:
        all_keys &= set(ts.keys())

    for key in sorted(all_keys):
        base_w = base_state[key]

        if base_w.ndim == 2:
            # Compute task vectors: tau_i = W_i - W_base
            task_vecs = [ts[key] - base_w for ts in task_states]

            merged_w = dc_merge(
                task_vectors=task_vecs,
                rank=rank,
                smoothing=smoothing,
                rho=rho,
                merge_method=merge_method,
                ties_top_k=ties_top_k,
                use_mask=use_mask,
                alpha=alpha,
                base_weights=base_w,
            )
            merged_state[key] = merged_w

        elif base_w.ndim == 1:
            # 1D params (bias, layer norm): simple averaging per paper E.2
            task_vecs = mx.stack([ts[key] - base_w for ts in task_states], axis=0)
            avg_delta = mx.mean(task_vecs, axis=0)
            merged_state[key] = base_w + alpha * avg_delta
            mx.eval(merged_state[key])

        else:
            # Higher-dim tensors: simple average
            task_vecs = mx.stack([ts[key] - base_w for ts in task_states], axis=0)
            avg_delta = mx.mean(task_vecs, axis=0)
            merged_state[key] = base_w + alpha * avg_delta
            mx.eval(merged_state[key])

    return merged_state

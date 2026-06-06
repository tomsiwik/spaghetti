"""
Fisher-Rao Karcher Mean Merging Algorithm.

Implements the full merging pipeline from arXiv:2603.04972:
  Section 3.4: Blockwise spherical proxy with norm preservation.

Pipeline per parameter block:
  1. Flatten each model's parameter tensor to a vector
  2. Compute and store the L2 norm of each source (Section 3.4)
  3. Normalize each source to S^(d-1) (Section 3.4)
  4. Compute weighted Karcher mean on S^(d-1) (Section 3.3)
  5. Rescale by representative norm (mean/median/max) (Section 3.4)
  6. Reshape back to original tensor shape

For N=2 models, uses SLERP as the efficient special case (Section 3.3).

All operations use mlx.core (mx). No PyTorch.
"""

from typing import Optional
import mlx.core as mx
import yaml
from pathlib import Path

from .model import normalize, karcher_mean_spherical, slerp


def load_config(config_path: Optional[str] = None) -> dict:
    """Load merge configuration from YAML."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "base.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_block_norms(params_list: list[dict[str, mx.array]]) -> dict[str, list[mx.array]]:
    """Compute L2 norms for each parameter block across all models.

    Section 3.4: norms are stored for rescaling after spherical averaging.
    """
    norms = {}
    keys = list(params_list[0].keys())
    for key in keys:
        norms[key] = []
        for params in params_list:
            v = params[key].reshape(-1)
            norm = mx.sqrt(mx.sum(v * v) + 1e-8)
            norms[key].append(norm)
    return norms


def representative_norm(
    norms: list[mx.array],
    method: str = "mean",
) -> mx.array:
    """Compute the representative norm for rescaling.

    Section 3.4: "rescale by a representative norm (e.g., the mean norm
    of sources for that block)"

    Args:
        norms: list of scalar norms, one per source model
        method: "mean" (paper default), "median", or "max"
    """
    stacked = mx.stack(norms)
    if method == "mean":
        return mx.mean(stacked)
    elif method == "median":
        return mx.sort(stacked)[len(norms) // 2]
    elif method == "max":
        return mx.max(stacked)
    else:
        raise ValueError(f"Unknown norm_rescale method: {method}")


def merge_parameters(
    params_list: list[dict[str, mx.array]],
    weights: Optional[list[float]] = None,
    config: Optional[dict] = None,
) -> dict[str, mx.array]:
    """Merge N models' parameters using Fisher-Rao Karcher mean.

    This is the main entry point for the merging algorithm.

    Section 3.4 pipeline:
      For each parameter block:
        1. Normalize to unit sphere
        2. Compute Karcher mean on S^(d-1)
        3. Rescale by mean source norm

    Args:
        params_list: list of N parameter dicts {name: mx.array}
        weights: mixture weights alpha^(i), must sum to 1.
                 Default: equal weights 1/N (Section 3.1).
        config: merge configuration dict (or loads base.yaml)

    Returns:
        Merged parameter dict with same keys and shapes as inputs.
    """
    if config is None:
        config = load_config()

    n = len(params_list)
    assert n >= 2, "Need at least 2 models to merge"

    # Default: equal weights (Section 3.1)
    if weights is None:
        w = 1.0 / n
        weights = [w] * n
    assert len(weights) == n
    assert abs(sum(weights) - 1.0) < 1e-6

    step_size = float(config.get("step_size", 1.0))
    max_iter = int(config.get("max_iterations", 100))
    tol = float(config.get("convergence_tol", 1e-7))
    norm_method = config.get("norm_rescale", "mean")
    min_norm = float(config.get("min_norm", 1e-8))

    keys = sorted(params_list[0].keys())
    merged = {}

    for key in keys:
        original_shape = params_list[0][key].shape

        # Flatten each model's block to a vector
        flat_list = [p[key].reshape(-1) for p in params_list]

        # Step 1-2: Compute norms and normalize to unit sphere (Section 3.4)
        unit_vectors = []
        norms = []
        for v in flat_list:
            uv, norm = normalize(v)
            unit_vectors.append(uv)
            norms.append(norm)

        # Compute representative norm for rescaling (Section 3.4)
        rep_norm = representative_norm(norms, method=norm_method)
        mx.eval(rep_norm)

        # Skip near-zero blocks (e.g., unused biases)
        if rep_norm.item() < min_norm:
            merged[key] = params_list[0][key]
            continue

        # Step 3: Compute Karcher mean on S^(d-1) (Section 3.3)
        if n == 2:
            # Special case: SLERP (Section 3.3: "reduces to SLERP")
            # For equal weights, t=0.5 gives the geodesic midpoint
            # For unequal weights, t = weights[1]
            t = weights[1]
            direction = slerp(unit_vectors[0], unit_vectors[1], t=t)
            mx.eval(direction)
        else:
            # General case: Karcher mean for N > 2 (Section 3.3 Eq. 5)
            direction = karcher_mean_spherical(
                points=unit_vectors,
                weights=weights,
                max_iter=max_iter,
                step_size=step_size,
                tol=tol,
            )

        # Step 4: Rescale by representative norm (Section 3.4)
        merged_flat = rep_norm * direction

        # Reshape back
        merged[key] = merged_flat.reshape(original_shape)
        mx.eval(merged[key])

    return merged


def merge_state_dicts(
    state_dicts: list[dict],
    weights: Optional[list[float]] = None,
    config_path: Optional[str] = None,
) -> dict:
    """High-level API: merge multiple model state dicts.

    Handles nested state dicts by flattening, merging, and unflattening.

    Args:
        state_dicts: list of N state dicts (possibly nested)
        weights: mixture weights (default: equal)
        config_path: path to YAML config (default: base.yaml)

    Returns:
        Merged state dict with same structure as inputs.
    """
    config = load_config(config_path)

    # Flatten nested dicts
    flat_dicts = [_flatten_dict(sd) for sd in state_dicts]

    # Merge
    merged_flat = merge_parameters(flat_dicts, weights=weights, config=config)

    # Unflatten back to original structure
    return _unflatten_dict(merged_flat, template=state_dicts[0])


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten a nested dict into {dotted.key: array}."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full_key))
        elif isinstance(v, mx.array):
            out[full_key] = v
        # Skip non-array values (e.g., metadata)
    return out


def _unflatten_dict(flat: dict[str, mx.array], template: dict) -> dict:
    """Unflatten a flat dict back into nested structure matching template."""
    result = {}
    for k, v in template.items():
        if isinstance(v, dict):
            # Collect all sub-keys
            sub_flat = {}
            prefix = k + "."
            for fk, fv in flat.items():
                if fk.startswith(prefix):
                    sub_flat[fk[len(prefix):]] = fv
            result[k] = _unflatten_dict(sub_flat, v)
        elif isinstance(v, mx.array):
            if k in flat:
                result[k] = flat[k]
            else:
                result[k] = v
        else:
            result[k] = v
    return result

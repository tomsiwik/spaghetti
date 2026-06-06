"""
MoErging: merge dense experts into MoE structure (Section 2.2).

Average all non-MLP parameters across experts, keep MLP blocks separate
as individual MoE experts.

Paper: arxiv.org/abs/2603.29765
"""

from __future__ import annotations

import mlx.core as mx


def _is_mlp_key(key: str) -> bool:
    """Check if a parameter key belongs to an MLP block.

    Handles common naming conventions:
      - layers.N.mlp.*
      - layers.N.feed_forward.*
      - transformer.h.N.mlp.*
      - blocks.N.ffn.*
    """
    parts = key.lower().split(".")
    return any(p in ("mlp", "feed_forward", "ffn", "ff") for p in parts)


def moerge(
    expert_params: list[dict[str, mx.array]],
    mlp_key_fn=None,
) -> tuple[dict[str, mx.array], dict[str, list[mx.array]]]:
    """MoErging: average non-MLP params, keep MLP blocks as separate experts.

    Section 2.2: "The MoErging approach averages the parameters of all layers
    of the dense experts except the MLP layers."

    Args:
        expert_params: list of D parameter dicts, one per dense expert.
            Each dict maps parameter name -> mx.array.
        mlp_key_fn: optional callable(key: str) -> bool to identify MLP params.
            Defaults to _is_mlp_key.

    Returns:
        shared_params: dict of averaged non-MLP parameters.
        expert_mlps: dict mapping MLP param key -> list of D mx.arrays,
            one per expert.  These become the MoE expert blocks.
    """
    if mlp_key_fn is None:
        mlp_key_fn = _is_mlp_key

    num_experts = len(expert_params)
    assert num_experts >= 2, "Need at least 2 experts to merge"

    # Collect all keys (assume all experts share the same architecture)
    all_keys = set(expert_params[0].keys())

    shared_params: dict[str, mx.array] = {}
    expert_mlps: dict[str, list[mx.array]] = {}

    for key in sorted(all_keys):
        if mlp_key_fn(key):
            # Keep MLP params separate — one per expert
            expert_mlps[key] = [ep[key] for ep in expert_params]
        else:
            # Average non-MLP params across all experts
            stacked = mx.stack([ep[key] for ep in expert_params])
            shared_params[key] = mx.mean(stacked, axis=0)

    mx.eval(shared_params)
    return shared_params, expert_mlps


def split_expert_mlps_by_layer(
    expert_mlps: dict[str, list[mx.array]],
) -> dict[int, dict[str, list[mx.array]]]:
    """Group MLP expert parameters by transformer layer index.

    Assumes keys contain a layer index like 'layers.N.mlp.gate.weight'.

    Returns:
        Dict mapping layer_index -> {param_suffix: [expert_0_val, ..., expert_D_val]}
    """
    import re

    by_layer: dict[int, dict[str, list[mx.array]]] = {}
    pattern = re.compile(r"layers\.(\d+)\.")

    for key, expert_list in expert_mlps.items():
        m = pattern.search(key)
        if m:
            layer_idx = int(m.group(1))
            # Get the suffix after 'layers.N.'
            suffix = key[m.end():]
            if layer_idx not in by_layer:
                by_layer[layer_idx] = {}
            by_layer[layer_idx][suffix] = expert_list

    return by_layer

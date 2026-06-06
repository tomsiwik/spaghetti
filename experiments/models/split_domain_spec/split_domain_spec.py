"""Split Domain Specialization: do split children specialize faster on domain data?

exp_split_leaf_actual validated the split mechanism on mixed data. This experiment
tests the ORIGINAL motivation: split a generalist leaf, fine-tune each child on a
DIFFERENT domain, and measure whether inherited features accelerate domain-specific
convergence compared to randomly-initialized children.

Two kill criteria:
KC1 (Convergence Speed): Split children must reach independent-child quality
    at least 10% faster (fewer training steps to reach 99% of final quality).
KC2 (Domain Separation): Split children must show Jaccard overlap < 0.95
    between domains' active capsule sets (i.e., the children actually specialize).

Architecture is identical to HierarchicalTreeGPT via SplitLeafActualGPT.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..hierarchical_tree.hierarchical_tree import HierarchicalTreeGPT
from ..split_leaf_actual.split_leaf_actual import (
    split_leaf_into_tree,
    save_leaf_weights,
    _get_parent_gate,
)
from ..capsule_moe.capsule_moe import CapsuleGroup


@register("split_domain_spec", parent="split_leaf_actual")
class SplitDomainSpecGPT(HierarchicalTreeGPT):
    """HierarchicalTreeGPT for split domain specialization test.

    Architecturally identical. Separate registry entry for experiment tracking.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def profile_leaf_activations(model, dataset, leaf_indices, n_batches=20,
                              batch_size=32, seed=0):
    """Profile which capsules fire in specific tree leaves.

    For each specified leaf, across all layers, record the set of capsule
    indices that fire (activation > 0) for at least one token in the dataset.

    Args:
        model: HierarchicalTreeGPT or subclass
        dataset: CharDataset
        leaf_indices: list of leaf indices to profile
        n_batches: number of evaluation batches
        batch_size: batch size
        seed: RNG seed

    Returns:
        dict: {leaf_idx: {layer_idx: set of active capsule indices}}
    """
    import random
    rng = random.Random(seed)

    active_sets = {li: {l: set() for l in range(len(model.layers))}
                   for li in leaf_indices}

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)
            x_norm2 = layer.norm2(x)

            for li in leaf_indices:
                leaf = layer.tree.leaves[li]
                # Compute capsule activations: ReLU(A @ x)
                h = mx.matmul(x_norm2, leaf.A.weight.T)  # (B, T, n_caps)
                activations = nn.relu(h)
                mx.eval(activations)

                # Find capsules that fired anywhere in this batch
                # Sum over batch and token dims
                cap_sums = mx.sum(activations, axis=(0, 1))  # (n_caps,)
                mx.eval(cap_sums)
                for c_idx, val in enumerate(cap_sums.tolist()):
                    if val > 0:
                        active_sets[li][l_idx].add(c_idx)

            # Continue forward pass
            x = x + layer.tree(x_norm2)

    return active_sets


def compute_domain_jaccard(active_A, active_B, n_layers):
    """Compute Jaccard similarity of active capsule sets between two domains.

    Args:
        active_A: {layer_idx: set of active capsule indices} for domain A leaf
        active_B: {layer_idx: set of active capsule indices} for domain B leaf
        n_layers: number of layers

    Returns:
        per_layer: list of (layer, jaccard, n_active_A, n_active_B, n_intersection)
        mean_jaccard: mean Jaccard across layers
    """
    per_layer = []
    for l in range(n_layers):
        set_a = active_A.get(l, set())
        set_b = active_B.get(l, set())
        if not set_a and not set_b:
            j = 1.0
        else:
            intersection = set_a & set_b
            union = set_a | set_b
            j = len(intersection) / len(union) if union else 1.0
        per_layer.append((l, j, len(set_a), len(set_b),
                          len(set_a & set_b) if (set_a or set_b) else 0))

    mean_j = sum(j for _, j, _, _, _ in per_layer) / n_layers
    return per_layer, mean_j

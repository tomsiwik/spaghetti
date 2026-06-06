"""Split-and-Freeze Contribution Protocol for Tree-Structured Experts.

This experiment tests two mechanisms on hierarchical capsule trees:

1. SPLITTING: When a leaf expert handles multiple sub-domains (detected by
   high activation entropy over the routing distribution), split it into two
   child branches. Each child inherits half the parent's capsules plus noise.
   Compare quality of split branches vs training a new flat expert from scratch.

2. FREEZING: When a branch is mature (stable dead-capsule identity over
   training windows, measured by Jaccard > threshold), freeze its weights.
   New domains graft new branches alongside frozen ones. Frozen branches
   must not degrade when new branches are added.

The model architecture is identical to HierarchicalTreeGPT. This module
provides the split/freeze protocol utilities and the experiment runner.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..hierarchical_tree.hierarchical_tree import (
    HierarchicalTreeGPT,
    HierarchicalCapsuleTree,
    TreeGate,
    HierarchicalBlock,
)
from ..capsule_moe.capsule_moe import CapsuleGroup


@register("split_freeze_protocol", parent="hierarchical_tree")
class SplitFreezeTreeGPT(HierarchicalTreeGPT):
    """HierarchicalTreeGPT with split-and-freeze lifecycle protocol.

    Architecturally identical to HierarchicalTreeGPT. This model exists as a
    separate registry entry to track the split-freeze experiment results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ── Split utilities ─────────────────────────────────────────────────────────

def compute_leaf_entropy(model, dataset, n_batches=20, batch_size=32, seed=0):
    """Compute per-leaf activation entropy across a dataset.

    For each leaf, measures H = -sum p_i log p_i of the routing probability
    distribution over that leaf across different inputs. High entropy means
    the leaf handles diverse inputs (candidate for splitting).

    Returns:
        per_layer: list of dicts per layer, each with:
            leaf_mean_probs: (n_leaves,) mean routing probability per leaf
            leaf_entropy: scalar, entropy of the mean probability distribution
            per_token_entropy: scalar, mean per-token entropy of leaf distribution
    """
    import random
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Accumulate leaf probabilities per layer
    accum = [mx.zeros(model.layers[0].tree.n_leaves) for _ in range(n_layers)]
    total_tokens = 0

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        total_tokens += B * T

        # Forward pass to get leaf probs
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)

            x_norm = layer.norm2(x)
            tree = layer.tree
            leaf_probs = tree._tree_beam_routing(x_norm)  # (B, T, n_leaves)

            # Accumulate mean probs
            accum[l_idx] = accum[l_idx] + mx.sum(leaf_probs, axis=(0, 1))
            mx.eval(accum[l_idx])

            # Complete the forward pass
            x = x + tree(x_norm)

    results = []
    for l_idx in range(n_layers):
        mean_probs = accum[l_idx] / total_tokens  # (n_leaves,)
        mx.eval(mean_probs)

        # Entropy of the mean distribution
        eps = 1e-8
        mp = mean_probs.tolist()
        entropy = -sum(p * (mx.log(mx.array(p + eps)).item()) for p in mp)

        results.append({
            "leaf_mean_probs": mean_probs,
            "leaf_entropy": entropy,
        })

    return results


def split_leaf(tree, leaf_idx, n_embd, noise_scale=0.01):
    """Split a leaf CapsuleGroup into two children.

    The parent leaf's capsules are divided in half. Each child gets half
    the capsules plus small noise for symmetry breaking. A new gate is
    created at the parent's position to route between children.

    This is a conceptual split -- at micro scale with fixed tree depth,
    we implement it by replacing one leaf with a pair (using two existing
    leaves) and setting up the parent gate to route between them.

    Args:
        tree: HierarchicalCapsuleTree
        leaf_idx: index of the leaf to split
        n_embd: embedding dimension
        noise_scale: scale of noise added for symmetry breaking

    Returns:
        dict with split statistics
    """
    parent_leaf = tree.leaves[leaf_idx]
    A_parent = parent_leaf.A.weight  # (n_capsules, d)
    B_parent = parent_leaf.B.weight  # (d, n_capsules)
    mx.eval(A_parent, B_parent)

    n_caps = A_parent.shape[0]
    half = n_caps // 2

    # Child 0 gets first half, child 1 gets second half
    A_child0 = A_parent[:half] + mx.random.normal(A_parent[:half].shape) * noise_scale
    B_child0 = B_parent[:, :half] + mx.random.normal(B_parent[:, :half].shape) * noise_scale
    A_child1 = A_parent[half:] + mx.random.normal(A_parent[half:].shape) * noise_scale
    B_child1 = B_parent[:, half:] + mx.random.normal(B_parent[:, half:].shape) * noise_scale
    mx.eval(A_child0, B_child0, A_child1, B_child1)

    return {
        "parent_n_caps": n_caps,
        "child0_n_caps": half,
        "child1_n_caps": n_caps - half,
        "A_child0": A_child0,
        "B_child0": B_child0,
        "A_child1": A_child1,
        "B_child1": B_child1,
    }


# ── Freeze utilities ────────────────────────────────────────────────────────

def compute_activation_identity(model, dataset, n_batches=20, batch_size=32,
                                 seed=0, threshold=0.0):
    """Profile per-leaf capsule activation patterns (dead/alive identity).

    For each leaf in each layer, returns a boolean mask of which capsules
    are alive (fire on any input in the profiling set).

    Args:
        model: HierarchicalTreeGPT (or SplitFreezeTreeGPT)
        dataset: CharDataset for profiling
        n_batches: number of profiling batches
        batch_size: batch size
        seed: RNG seed
        threshold: activation frequency threshold for "alive"

    Returns:
        list of lists: identity[layer][leaf] = set of alive capsule indices
    """
    import random
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # For each layer, for each leaf, accumulate fire counts
    fire_counts = []
    for layer in model.layers:
        tree = layer.tree
        layer_counts = []
        for leaf in tree.leaves:
            n_caps = leaf.A.weight.shape[0]
            layer_counts.append(mx.zeros(n_caps))
        fire_counts.append(layer_counts)

    total_positions = 0

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        total_positions += B * T

        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)

            x_norm = layer.norm2(x)
            tree = layer.tree

            # Profile each leaf's capsule activations
            for li, leaf in enumerate(tree.leaves):
                h = nn.relu(leaf.A(x_norm))  # (B, T, n_caps)
                fired = (h > 0).astype(mx.float32)
                fire_counts[l_idx][li] = fire_counts[l_idx][li] + mx.sum(fired, axis=(0, 1))
                mx.eval(fire_counts[l_idx][li])

            # Complete forward pass
            x = x + tree(x_norm)

    # Convert to identity sets
    identity = []
    for l_idx in range(n_layers):
        layer_identity = []
        for li in range(len(fire_counts[l_idx])):
            freq = fire_counts[l_idx][li] / total_positions
            mx.eval(freq)
            alive_set = set()
            for ci in range(freq.shape[0]):
                if freq[ci].item() > threshold:
                    alive_set.add(ci)
            layer_identity.append(alive_set)
        identity.append(layer_identity)

    return identity


def compute_identity_jaccard(identity_a, identity_b):
    """Compute Jaccard similarity between two identity snapshots.

    Args:
        identity_a, identity_b: lists from compute_activation_identity

    Returns:
        per_layer_per_leaf: list of lists of Jaccard values
        mean_jaccard: overall mean Jaccard
    """
    per_layer = []
    all_jaccards = []
    for l_idx in range(len(identity_a)):
        layer_jacs = []
        for li in range(len(identity_a[l_idx])):
            a = identity_a[l_idx][li]
            b = identity_b[l_idx][li]
            if not a and not b:
                j = 1.0
            elif not a or not b:
                j = 0.0
            else:
                j = len(a & b) / len(a | b)
            layer_jacs.append(j)
            all_jaccards.append(j)
        per_layer.append(layer_jacs)

    mean_j = sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0.0
    return per_layer, mean_j


def freeze_leaves(model, leaf_indices):
    """Freeze specific leaves across all layers.

    Args:
        model: HierarchicalTreeGPT
        leaf_indices: set/list of leaf indices to freeze
    """
    for layer in model.layers:
        for li in leaf_indices:
            if li < len(layer.tree.leaves):
                layer.tree.leaves[li].freeze()


def freeze_gates(model, gate_indices):
    """Freeze specific gates across all layers.

    Args:
        model: HierarchicalTreeGPT
        gate_indices: set/list of gate indices to freeze
    """
    for layer in model.layers:
        for gi in gate_indices:
            if gi < len(layer.tree.gates):
                layer.tree.gates[gi].freeze()

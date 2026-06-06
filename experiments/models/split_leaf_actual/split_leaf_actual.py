"""Actual split_leaf() mechanism test.

This experiment tests what exp_split_freeze_protocol left untested: the actual
split_leaf() operation that divides one trained parent leaf's capsules into two
children. The prior experiment validated warm-start (reusing existing sibling
leaf weights) but never invoked split_leaf().

Three kill criteria:
1. Function preservation: ||f_c0 + f_c1 - f_parent|| > 5% of ||f_parent||
2. Split quality: split children >5% worse than independently-trained leaf pair
3. (Directional) Convergence speed: split children should not be slower to converge

Architecture is identical to HierarchicalTreeGPT. This module provides the
experiment runner and analysis utilities.
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


@register("split_leaf_actual", parent="split_freeze_protocol")
class SplitLeafActualGPT(HierarchicalTreeGPT):
    """HierarchicalTreeGPT for the split_leaf mechanism test.

    Architecturally identical to HierarchicalTreeGPT. Exists as a separate
    registry entry to track split_leaf experiment results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# -- Split operation (improved from split_freeze_protocol) --------------------

def split_leaf_into_tree(model, leaf_idx, noise_scale=0.01):
    """Split a single leaf into two children across all layers.

    For each layer:
    1. Take the target leaf's CapsuleGroup (A: n_caps x d, B: d x n_caps)
    2. Create child0 from first half of capsules + noise
    3. Create child1 from second half of capsules + noise
    4. Replace two adjacent leaf positions with the children
    5. Initialize the parent gate to route 50/50

    Since we cannot dynamically grow the tree at micro scale, we use
    a depth-2 tree (4 leaves) and designate leaf 0 as the "parent" that
    will be split. After splitting, leaf 0 gets child0's weights and
    leaf 1 gets child1's weights. Gate 1 (parent of leaves 0,1) routes
    between them.

    Args:
        model: HierarchicalTreeGPT (or subclass)
        leaf_idx: the leaf to split (must be 0 or 2 for sibling pairs)
        noise_scale: scale of symmetry-breaking noise

    Returns:
        dict with:
            parent_norms: per-layer L2 norm of parent output
            reconstruction_errors: per-layer ||f_c0+f_c1 - f_parent|| / ||f_parent||
    """
    sibling_idx = leaf_idx + 1  # the sibling that will receive child1

    stats = {
        "parent_n_caps": [],
        "child0_n_caps": [],
        "child1_n_caps": [],
    }

    for layer in model.layers:
        tree = layer.tree
        parent_leaf = tree.leaves[leaf_idx]

        A_parent = parent_leaf.A.weight  # (n_caps, d)
        B_parent = parent_leaf.B.weight  # (d, n_caps)
        mx.eval(A_parent, B_parent)

        n_caps = A_parent.shape[0]
        d = A_parent.shape[1]
        half = n_caps // 2

        # Split capsules
        A_child0 = A_parent[:half] + mx.random.normal(A_parent[:half].shape) * noise_scale
        B_child0 = B_parent[:, :half] + mx.random.normal(B_parent[:, :half].shape) * noise_scale
        A_child1 = A_parent[half:] + mx.random.normal(A_parent[half:].shape) * noise_scale
        B_child1 = B_parent[:, half:] + mx.random.normal(B_parent[:, half:].shape) * noise_scale
        mx.eval(A_child0, B_child0, A_child1, B_child1)

        # Create new CapsuleGroup modules with half the capsules
        child0_group = CapsuleGroup(d, half)
        child1_group = CapsuleGroup(d, n_caps - half)

        # Set their weights from the split
        child0_group.A = nn.Linear(d, half, bias=False)
        child0_group.B = nn.Linear(half, d, bias=False)
        child0_group.A.load_weights([("weight", A_child0)])
        child0_group.B.load_weights([("weight", B_child0)])

        child1_group.A = nn.Linear(d, n_caps - half, bias=False)
        child1_group.B = nn.Linear(n_caps - half, d, bias=False)
        child1_group.A.load_weights([("weight", A_child1)])
        child1_group.B.load_weights([("weight", B_child1)])

        # Replace leaves in the tree
        tree.leaves[leaf_idx] = child0_group
        tree.leaves[sibling_idx] = child1_group

        # Initialize parent gate to route 50/50 (sigmoid(0) = 0.5)
        parent_gate_idx = _get_parent_gate(leaf_idx, tree.depth)
        gate = tree.gates[parent_gate_idx]
        gate.proj.load_weights([
            ("weight", mx.zeros_like(gate.proj.weight)),
            ("bias", mx.zeros_like(gate.proj.bias)),
        ])

        mx.eval(tree.parameters())

        stats["parent_n_caps"].append(n_caps)
        stats["child0_n_caps"].append(half)
        stats["child1_n_caps"].append(n_caps - half)

    return stats


def _get_parent_gate(leaf_idx, depth):
    """Get the internal gate index that is the parent of a leaf pair.

    In a depth-D tree with 0-indexed internal nodes:
    - Leaves 0,1 share parent gate at internal index depth-related position

    For depth=3: gates indexed 0-6, leaves 0-7
    Gate-to-leaf mapping: gate i's children are leaves at indices derived
    from the tree structure.

    For the standard binary tree:
    - Gate 0 (root): left child is gate 1, right child is gate 2
    - Gate 1: left child is gate 3, right child is gate 4
    - Gate 2: left child is gate 5, right child is gate 6
    - Gate 3: leaves 0,1
    - Gate 4: leaves 2,3
    - Gate 5: leaves 4,5
    - Gate 6: leaves 6,7

    So parent_gate(leaf_0) = 3, parent_gate(leaf_2) = 4, etc.
    General: parent_gate(leaf_idx) = (n_internal - 1) // 2 + leaf_idx // 2
    Actually: the last level of gates starts at index n_internal - n_leaves/2
    """
    n_leaves = 2 ** depth
    n_internal = n_leaves - 1
    # Last row of internal nodes starts at: n_internal - n_leaves//2
    last_gate_row_start = n_internal - n_leaves // 2
    return last_gate_row_start + leaf_idx // 2


def measure_function_preservation(model, dataset, leaf_idx, parent_weights,
                                  n_batches=20, batch_size=32, seed=0):
    """Measure how well split children preserve the parent's function.

    Computes ||f_child0(x) + f_child1(x) - f_parent(x)|| / ||f_parent(x)||
    averaged over a dataset, for each layer.

    Args:
        model: the model AFTER splitting (children in place)
        dataset: CharDataset for evaluation
        leaf_idx: original parent leaf index
        parent_weights: dict of {layer_idx: (A_parent, B_parent)} saved before split
        n_batches: number of evaluation batches
        batch_size: batch size
        seed: RNG seed

    Returns:
        per_layer_error: list of relative errors per layer
        mean_error: mean relative error across layers
    """
    import random
    rng = random.Random(seed)

    sibling_idx = leaf_idx + 1
    n_layers = len(model.layers)

    # Accumulate errors per layer
    parent_norms = [0.0] * n_layers
    error_norms = [0.0] * n_layers
    total_tokens = 0

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        total_tokens += B * T

        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)
            x_norm2 = layer.norm2(x)

            # Get parent's output using saved weights
            A_p, B_p = parent_weights[l_idx]
            h_parent = nn.relu(mx.matmul(x_norm2, A_p.T))  # (B, T, n_caps)
            f_parent = mx.matmul(h_parent, B_p.T)  # (B, T, d)
            # Note: B_p is (d, n_caps), so f_parent = h_parent @ B_p^T = (B,T,n_caps) @ (n_caps,d) = (B,T,d)
            # Actually CapsuleGroup stores B as Linear(n_caps, d), so B.weight is (d, n_caps)
            # B(h) = h @ B.weight.T... no. nn.Linear(in, out).weight is (out, in)
            # So B = Linear(n_caps, d) => B.weight is (d, n_caps)
            # B(h) computes h @ weight.T = (B,T,n_caps) @ (n_caps, d) = (B,T,d). Correct.
            # But we have B_p = B.weight = (d, n_caps).
            # f_parent = h_parent @ B_p.T  -- this gives (B,T,n_caps)@(n_caps,d) = (B,T,d). Correct.
            # Wait: B_p.T is (n_caps, d). h_parent is (B,T,n_caps).
            # h_parent @ B_p.T = (B,T,n_caps) @ (n_caps, d) = (B,T,d). Yes, correct.

            # Get children's outputs using current model weights
            child0 = layer.tree.leaves[leaf_idx]
            child1 = layer.tree.leaves[sibling_idx]
            f_child0 = child0(x_norm2)  # (B, T, d)
            f_child1 = child1(x_norm2)  # (B, T, d)
            f_combined = f_child0 + f_child1

            # Compute norms
            p_norm = mx.sqrt(mx.sum(f_parent * f_parent)).item()
            e_norm = mx.sqrt(mx.sum((f_combined - f_parent) * (f_combined - f_parent))).item()

            parent_norms[l_idx] += p_norm
            error_norms[l_idx] += e_norm

            # Continue forward pass through the tree
            x = x + layer.tree(x_norm2)

    per_layer_error = []
    for l_idx in range(n_layers):
        if parent_norms[l_idx] > 1e-8:
            rel_err = error_norms[l_idx] / parent_norms[l_idx]
        else:
            rel_err = 0.0
        per_layer_error.append(rel_err)

    mean_error = sum(per_layer_error) / len(per_layer_error)
    return per_layer_error, mean_error


def save_leaf_weights(model, leaf_idx):
    """Save a leaf's weights from all layers for function preservation measurement.

    Returns:
        dict: {layer_idx: (A_weight, B_weight)}
    """
    saved = {}
    for l_idx, layer in enumerate(model.layers):
        leaf = layer.tree.leaves[leaf_idx]
        A = mx.array(leaf.A.weight)
        B = mx.array(leaf.B.weight)
        mx.eval(A, B)
        saved[l_idx] = (A, B)
    return saved

"""Tests for split-and-freeze protocol mechanisms."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn
from experiments.models import get_model
from experiments.models.split_freeze_protocol.split_freeze_protocol import (
    compute_leaf_entropy,
    split_leaf,
    compute_activation_identity,
    compute_identity_jaccard,
    freeze_leaves,
    freeze_gates,
)


def test_model_registers():
    """SplitFreezeTreeGPT should be registered and instantiable."""
    model = get_model("split_freeze_protocol", vocab_size=28, block_size=32)
    mx.eval(model.parameters())
    x = mx.zeros((1, 8), dtype=mx.int32)
    out = model(x)
    assert out.shape == (1, 8, 28), f"Expected (1, 8, 28), got {out.shape}"
    print("PASS: model registers and runs forward")


def test_split_leaf_shapes():
    """split_leaf should produce children with half the parent's capsules."""
    from experiments.models.hierarchical_tree.hierarchical_tree import HierarchicalCapsuleTree
    mx.random.seed(42)
    tree = HierarchicalCapsuleTree(n_embd=64, depth=3,
                                    n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())

    result = split_leaf(tree, leaf_idx=0, n_embd=64, noise_scale=0.01)
    assert result["parent_n_caps"] == 32
    assert result["child0_n_caps"] == 16
    assert result["child1_n_caps"] == 16
    assert result["A_child0"].shape == (16, 64)
    assert result["B_child0"].shape == (64, 16)
    assert result["A_child1"].shape == (16, 64)
    assert result["B_child1"].shape == (64, 16)
    print("PASS: split_leaf produces correct shapes")


def test_freeze_leaves_works():
    """Frozen leaves should not change during training."""
    mx.random.seed(42)
    model = get_model("split_freeze_protocol", vocab_size=28, block_size=32,
                       tree_depth=2, n_capsules_per_leaf=16, beam_width=2)
    mx.eval(model.parameters())

    # Record leaf 0 weights before freeze
    w_before = mx.array(model.layers[0].tree.leaves[0].A.weight)
    mx.eval(w_before)

    # Freeze leaf 0
    freeze_leaves(model, {0})

    # Verify leaf 0 is not in trainable params
    trainable_keys = {k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    for k in trainable_keys:
        assert "leaves.0.A" not in k or "layers.0" not in k, \
            f"Frozen leaf param found in trainable: {k}"

    print("PASS: freeze_leaves correctly freezes parameters")


def test_freeze_gates_works():
    """Frozen gates should not appear in trainable params."""
    mx.random.seed(42)
    model = get_model("split_freeze_protocol", vocab_size=28, block_size=32,
                       tree_depth=2, n_capsules_per_leaf=16, beam_width=2)
    mx.eval(model.parameters())

    freeze_gates(model, {0, 1})

    trainable_keys = {k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    for k in trainable_keys:
        assert not (("gates.0." in k or "gates.1." in k) and "layers.0" in k), \
            f"Frozen gate param found in trainable: {k}"

    print("PASS: freeze_gates correctly freezes parameters")


def test_identity_jaccard_self():
    """Identity Jaccard of a model with itself should be 1.0."""
    id_a = [[{0, 1, 2}, {3, 4}], [{0}, {1, 2, 3}]]
    id_b = [[{0, 1, 2}, {3, 4}], [{0}, {1, 2, 3}]]
    per_layer, mean_j = compute_identity_jaccard(id_a, id_b)
    assert abs(mean_j - 1.0) < 1e-6, f"Self Jaccard should be 1.0, got {mean_j}"
    print("PASS: identity Jaccard self-consistency")


def test_identity_jaccard_disjoint():
    """Disjoint identity sets should have Jaccard 0."""
    id_a = [[{0, 1, 2}]]
    id_b = [[{3, 4, 5}]]
    per_layer, mean_j = compute_identity_jaccard(id_a, id_b)
    assert abs(mean_j - 0.0) < 1e-6, f"Disjoint Jaccard should be 0.0, got {mean_j}"
    print("PASS: identity Jaccard disjoint sets")


def test_gradient_flow_with_frozen_leaves():
    """Gradients should flow through unfrozen parts when some leaves are frozen."""
    mx.random.seed(42)
    model = get_model("split_freeze_protocol", vocab_size=28, block_size=16,
                       n_embd=64, n_head=4, n_layer=2,
                       tree_depth=2, n_capsules_per_leaf=16, beam_width=2)
    mx.eval(model.parameters())

    # Freeze leaves 0 and 1
    freeze_leaves(model, {0, 1})

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
        ) + model.aux_loss()

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 0]])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens, targets)
    mx.eval(loss, grads)

    # Check unfrozen leaf 2 has non-zero gradients
    leaf2_grad = grads["layers"][0]["tree"]["leaves"][2]["A"]["weight"]
    mx.eval(leaf2_grad)
    assert mx.any(leaf2_grad != 0).item(), "Unfrozen leaf should have non-zero gradients"

    print("PASS: gradient flow with frozen leaves")


if __name__ == "__main__":
    tests = [
        test_model_registers,
        test_split_leaf_shapes,
        test_freeze_leaves_works,
        test_freeze_gates_works,
        test_identity_jaccard_self,
        test_identity_jaccard_disjoint,
        test_gradient_flow_with_frozen_leaves,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()

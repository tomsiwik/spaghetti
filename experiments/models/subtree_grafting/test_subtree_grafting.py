"""Tests for subtree grafting composition mechanism."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn
from experiments.models import get_model
from experiments.models.subtree_grafting.run_experiment import (
    get_subtree_params, set_param_by_path, freeze_except_subtree,
    freeze_except_root_gate, get_tree_params,
)


def test_model_registers():
    """SubtreeGraftingGPT should be registered and instantiable."""
    model = get_model("subtree_grafting", vocab_size=28, block_size=32)
    mx.eval(model.parameters())
    x = mx.zeros((1, 8), dtype=mx.int32)
    out = model(x)
    assert out.shape == (1, 8, 28), f"Expected (1, 8, 28), got {out.shape}"
    print("PASS: model registers and runs forward")


def test_subtree_param_extraction():
    """Left and right subtree params should be disjoint and complete."""
    model = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                       tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    left_params = get_subtree_params(model, "left")
    right_params = get_subtree_params(model, "right")

    # Disjoint: no key appears in both
    overlap = set(left_params.keys()) & set(right_params.keys())
    assert len(overlap) == 0, f"Overlap: {overlap}"

    # Coverage: left + right should cover all non-root gate/leaf params
    # Root gate (gate 0) is excluded from both
    all_tree = get_tree_params(model)
    root_keys = set()
    for layer_idx in range(4):
        for k, v in nn.utils.tree_flatten(model.layers[layer_idx].tree.gates[0].parameters()):
            root_keys.add(f"layers.{layer_idx}.tree.gates.0.{k}")

    non_root = set(all_tree.keys()) - root_keys
    subtree_combined = set(left_params.keys()) | set(right_params.keys())
    assert non_root == subtree_combined, \
        f"Missing: {non_root - subtree_combined}, Extra: {subtree_combined - non_root}"

    print(f"PASS: subtree extraction (left={len(left_params)}, right={len(right_params)}, "
          f"root={len(root_keys)}, total_tree={len(all_tree)})")


def test_grafting_preserves_subtree_params():
    """After grafting, each domain's subtree params should be exactly preserved."""
    mx.random.seed(42)
    model_a = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                         tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model_a.parameters())

    mx.random.seed(99)
    model_b = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                         tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model_b.parameters())

    # Extract subtrees
    left_params = get_subtree_params(model_a, "left")
    right_params = get_subtree_params(model_b, "right")

    # Create grafted model from model_a base
    mx.random.seed(42)
    graft = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                       tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    base_weights = {k: mx.array(v) for k, v in nn.utils.tree_flatten(model_a.parameters())}
    graft.load_weights(list(base_weights.items()))
    mx.eval(graft.parameters())

    # Apply grafts
    for key, val in left_params.items():
        set_param_by_path(graft, key, val)
    for key, val in right_params.items():
        set_param_by_path(graft, key, val)
    mx.eval(graft.parameters())

    # Verify left subtree matches model_a
    graft_left = get_subtree_params(graft, "left")
    for key in left_params:
        diff = mx.abs(graft_left[key] - left_params[key]).sum().item()
        assert diff < 1e-6, f"Left subtree mismatch at {key}: diff={diff}"

    # Verify right subtree matches model_b
    graft_right = get_subtree_params(graft, "right")
    for key in right_params:
        diff = mx.abs(graft_right[key] - right_params[key]).sum().item()
        assert diff < 1e-6, f"Right subtree mismatch at {key}: diff={diff}"

    print("PASS: grafting preserves subtree params exactly")


def test_freeze_except_subtree():
    """Freezing should leave only the assigned subtree trainable."""
    model = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                       tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    freeze_except_subtree(model, side="left")

    # Check trainable params: should only be gates 1,3,4 and leaves 0,1,2,3
    trainable = {k for k, v in nn.utils.tree_flatten(model.trainable_parameters())}
    frozen = {k for k, v in nn.utils.tree_flatten(model.parameters())} - trainable

    # Verify left subtree params are trainable
    for layer_idx in range(4):
        for gi in [1, 3, 4]:
            gate_keys = [f"layers.{layer_idx}.tree.gates.{gi}.proj.weight",
                         f"layers.{layer_idx}.tree.gates.{gi}.proj.bias"]
            for gk in gate_keys:
                assert gk in trainable, f"Expected trainable: {gk}"
        for li in [0, 1, 2, 3]:
            leaf_keys = [f"layers.{layer_idx}.tree.leaves.{li}.A.weight",
                         f"layers.{layer_idx}.tree.leaves.{li}.B.weight"]
            for lk in leaf_keys:
                assert lk in trainable, f"Expected trainable: {lk}"

    # Verify right subtree and root are frozen
    for layer_idx in range(4):
        for gi in [0, 2, 5, 6]:
            gate_keys = [f"layers.{layer_idx}.tree.gates.{gi}.proj.weight",
                         f"layers.{layer_idx}.tree.gates.{gi}.proj.bias"]
            for gk in gate_keys:
                assert gk not in trainable, f"Expected frozen: {gk}"

    print("PASS: freeze_except_subtree works correctly")


def test_freeze_except_root_gate():
    """Freezing for graft calibration should only leave root gate trainable."""
    model = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                       tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    freeze_except_root_gate(model)

    trainable = {k for k, v in nn.utils.tree_flatten(model.trainable_parameters())}

    # Only root gate should be trainable
    for layer_idx in range(4):
        # Root gate should be trainable
        assert f"layers.{layer_idx}.tree.gates.0.proj.weight" in trainable
        assert f"layers.{layer_idx}.tree.gates.0.proj.bias" in trainable
        # All other gates should be frozen
        for gi in range(1, 7):
            assert f"layers.{layer_idx}.tree.gates.{gi}.proj.weight" not in trainable

    n_trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    # Root gate per layer: d + 1 = 65 params. 4 layers = 260 params
    expected = 4 * 65
    assert n_trainable == expected, f"Expected {expected} trainable params, got {n_trainable}"

    print(f"PASS: freeze_except_root_gate ({n_trainable} trainable params)")


if __name__ == "__main__":
    test_model_registers()
    test_subtree_param_extraction()
    test_grafting_preserves_subtree_params()
    test_freeze_except_subtree()
    test_freeze_except_root_gate()
    print("\nAll tests passed!")

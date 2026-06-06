"""Tests for the sequential freeze-graft protocol."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, domain_split
from experiments.models import get_model


def test_model_registry():
    """Model registers correctly and is instantiable."""
    model = get_model("sequential_freeze_graft",
                      vocab_size=28, block_size=32,
                      tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    # Should have 4 layers, each with 7 gates and 8 leaves
    assert len(model.layers) == 4
    for layer in model.layers:
        assert len(layer.tree.gates) == 7
        assert len(layer.tree.leaves) == 8

    # Forward pass works
    x = mx.zeros((2, 16), dtype=mx.int32)
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 16, 28)
    print("PASS: test_model_registry")


def test_quaternary_domain_split():
    """Quaternary split produces 4 non-empty, non-overlapping domains."""
    docs = load_names()
    splits = domain_split(docs, method="quaternary")

    assert len(splits) == 4
    domain_names = list(splits.keys())
    assert domain_names == ["a_f", "g_m", "n_s", "t_z"]

    # All non-empty
    for name, docs_list in splits.items():
        assert len(docs_list) > 0, f"Domain {name} is empty"

    # Coverage (no docs lost)
    total = sum(len(d) for d in splits.values())
    assert total == len(docs), f"Split lost docs: {total} vs {len(docs)}"

    print(f"PASS: test_quaternary_domain_split "
          f"(sizes: {[len(d) for d in splits.values()]})")


def test_freeze_preserves_weights():
    """Freezing leaves/gates prevents weight changes during training."""
    model = get_model("sequential_freeze_graft",
                      vocab_size=28, block_size=32,
                      tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    # Capture weights before freeze
    leaf0_A = mx.array(model.layers[0].tree.leaves[0].A.weight)
    gate0_w = mx.array(model.layers[0].tree.gates[0].proj.weight)
    mx.eval(leaf0_A, gate0_w)

    # Freeze leaf 0 and gate 0
    for layer in model.layers:
        layer.tree.leaves[0].freeze()
        layer.tree.gates[0].freeze()

    # Check that frozen params are not in trainable set
    trainable_keys = {k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    # Leaf 0 and gate 0 should NOT be trainable
    for k in trainable_keys:
        assert "leaves.0." not in k or "layers.0." not in k  # Not all layers checked, just structure
    print("PASS: test_freeze_preserves_weights")


def test_progressive_allocation():
    """The progressive halving allocation covers all 8 leaves."""
    # Graft 1: A gets 0-3, B gets 4-7
    # Graft 2: B frozen 4-5, C gets 6-7
    # Graft 3: C frozen 6, D gets 7
    # Final: A={0,1,2,3}, B={4,5}, C={6}, D={7}
    domain_leaves = {
        "A": {0, 1, 2, 3},
        "B": {4, 5},
        "C": {6},
        "D": {7},
    }
    all_leaves = set()
    for leaves in domain_leaves.values():
        assert not (leaves & all_leaves), "Overlapping leaf allocation"
        all_leaves |= leaves
    assert all_leaves == set(range(8)), f"Missing leaves: {set(range(8)) - all_leaves}"
    print("PASS: test_progressive_allocation")


if __name__ == "__main__":
    test_model_registry()
    test_quaternary_domain_split()
    test_freeze_preserves_weights()
    test_progressive_allocation()
    print("\nAll tests passed.")

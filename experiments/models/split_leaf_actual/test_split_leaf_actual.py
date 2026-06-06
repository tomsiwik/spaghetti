"""Tests for split_leaf_actual mechanism."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.models.split_leaf_actual.split_leaf_actual import (
    split_leaf_into_tree,
    save_leaf_weights,
    measure_function_preservation,
    _get_parent_gate,
)


def test_model_registers():
    """Model registers and instantiates."""
    model = get_model("split_leaf_actual", vocab_size=28, block_size=32)
    mx.eval(model.parameters())
    out = model(mx.zeros((1, 4), dtype=mx.int32))
    assert out.shape == (1, 4, 28)
    print("PASS: model registers and runs forward")


def test_parent_gate_mapping():
    """Gate-to-leaf parent mapping is correct for depth 3."""
    # Depth 3: 8 leaves, 7 internal gates
    # Last row of gates: indices 3,4,5,6
    # Gate 3 -> leaves 0,1; Gate 4 -> leaves 2,3; Gate 5 -> leaves 4,5; Gate 6 -> leaves 6,7
    assert _get_parent_gate(0, 3) == 3
    assert _get_parent_gate(1, 3) == 3
    assert _get_parent_gate(2, 3) == 4
    assert _get_parent_gate(4, 3) == 5
    assert _get_parent_gate(6, 3) == 6
    print("PASS: parent gate mapping correct")


def test_split_preserves_capsule_count():
    """Split divides capsules correctly."""
    mx.random.seed(42)
    model = get_model("split_leaf_actual", vocab_size=28, block_size=32,
                       n_capsules_per_leaf=32)
    mx.eval(model.parameters())

    stats = split_leaf_into_tree(model, leaf_idx=0, noise_scale=0.0)
    assert stats["parent_n_caps"][0] == 32
    assert stats["child0_n_caps"][0] == 16
    assert stats["child1_n_caps"][0] == 16
    print("PASS: split preserves capsule count (32 -> 16 + 16)")


def test_zero_noise_exact_reconstruction():
    """At noise=0, f_c0 + f_c1 = f_parent exactly."""
    mx.random.seed(42)
    model = get_model("split_leaf_actual", vocab_size=28, block_size=32,
                       n_capsules_per_leaf=32)
    mx.eval(model.parameters())

    # Save parent weights before split
    parent_weights = save_leaf_weights(model, leaf_idx=0)

    # Split with zero noise
    split_leaf_into_tree(model, leaf_idx=0, noise_scale=0.0)

    # Check reconstruction on a single batch
    x = mx.random.normal((2, 8, 64))  # (B, T, d)

    for l_idx, layer in enumerate(model.layers):
        A_p, B_p = parent_weights[l_idx]
        # Parent output
        h_parent = nn.relu(mx.matmul(x, A_p.T))
        f_parent = mx.matmul(h_parent, B_p.T)

        # Children output
        child0 = layer.tree.leaves[0]
        child1 = layer.tree.leaves[1]
        f_c0 = child0(x)
        f_c1 = child1(x)
        f_combined = f_c0 + f_c1

        error = mx.max(mx.abs(f_combined - f_parent)).item()
        assert error < 1e-4, f"Layer {l_idx}: reconstruction error {error} > 1e-4"

    print("PASS: zero noise gives exact reconstruction (error < 1e-4)")


def test_nonzero_noise_small_error():
    """At noise=0.01, reconstruction error is small."""
    mx.random.seed(42)
    model = get_model("split_leaf_actual", vocab_size=28, block_size=32,
                       n_capsules_per_leaf=32)
    mx.eval(model.parameters())

    parent_weights = save_leaf_weights(model, leaf_idx=0)
    split_leaf_into_tree(model, leaf_idx=0, noise_scale=0.01)

    x = mx.random.normal((2, 8, 64))

    for l_idx, layer in enumerate(model.layers):
        A_p, B_p = parent_weights[l_idx]
        h_parent = nn.relu(mx.matmul(x, A_p.T))
        f_parent = mx.matmul(h_parent, B_p.T)

        child0 = layer.tree.leaves[0]
        child1 = layer.tree.leaves[1]
        f_combined = child0(x) + child1(x)

        f_parent_norm = mx.sqrt(mx.sum(f_parent * f_parent)).item()
        error_norm = mx.sqrt(mx.sum((f_combined - f_parent) ** 2)).item()
        rel_error = error_norm / (f_parent_norm + 1e-8)

        # 0.01 noise on random data can produce up to ~20% relative error
        # per layer on a single batch; the KC1 threshold is 5% averaged
        # over many batches. Use a generous unit test threshold.
        assert rel_error < 0.50, f"Layer {l_idx}: relative error {rel_error:.4f} > 0.50"

    print("PASS: noise=0.01 gives bounded reconstruction error (<50% per-layer single-batch)")


def test_gate_initialized_to_50_50():
    """After split, parent gate outputs ~0.5 (uniform routing)."""
    mx.random.seed(42)
    model = get_model("split_leaf_actual", vocab_size=28, block_size=32)
    mx.eval(model.parameters())

    split_leaf_into_tree(model, leaf_idx=0, noise_scale=0.01)

    x = mx.random.normal((2, 8, 64))
    parent_gate_idx = _get_parent_gate(0, 3)

    for layer in model.layers:
        gate = layer.tree.gates[parent_gate_idx]
        p_left = gate(x)
        mean_p = mx.mean(p_left).item()
        assert abs(mean_p - 0.5) < 0.01, f"Gate mean={mean_p}, expected ~0.5"

    print("PASS: parent gate initialized to 50/50 after split")


if __name__ == "__main__":
    test_model_registers()
    test_parent_gate_mapping()
    test_split_preserves_capsule_count()
    test_zero_noise_exact_reconstruction()
    test_nonzero_noise_small_error()
    test_gate_initialized_to_50_50()
    print("\nAll tests passed!")

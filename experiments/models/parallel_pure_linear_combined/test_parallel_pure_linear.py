"""Tests for combined parallel block + pure-linear attention model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model


def test_parallel_pure_linear_forward():
    """Test forward pass produces correct output shape."""
    model = get_model("parallel_pure_linear_capsule_moe",
                      vocab_size=28, block_size=32, n_embd=64, n_head=4,
                      n_layer=4, n_groups=4, n_capsules_per_group=64,
                      top_k_groups=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_sequential_hybrid_forward():
    """Test control model forward pass."""
    model = get_model("sequential_hybrid_capsule_moe",
                      vocab_size=28, block_size=32, n_embd=64, n_head=4,
                      n_layer=4, n_groups=4, n_capsules_per_group=64,
                      top_k_groups=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_parallel_all_linear():
    """Test that parallel model uses all linear attention by default."""
    model = get_model("parallel_pure_linear_capsule_moe",
                      n_layer=4)
    assert model.layer_types == ["linear"] * 4


def test_sequential_hybrid_default():
    """Test that sequential model uses 3:1 hybrid by default."""
    model = get_model("sequential_hybrid_capsule_moe",
                      n_layer=4)
    assert model.layer_types == ["linear", "linear", "linear", "full"]


def test_parallel_single_norm():
    """Test that parallel blocks have one norm, not two."""
    model = get_model("parallel_pure_linear_capsule_moe", n_layer=2)
    for layer in model.layers:
        assert hasattr(layer, "norm"), "Parallel block should have single norm"
        assert not hasattr(layer, "norm1"), "Parallel block should not have norm1"
        assert not hasattr(layer, "norm2"), "Parallel block should not have norm2"


def test_sequential_two_norms():
    """Test that sequential blocks have two norms."""
    model = get_model("sequential_hybrid_capsule_moe", n_layer=2)
    for layer in model.layers:
        assert hasattr(layer, "norm1"), "Sequential block should have norm1"
        assert hasattr(layer, "norm2"), "Sequential block should have norm2"


def test_aux_loss():
    """Test that aux_loss is callable and returns a scalar."""
    model = get_model("parallel_pure_linear_capsule_moe",
                      vocab_size=28, n_layer=2, n_groups=4)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_composed_groups():
    """Test that composed model has correct number of groups."""
    model = get_model("parallel_pure_linear_capsule_moe",
                      n_layer=2, n_groups=8, top_k_groups=4)
    for layer in model.layers:
        assert len(layer.capsule_pool.groups) == 8


if __name__ == "__main__":
    test_parallel_pure_linear_forward()
    print("PASS: test_parallel_pure_linear_forward")
    test_sequential_hybrid_forward()
    print("PASS: test_sequential_hybrid_forward")
    test_parallel_all_linear()
    print("PASS: test_parallel_all_linear")
    test_sequential_hybrid_default()
    print("PASS: test_sequential_hybrid_default")
    test_parallel_single_norm()
    print("PASS: test_parallel_single_norm")
    test_sequential_two_norms()
    print("PASS: test_sequential_two_norms")
    test_aux_loss()
    print("PASS: test_aux_loss")
    test_composed_groups()
    print("PASS: test_composed_groups")
    print("\nAll tests passed!")

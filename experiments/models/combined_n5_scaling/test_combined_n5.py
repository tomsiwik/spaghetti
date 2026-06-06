"""Tests for combined N=5 scaling model registration and forward pass."""

import mlx.core as mx

from experiments.models import get_model


def test_registration():
    """Test that model is registered correctly."""
    model = get_model("combined_n5_scaling", vocab_size=28)
    mx.eval(model.parameters())
    assert model._registry_name == "combined_n5_scaling"


def test_forward():
    """Test forward pass with default config."""
    model = get_model("combined_n5_scaling",
                      vocab_size=28, block_size=32, n_embd=64, n_head=4,
                      n_layer=4, n_groups=4, n_capsules_per_group=64,
                      top_k_groups=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_composed_n5_groups():
    """Test that a composed N=5 model has correct group count."""
    # N=5 domains, 4 groups each -> 20 composed groups, top_k = 10
    model = get_model("combined_n5_scaling",
                      vocab_size=28, n_layer=2, n_groups=20,
                      n_capsules_per_group=64, top_k_groups=10)
    mx.eval(model.parameters())
    for layer in model.layers:
        assert len(layer.capsule_pool.groups) == 20


def test_all_linear_default():
    """Test that model uses all-linear attention by default (pure-linear)."""
    model = get_model("combined_n5_scaling", n_layer=4)
    assert model.layer_types == ["linear"] * 4


def test_aux_loss():
    """Test aux_loss is callable and returns scalar."""
    model = get_model("combined_n5_scaling",
                      vocab_size=28, n_layer=2, n_groups=4)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3]])
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.shape == ()


if __name__ == "__main__":
    test_registration()
    print("PASS: test_registration")
    test_forward()
    print("PASS: test_forward")
    test_composed_n5_groups()
    print("PASS: test_composed_n5_groups")
    test_all_linear_default()
    print("PASS: test_all_linear_default")
    test_aux_loss()
    print("PASS: test_aux_loss")
    print("\nAll tests passed!")

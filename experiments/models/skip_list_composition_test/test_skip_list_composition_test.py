"""Tests for skip_list_composition_test model registration and basic forward pass."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
from experiments.models import get_model


def test_registration():
    """Model should be registered in the arena."""
    model = get_model("skip_list_composition_test", vocab_size=28, block_size=32)
    assert model is not None
    print("PASS: model registered")


def test_forward():
    """Forward pass should produce correct output shape."""
    mx.random.seed(42)
    model = get_model("skip_list_composition_test", vocab_size=28, block_size=32,
                      n_experts=8, n_capsules_per_expert=32, top_k=2)
    mx.eval(model.parameters())

    tokens = mx.array([[0, 1, 2, 3, 4]])  # (1, 5)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"
    print("PASS: forward pass shape correct")


def test_aux_loss():
    """aux_loss should return a scalar."""
    mx.random.seed(42)
    model = get_model("skip_list_composition_test", vocab_size=28, block_size=32)
    mx.eval(model.parameters())

    tokens = mx.array([[0, 1, 2, 3, 4]])
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    print("PASS: aux_loss returns scalar")


def test_routing_stats():
    """get_routing_stats should return level usage information."""
    mx.random.seed(42)
    model = get_model("skip_list_composition_test", vocab_size=28, block_size=32)
    mx.eval(model.parameters())

    tokens = mx.array([[0, 1, 2, 3, 4]])
    _ = model(tokens)
    mx.eval(model.parameters())

    stats = model.get_routing_stats()
    assert len(stats) > 0, "Expected routing stats for at least one layer"
    for lname, lstats in stats.items():
        assert "level_usage" in lstats
        assert "avg_depth" in lstats
        usage = lstats["level_usage"]
        total = sum(usage)
        assert abs(total - 1.0) < 0.01, f"Level weights should sum to 1, got {total}"
    print("PASS: routing stats valid")


def test_freeze_experts():
    """Freezing experts should leave routers and gates trainable."""
    mx.random.seed(42)
    model = get_model("skip_list_composition_test", vocab_size=28, block_size=32)
    mx.eval(model.parameters())

    # Freeze experts (simulating calibration phase)
    for layer in model.layers:
        for expert in layer.skip_pool.experts:
            expert.freeze()

    # Check that routers and gates are still trainable
    import mlx.nn as nn
    flat_params = nn.utils.tree_flatten(model.trainable_parameters())
    has_router = any("router" in k for k, _ in flat_params)
    flat_params2 = nn.utils.tree_flatten(model.trainable_parameters())
    has_gate = any("confidence_gate" in k for k, _ in flat_params2)
    assert has_router, "Routers should remain trainable after freezing experts"
    assert has_gate, "Confidence gates should remain trainable after freezing experts"
    print("PASS: freeze experts keeps routers/gates trainable")


if __name__ == "__main__":
    test_registration()
    test_forward()
    test_aux_loss()
    test_routing_stats()
    test_freeze_experts()
    print("\nAll tests passed.")

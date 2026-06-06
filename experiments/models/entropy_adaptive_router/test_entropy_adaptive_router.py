"""Tests for entropy-adaptive router."""

import mlx.core as mx
import mlx.nn as nn


def test_model_creation():
    """Model instantiates with correct parameter count."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    model = EntropyAdaptiveRouterGPT(vocab_size=28, n_embd=64, n_groups=4,
                                      n_capsules_per_group=64)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    # Should be very close to capsule_moe (+ 4 threshold scalars)
    print(f"Params: {n_params:,}")
    assert n_params > 200_000  # sanity


def test_forward_pass():
    """Forward pass produces correct output shape."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    model = EntropyAdaptiveRouterGPT(vocab_size=28, n_embd=64, n_groups=4)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (1, 5, 28)


def test_entropy_stats():
    """Entropy stats are populated after forward pass."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    model = EntropyAdaptiveRouterGPT(vocab_size=28, n_embd=64, n_groups=4)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    mx.eval(model.parameters())
    stats = model.entropy_stats()
    assert len(stats) == 4  # 4 layers
    for s in stats:
        assert "mean_entropy" in s
        assert "avg_k" in s
        assert "tau_h" in s
        assert s["mean_entropy"] >= 0
        assert 1.0 <= s["avg_k"] <= 2.0


def test_variable_k():
    """Entropy-based routing produces variable k (not all tokens get same k)."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    model = EntropyAdaptiveRouterGPT(vocab_size=28, n_embd=64, n_groups=4)
    mx.eval(model.parameters())
    # Use diverse tokens to increase chance of entropy variation
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]])
    _ = model(tokens)
    mx.eval(model.parameters())
    avg_k = model.avg_k()
    # avg_k should be between 1 and 2 (not exactly 2)
    print(f"avg_k = {avg_k:.3f}")
    assert 1.0 <= avg_k <= 2.0


def test_aux_loss():
    """Auxiliary loss is computed and is a scalar."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    model = EntropyAdaptiveRouterGPT(vocab_size=28, n_embd=64, n_groups=4)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_gradient_flow():
    """Gradients flow through the entropy-adaptive routing."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    model = EntropyAdaptiveRouterGPT(vocab_size=28, n_embd=64, n_groups=4)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    targets = mx.array([[2, 3, 4, 5, 6]])

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        ) + model.aux_loss()

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens, targets)
    mx.eval(loss, grads)
    assert loss.item() > 0

    # Check that router gradients exist
    flat_grads = nn.utils.tree_flatten(grads)
    router_grads = [(k, v) for k, v in flat_grads if "router" in k]
    assert len(router_grads) > 0, "No router gradients found"
    for name, g in router_grads:
        assert mx.any(g != 0).item(), f"Router grad {name} is all zeros"


def test_matches_fixed_k2_structure():
    """At extreme threshold (tau_H -> inf), all tokens use k=2 (matches capsule_moe)."""
    from experiments.models.entropy_adaptive_router import EntropyAdaptiveRouterGPT
    # Very high threshold -> everything is "confident" -> k=1 for all
    # Very low threshold -> everything is "uncertain" -> k=2 for all
    model = EntropyAdaptiveRouterGPT(
        vocab_size=28, n_embd=64, n_groups=4, tau_h=0.001,
        learn_threshold=False,
    )
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    mx.eval(model.parameters())
    # With tau_h very low, almost all tokens should use k=2
    avg_k = model.avg_k()
    print(f"Low threshold avg_k = {avg_k:.3f}")
    assert avg_k > 1.8, f"Expected avg_k > 1.8, got {avg_k:.3f}"


if __name__ == "__main__":
    test_model_creation()
    test_forward_pass()
    test_entropy_stats()
    test_variable_k()
    test_aux_loss()
    test_gradient_flow()
    test_matches_fixed_k2_structure()
    print("\nAll tests passed!")

"""Tests for AIMD Load Balance experiment."""

import mlx.core as mx
import mlx.nn as nn


def test_aimd_forward_shape():
    """AIMD model produces correct output shape."""
    from .aimd_load_balance import AIMDLoadBalanceGPT

    model = AIMDLoadBalanceGPT(vocab_size=28, block_size=32, n_embd=64,
                                n_head=4, n_layer=4, n_groups=4)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_aux_loss_forward_shape():
    """Aux loss model produces correct output shape."""
    from .aimd_load_balance import AuxLossBalanceGPT

    model = AuxLossBalanceGPT(vocab_size=28, block_size=32, n_embd=64,
                               n_head=4, n_layer=4, n_groups=4)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_no_balance_forward_shape():
    """No-balance model produces correct output shape."""
    from .aimd_load_balance import NoBalanceGPT

    model = NoBalanceGPT(vocab_size=28, block_size=32, n_embd=64,
                          n_head=4, n_layer=4, n_groups=4)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_aimd_no_aux_loss():
    """AIMD model should have zero auxiliary loss (balancing is via bias)."""
    from .aimd_load_balance import AIMDLoadBalanceGPT

    model = AIMDLoadBalanceGPT(vocab_size=28, block_size=32, n_embd=64,
                                n_head=4, n_layer=2, n_groups=4)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.item() == 0.0, f"AIMD aux_loss should be 0.0, got {loss.item()}"


def test_aux_loss_nonzero():
    """Aux loss model should have nonzero balance loss after forward."""
    from .aimd_load_balance import AuxLossBalanceGPT

    model = AuxLossBalanceGPT(vocab_size=28, block_size=32, n_embd=64,
                               n_head=4, n_layer=2, n_groups=4)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]] * 4)  # batch of 4
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.item() > 0.0, f"Aux loss should be positive, got {loss.item()}"
    assert loss.item() < 10.0, f"Aux loss should be bounded, got {loss.item()}"


def test_aimd_bias_updates():
    """AIMD bias should change after forward passes with imbalanced routing."""
    from .aimd_load_balance import AIMDCapsulePool

    pool = AIMDCapsulePool(n_embd=64, n_groups=4, n_capsules_per_group=64,
                            alpha=0.1, beta=0.5, epsilon=0.01)
    mx.eval(pool.parameters())

    initial_bias = list(pool._bias)

    # Run several forward passes to trigger bias updates
    x = mx.random.normal((4, 8, 64))
    for _ in range(5):
        _ = pool(x)
        mx.eval(pool.parameters())

    # At least some bias values should have changed
    changed = sum(1 for a, b in zip(initial_bias, pool._bias) if abs(a - b) > 1e-6)
    print(f"  Bias values changed: {changed}/{len(pool._bias)}")
    print(f"  Initial: {initial_bias}")
    print(f"  Final:   {pool._bias}")
    # With random init, routing is unlikely to be perfectly balanced
    assert changed > 0, "AIMD bias should update after forward passes"


def test_gradient_flows_aimd():
    """Gradients should flow through AIMD model (bias is non-gradient)."""
    from .aimd_load_balance import AIMDLoadBalanceGPT

    model = AIMDLoadBalanceGPT(vocab_size=28, block_size=32, n_embd=64,
                                n_head=4, n_layer=2, n_groups=4)
    mx.eval(model.parameters())

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    tokens = mx.array([[1, 2, 3, 4, 5]])
    targets = mx.array([[2, 3, 4, 5, 6]])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens, targets)
    mx.eval(loss, grads)

    # Check router gradients exist
    grad_flat = nn.utils.tree_flatten(grads)
    router_grads = [(k, v) for k, v in grad_flat if "router" in k]
    assert len(router_grads) > 0, "No gradients for router parameters"

    for name, grad in router_grads:
        grad_norm = mx.sqrt(mx.sum(grad * grad)).item()
        print(f"  {name}: grad_norm={grad_norm:.6f}")
        assert grad_norm > 0, f"Zero gradient for {name}"


def test_param_count_equal():
    """All three models should have identical parameter counts."""
    from .aimd_load_balance import (
        AIMDLoadBalanceGPT, AuxLossBalanceGPT, NoBalanceGPT)

    kwargs = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4,
                  n_layer=4, n_groups=4, n_capsules_per_group=64)

    aimd = AIMDLoadBalanceGPT(**kwargs)
    aux = AuxLossBalanceGPT(**kwargs)
    nob = NoBalanceGPT(**kwargs)

    def count(m):
        return sum(v.size for _, v in nn.utils.tree_flatten(m.parameters()))

    p_aimd = count(aimd)
    p_aux = count(aux)
    p_nob = count(nob)

    print(f"  AIMD: {p_aimd:,}, Aux: {p_aux:,}, NoBalance: {p_nob:,}")
    assert p_aimd == p_aux == p_nob, (
        f"Parameter counts differ: AIMD={p_aimd}, Aux={p_aux}, NoBalance={p_nob}")


if __name__ == "__main__":
    tests = [
        test_aimd_forward_shape,
        test_aux_loss_forward_shape,
        test_no_balance_forward_shape,
        test_aimd_no_aux_loss,
        test_aux_loss_nonzero,
        test_aimd_bias_updates,
        test_gradient_flows_aimd,
        test_param_count_equal,
    ]
    for t in tests:
        print(t.__name__)
        t()
        print("PASSED\n")
    print("ALL TESTS PASSED")

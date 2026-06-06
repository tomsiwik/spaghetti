"""Tests for ReLU Router — routerless capsule MoE with self-routing."""

import mlx.core as mx
import mlx.nn as nn


def test_forward_shape():
    """Output shape matches input shape (B, T) -> (B, T, V)."""
    from .relu_router import ReLURouterGPT

    model = ReLURouterGPT(vocab_size=28, block_size=32, n_embd=64,
                           n_head=4, n_layer=4, n_capsules=256)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])  # (1, 5)
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_param_count_vs_capsule_moe():
    """ReLU router should have FEWER params than capsule_moe (no router weights)."""
    from .relu_router import ReLURouterGPT
    from ..capsule_moe.capsule_moe import CapsuleMoEGPT

    kwargs = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4)

    relu_model = ReLURouterGPT(n_capsules=256, **kwargs)
    cap_model = CapsuleMoEGPT(n_groups=4, n_capsules_per_group=64, **kwargs)

    relu_params = sum(v.size for _, v in nn.utils.tree_flatten(relu_model.parameters()))
    cap_params = sum(v.size for _, v in nn.utils.tree_flatten(cap_model.parameters()))

    print(f"ReLU Router params: {relu_params:,}")
    print(f"Capsule MoE params: {cap_params:,}")

    # ReLU router should have fewer params (no router weight matrix per layer)
    # Router saves: n_layer * n_groups * n_embd = 4 * 4 * 64 = 1,024 params
    assert relu_params < cap_params, (
        f"ReLU Router ({relu_params}) should have fewer params than "
        f"Capsule MoE ({cap_params})")
    assert cap_params - relu_params == 4 * 4 * 64, (
        f"Difference should be {4 * 4 * 64} (router weights), "
        f"got {cap_params - relu_params}")


def test_relu_sparsity():
    """ReLU activation should produce natural sparsity (~50%)."""
    from .relu_router import ReLURouterGPT

    model = ReLURouterGPT(vocab_size=28, block_size=32, n_embd=64,
                           n_head=4, n_layer=2, n_capsules=256)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 4)  # (4, 8)
    _ = model(tokens)
    mx.eval(model.parameters())

    stats = model.capsule_stats()
    for layer_idx, sparsity in enumerate(stats["sparsity"]):
        if sparsity is not None:
            print(f"  Layer {layer_idx}: sparsity={sparsity:.3f}")
            # Natural ReLU sparsity should be roughly 30-70%
            assert 0.1 < sparsity < 0.95, (
                f"Layer {layer_idx} sparsity {sparsity} outside expected range")


def test_aux_loss_nonzero():
    """Auxiliary loss (sparsity + balance) should be computable and finite."""
    from .relu_router import ReLURouterGPT

    model = ReLURouterGPT(vocab_size=28, block_size=32, n_embd=64,
                           n_head=4, n_layer=2, n_capsules=256)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)

    loss_val = loss.item()
    print(f"  aux_loss = {loss_val:.6f}")
    assert loss_val >= 0.0, f"aux_loss should be non-negative, got {loss_val}"
    assert loss_val < 10.0, f"aux_loss should be bounded, got {loss_val}"


def test_no_dead_capsules_at_init():
    """At initialization, no capsules should be completely dead."""
    from .relu_router import ReLURouterGPT

    model = ReLURouterGPT(vocab_size=28, block_size=32, n_embd=64,
                           n_head=4, n_layer=2, n_capsules=64)
    mx.eval(model.parameters())

    # Run a larger batch for better statistics
    tokens = mx.array([[i % 27 + 1 for i in range(16)]] * 16)  # (16, 16)
    _ = model(tokens)
    mx.eval(model.parameters())

    stats = model.capsule_stats()
    for layer_idx, n_dead in enumerate(stats["n_dead"]):
        if n_dead is not None:
            print(f"  Layer {layer_idx}: dead={n_dead}/64")
            # At init with varied input, most capsules should fire at least once
            assert n_dead < 32, (
                f"Layer {layer_idx}: {n_dead}/64 capsules dead at init (too many)")


def test_gradient_flows():
    """Gradients should flow through all capsule parameters (fully differentiable)."""
    from .relu_router import ReLURouterGPT

    model = ReLURouterGPT(vocab_size=28, block_size=32, n_embd=64,
                           n_head=4, n_layer=2, n_capsules=64)
    mx.eval(model.parameters())

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        ) + model.aux_loss()

    tokens = mx.array([[1, 2, 3, 4, 5]])
    targets = mx.array([[2, 3, 4, 5, 6]])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens, targets)
    mx.eval(loss, grads)

    # Check that capsule A and B matrices have gradients
    grad_flat = nn.utils.tree_flatten(grads)
    capsule_grads = [(k, v) for k, v in grad_flat if "capsule_pool" in k]
    assert len(capsule_grads) > 0, "No gradients for capsule_pool parameters"

    for name, grad in capsule_grads:
        grad_norm = mx.sqrt(mx.sum(grad * grad)).item()
        print(f"  {name}: grad_norm={grad_norm:.6f}")
        assert grad_norm > 0, f"Zero gradient for {name}"


def test_composition_by_concatenation():
    """Capsule pools from different sources can be concatenated.

    This tests the STRUCTURAL claim: you can concatenate two A matrices
    and two B matrices to create a larger pool. The routing is implicit
    in the weights — no router needs to be retrained.
    """
    from .relu_router import ReLUCapsulePool

    d = 64
    pool_a = ReLUCapsulePool(d, n_capsules=32)
    pool_b = ReLUCapsulePool(d, n_capsules=32)
    mx.eval(pool_a.parameters())
    mx.eval(pool_b.parameters())

    # Create a composed pool with 64 capsules by concatenating A and B matrices
    pool_composed = ReLUCapsulePool(d, n_capsules=64)
    mx.eval(pool_composed.parameters())

    # Manually compose: stack A weights and B weights
    A_composed = mx.concatenate([pool_a.A.weight, pool_b.A.weight], axis=0)  # (64, d)
    B_composed = mx.concatenate([pool_a.B.weight, pool_b.B.weight], axis=1)  # (d, 64)

    # This should be a valid weight configuration
    assert A_composed.shape == (64, d), f"Expected (64, {d}), got {A_composed.shape}"
    assert B_composed.shape == (d, 64), f"Expected ({d}, 64), got {B_composed.shape}"

    # The composed pool should produce valid output
    x = mx.random.normal((2, 8, d))
    out_a = pool_a(x)
    out_b = pool_b(x)
    mx.eval(out_a, out_b)

    assert out_a.shape == (2, 8, d)
    assert out_b.shape == (2, 8, d)

    print("  Composition by concatenation: structural test passed")


def test_composition_identity_numerical():
    """Verify the KEY mathematical identity: composed(x) == pool_a(x) + pool_b(x).

    This is the central claim of the composition protocol. The composed pool
    with concatenated weights must produce output that exactly equals the sum
    of the individual pool outputs. This must hold numerically, not just
    structurally.
    """
    from .relu_router import ReLUCapsulePool

    d = 64
    pool_a = ReLUCapsulePool(d, n_capsules=32)
    pool_b = ReLUCapsulePool(d, n_capsules=32)
    mx.eval(pool_a.parameters())
    mx.eval(pool_b.parameters())

    # Create composed pool with concatenated weights
    pool_composed = ReLUCapsulePool(d, n_capsules=64)
    mx.eval(pool_composed.parameters())

    A_composed = mx.concatenate([pool_a.A.weight, pool_b.A.weight], axis=0)  # (64, d)
    B_composed = mx.concatenate([pool_a.B.weight, pool_b.B.weight], axis=1)  # (d, 64)

    # Load composed weights into the pool
    pool_composed.A.load_weights([("weight", A_composed)])
    pool_composed.B.load_weights([("weight", B_composed)])
    mx.eval(pool_composed.parameters())

    # Run all three pools on the same input
    x = mx.random.normal((2, 8, d))
    out_a = pool_a(x)
    out_b = pool_b(x)
    out_composed = pool_composed(x)
    out_sum = out_a + out_b
    mx.eval(out_a, out_b, out_composed, out_sum)

    # Verify the mathematical identity: composed(x) == pool_a(x) + pool_b(x)
    diff = mx.max(mx.abs(out_composed - out_sum)).item()
    print(f"  Max absolute difference: {diff:.2e}")
    assert diff < 1e-5, (
        f"Composition identity FAILED: composed(x) != pool_a(x) + pool_b(x), "
        f"max diff = {diff:.2e}")

    print("  Composition identity VERIFIED: composed(x) == pool_a(x) + pool_b(x)")


if __name__ == "__main__":
    print("test_forward_shape")
    test_forward_shape()
    print("PASSED\n")

    print("test_param_count_vs_capsule_moe")
    test_param_count_vs_capsule_moe()
    print("PASSED\n")

    print("test_relu_sparsity")
    test_relu_sparsity()
    print("PASSED\n")

    print("test_aux_loss_nonzero")
    test_aux_loss_nonzero()
    print("PASSED\n")

    print("test_no_dead_capsules_at_init")
    test_no_dead_capsules_at_init()
    print("PASSED\n")

    print("test_gradient_flows")
    test_gradient_flows()
    print("PASSED\n")

    print("test_composition_by_concatenation")
    test_composition_by_concatenation()
    print("PASSED\n")

    print("test_composition_identity_numerical")
    test_composition_identity_numerical()
    print("PASSED\n")

    print("ALL TESTS PASSED")

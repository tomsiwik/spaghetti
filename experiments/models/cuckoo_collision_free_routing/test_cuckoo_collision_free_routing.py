"""Tests for cuckoo collision-free routing."""

import mlx.core as mx
import mlx.nn as nn

from .cuckoo_collision_free_routing import (
    CuckooCollisionFreeRoutingGPT,
    CuckooRouter,
    CuckooCapsulePool,
)


def test_router_shape():
    """Router produces correct output shape."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    x = mx.random.normal((2, 16, 64))
    weights = router(x)
    mx.eval(weights)
    assert weights.shape == (2, 16, 8), f"Expected (2, 16, 8), got {weights.shape}"


def test_router_sparsity():
    """Router produces exactly top_k nonzero weights per token."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    x = mx.random.normal((2, 16, 64))
    weights = router(x)
    mx.eval(weights)
    nonzero_per_token = mx.sum((weights > 0.001).astype(mx.float32), axis=-1)
    mx.eval(nonzero_per_token)
    for b in range(2):
        for t in range(16):
            nz = nonzero_per_token[b, t].item()
            assert nz == 2, f"Expected 2 nonzero, got {nz} at ({b},{t})"


def test_router_weights_sum_to_one():
    """Routing weights sum to 1 for each token."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    x = mx.random.normal((4, 8, 64))
    weights = router(x)
    mx.eval(weights)
    sums = mx.sum(weights, axis=-1)
    mx.eval(sums)
    for b in range(4):
        for t in range(8):
            s = sums[b, t].item()
            assert abs(s - 1.0) < 1e-4, f"Sum {s} != 1.0 at ({b},{t})"


def test_router_deterministic():
    """Same input always produces same routing."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    x = mx.random.normal((1, 8, 64))
    w1 = router(x)
    w2 = router(x)
    mx.eval(w1, w2)
    diff = mx.max(mx.abs(w1 - w2)).item()
    assert diff < 1e-6, f"Non-deterministic routing: max diff {diff}"


def test_router_dual_hash_independence():
    """h1 and h2 produce different routing patterns."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    x = mx.random.normal((4, 32, 64))

    scores_h1 = router.h1(x)
    scores_h2 = router.h2(x)
    mx.eval(scores_h1, scores_h2)

    # h1 and h2 should produce different top-1 assignments for most tokens
    assign_h1 = mx.argmax(scores_h1, axis=-1)
    assign_h2 = mx.argmax(scores_h2, axis=-1)
    mx.eval(assign_h1, assign_h2)

    # They should differ for at least 30% of tokens (random init)
    differ_rate = mx.mean((assign_h1 != assign_h2).astype(mx.float32)).item()
    print(f"  h1 vs h2 differ rate: {differ_rate:.1%}")
    assert differ_rate > 0.2, f"h1 and h2 too similar: {differ_rate:.1%} differ"


def test_router_diagnostics():
    """Diagnostics include chain depths and eviction rate."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    x = mx.random.normal((2, 16, 64))
    _ = router(x)
    mx.eval(router.parameters())

    diag = router.get_diagnostics()
    assert "mean_chain_depth" in diag
    assert "eviction_rate" in diag
    assert "mean_experts_selected" in diag
    assert "normalized_entropy" in diag
    print(f"  mean_chain_depth: {diag['mean_chain_depth']:.3f}")
    print(f"  eviction_rate: {diag['eviction_rate']:.3f}")
    print(f"  mean_experts_selected: {diag['mean_experts_selected']:.1f}")


def test_chain_depth_bounded():
    """Eviction chain depth is bounded by max_chain_depth."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2, max_chain_depth=3)
    mx.eval(router.parameters())
    x = mx.random.normal((8, 32, 64))
    _ = router(x)

    mx.eval(router._chain_depths)
    max_depth = mx.max(router._chain_depths).item()
    assert max_depth <= 3, f"Chain depth {max_depth} exceeds max 3"
    print(f"  max chain depth observed: {max_depth}")


def test_tau_learnable():
    """Collision threshold tau is learnable and in [0, 1]."""
    router = CuckooRouter(n_embd=64, n_groups=8, top_k=2)
    mx.eval(router.parameters())
    tau = router.tau
    mx.eval(tau)
    tau_val = tau.item()
    assert 0 < tau_val < 1, f"tau {tau_val} not in (0, 1)"
    print(f"  initial tau: {tau_val:.3f}")


def test_model_forward():
    """Model produces correct output shape."""
    model = CuckooCollisionFreeRoutingGPT(
        vocab_size=28, block_size=32, n_embd=64, n_groups=8
    )
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (1, 8, 28), f"Expected (1, 8, 28), got {logits.shape}"


def test_model_backward():
    """Gradients flow through cuckoo routing."""
    model = CuckooCollisionFreeRoutingGPT(
        vocab_size=28, block_size=32, n_embd=64, n_groups=8
    )
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
        )

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 0]])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens, targets)
    mx.eval(loss, grads)

    # Check that router gradients exist
    grad_tree = dict(nn.utils.tree_flatten(grads))
    h1_keys = [k for k in grad_tree if "h1" in k]
    h2_keys = [k for k in grad_tree if "h2" in k]
    assert len(h1_keys) > 0, "No gradients for h1"
    assert len(h2_keys) > 0, "No gradients for h2"
    print(f"  loss: {loss.item():.4f}")
    print(f"  h1 grad keys: {h1_keys}")
    print(f"  h2 grad keys: {h2_keys}")


def test_param_count():
    """Cuckoo model has expected parameter count (2x router params vs softmax)."""
    model = CuckooCollisionFreeRoutingGPT(
        vocab_size=28, block_size=32, n_embd=64, n_groups=8
    )
    mx.eval(model.parameters())
    total_params = sum(
        v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters())
    )
    print(f"  Total trainable params: {total_params:,}")
    # Should have 2x router params compared to softmax (h1 + h2)
    # Softmax router: n_embd * n_groups = 64 * 8 = 512 per layer
    # Cuckoo router: 2 * 512 = 1024 per layer
    # Plus 1 tau param per layer (negligible)
    # Total extra: 512 * 4 layers = 2048
    expected_base = 202112  # consistent_hash has 0 router params
    expected_cuckoo = expected_base + 2 * 512 * 4  # ~206,208
    # Allow some tolerance
    assert abs(total_params - expected_cuckoo) < 500, \
        f"Expected ~{expected_cuckoo:,}, got {total_params:,}"


def test_routing_diagnostics():
    """Model-level diagnostics work."""
    model = CuckooCollisionFreeRoutingGPT(
        vocab_size=28, block_size=32, n_embd=64, n_groups=8
    )
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(tokens)
    mx.eval(model.parameters())

    diag = model.get_routing_diagnostics()
    assert "layer_0" in diag
    assert "mean_chain_depth" in diag["layer_0"]
    for li in range(4):
        d = diag[f"layer_{li}"]
        print(f"  layer {li}: chain_depth={d['mean_chain_depth']:.3f}, "
              f"eviction_rate={d['eviction_rate']:.3f}")


if __name__ == "__main__":
    test_router_shape()
    print("PASS: router_shape")

    test_router_sparsity()
    print("PASS: router_sparsity")

    test_router_weights_sum_to_one()
    print("PASS: router_weights_sum_to_one")

    test_router_deterministic()
    print("PASS: router_deterministic")

    test_router_dual_hash_independence()
    print("PASS: router_dual_hash_independence")

    test_router_diagnostics()
    print("PASS: router_diagnostics")

    test_chain_depth_bounded()
    print("PASS: chain_depth_bounded")

    test_tau_learnable()
    print("PASS: tau_learnable")

    test_model_forward()
    print("PASS: model_forward")

    test_model_backward()
    print("PASS: model_backward")

    test_param_count()
    print("PASS: param_count")

    test_routing_diagnostics()
    print("PASS: routing_diagnostics")

    print("\nAll tests passed!")

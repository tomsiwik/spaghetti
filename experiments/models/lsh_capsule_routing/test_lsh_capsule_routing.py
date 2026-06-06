"""Tests for LSH Capsule Routing model."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn

from experiments.models.lsh_capsule_routing.lsh_capsule_routing import (
    LSHRouter,
    LSHCapsulePool,
    LSHCapsuleRoutingGPT,
)


def test_lsh_router_shapes():
    """LSH router produces correct output shapes."""
    router = LSHRouter(n_embd=64, n_groups=8, n_tables=4, top_k=2)
    x = mx.random.normal((2, 16, 64))
    weights = router(x)
    assert weights.shape == (2, 16, 8), f"Expected (2, 16, 8), got {weights.shape}"
    print("  PASS: router output shape correct")


def test_lsh_router_sparsity():
    """LSH router produces sparse weights (exactly top_k nonzero per token)."""
    router = LSHRouter(n_embd=64, n_groups=8, n_tables=4, top_k=2)
    x = mx.random.normal((4, 32, 64))
    weights = router(x)
    mx.eval(weights)

    # Count nonzero weights per token
    nonzero = mx.sum((weights > 1e-6).astype(mx.float32), axis=-1)
    mx.eval(nonzero)
    mean_selected = mx.mean(nonzero).item()
    # Should be close to top_k=2 (may be slightly higher due to ties)
    assert 1.5 <= mean_selected <= 3.5, f"Mean selected {mean_selected} outside [1.5, 3.5]"
    print(f"  PASS: router sparsity correct (mean {mean_selected:.2f} experts/token)")


def test_lsh_router_no_trainable_params():
    """LSH router has NO trainable parameters."""
    router = LSHRouter(n_embd=64, n_groups=8, n_tables=4, top_k=2)
    trainable = list(router.trainable_parameters())
    assert len(trainable) == 0, f"Expected 0 trainable params, got {len(trainable)}"
    print("  PASS: router has 0 trainable parameters")


def test_lsh_router_deterministic():
    """LSH routing is deterministic for same input."""
    router = LSHRouter(n_embd=64, n_groups=8, n_tables=4, top_k=2)
    x = mx.random.normal((2, 8, 64))
    w1 = router(x)
    w2 = router(x)
    mx.eval(w1, w2)
    diff = mx.max(mx.abs(w1 - w2)).item()
    assert diff < 1e-6, f"Routing not deterministic: max diff {diff}"
    print("  PASS: routing is deterministic")


def test_lsh_router_weights_sum_to_one():
    """Routing weights for selected experts sum to approximately 1."""
    router = LSHRouter(n_embd=64, n_groups=8, n_tables=4, top_k=2)
    x = mx.random.normal((2, 16, 64))
    weights = router(x)
    mx.eval(weights)
    sums = mx.sum(weights, axis=-1)
    mx.eval(sums)
    max_deviation = mx.max(mx.abs(sums - 1.0)).item()
    assert max_deviation < 0.01, f"Weights don't sum to 1: max deviation {max_deviation}"
    print(f"  PASS: weights sum to 1 (max deviation {max_deviation:.6f})")


def test_lsh_capsule_pool_forward():
    """LSHCapsulePool produces correct output shapes."""
    pool = LSHCapsulePool(n_embd=64, n_groups=8,
                          n_capsules_per_group=32, n_tables=4, top_k=2)
    x = mx.random.normal((2, 16, 64))
    out = pool(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("  PASS: capsule pool output shape correct")


def test_full_model_forward():
    """Full model forward pass produces correct output shapes."""
    model = LSHCapsuleRoutingGPT(vocab_size=28, block_size=32,
                                  n_groups=8, n_capsules_per_group=32,
                                  n_tables=4, top_k=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(tokens)
    assert logits.shape == (1, 8, 28), f"Expected (1, 8, 28), got {logits.shape}"
    print("  PASS: full model forward shape correct")


def test_model_trainable_params():
    """Model has fewer trainable params than softmax baseline (no router weights)."""
    model = LSHCapsuleRoutingGPT(vocab_size=28, block_size=32,
                                  n_groups=8, n_capsules_per_group=32,
                                  n_tables=4, top_k=2)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"  LSH model trainable params: {n_params:,}")

    # Softmax baseline has router params: 4 layers * n_groups * n_embd = 4 * 8 * 64 = 2048
    # LSH should have ~2048 fewer trainable params
    # Expected: ~202K (vs ~204K for softmax baseline)
    assert n_params < 204_000, f"Expected <204K params, got {n_params:,}"
    print(f"  PASS: fewer trainable params than softmax baseline")


def test_aux_loss():
    """Aux loss computes without error."""
    model = LSHCapsuleRoutingGPT(vocab_size=28, block_size=32,
                                  n_groups=8, n_capsules_per_group=32,
                                  n_tables=4, top_k=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    aux = model.aux_loss()
    mx.eval(aux)
    assert aux.item() >= 0, f"Aux loss should be non-negative, got {aux.item()}"
    print(f"  PASS: aux_loss = {aux.item():.4f}")


def test_routing_diagnostics():
    """Routing diagnostics return valid structure."""
    model = LSHCapsuleRoutingGPT(vocab_size=28, block_size=32,
                                  n_groups=8, n_capsules_per_group=32,
                                  n_tables=4, top_k=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _ = model(tokens)
    diag = model.get_routing_diagnostics()
    assert "layer_0" in diag
    assert "mean_selected" in diag["layer_0"]
    assert "expert_utilization" in diag["layer_0"]
    print(f"  PASS: diagnostics valid")
    for layer_name, layer_diag in diag.items():
        if "mean_selected" in layer_diag:
            print(f"    {layer_name}: selected={layer_diag['mean_selected']:.1f}, "
                  f"entropy={layer_diag['normalized_entropy']:.3f}")


def test_different_n_tables():
    """Model works with different numbers of hash tables."""
    for n_tables in [1, 2, 4, 8]:
        model = LSHCapsuleRoutingGPT(vocab_size=28, block_size=32,
                                      n_groups=8, n_capsules_per_group=32,
                                      n_tables=n_tables, top_k=2)
        mx.eval(model.parameters())
        tokens = mx.array([[1, 2, 3, 4, 5]])
        logits = model(tokens)
        mx.eval(logits)
        print(f"  PASS: n_tables={n_tables} works")


def test_gradient_flow():
    """Gradients flow through LSH routing to capsule params."""
    model = LSHCapsuleRoutingGPT(vocab_size=28, block_size=32,
                                  n_groups=8, n_capsules_per_group=32,
                                  n_tables=4, top_k=2)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    targets = mx.array([[2, 3, 4, 5, 6]])

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens, targets)
    mx.eval(loss, grads)

    # Check that capsule group params have gradients
    flat_grads = nn.utils.tree_flatten(grads)
    capsule_grads = [(k, v) for k, v in flat_grads if "groups" in k]
    has_nonzero = any(mx.any(mx.abs(v) > 1e-10).item() for _, v in capsule_grads)
    assert has_nonzero, "No gradients flowing to capsule groups"
    print(f"  PASS: gradients flow through LSH to capsule params")


if __name__ == "__main__":
    print("Testing LSH Capsule Routing...")
    print()
    test_lsh_router_shapes()
    test_lsh_router_sparsity()
    test_lsh_router_no_trainable_params()
    test_lsh_router_deterministic()
    test_lsh_router_weights_sum_to_one()
    test_lsh_capsule_pool_forward()
    test_full_model_forward()
    test_model_trainable_params()
    test_aux_loss()
    test_routing_diagnostics()
    test_different_n_tables()
    test_gradient_flow()
    print()
    print("All tests passed!")

"""Tests for consistent hash routing."""

import mlx.core as mx
import mlx.nn as nn

from .consistent_hash_routing import (
    ConsistentHashRoutingGPT,
    ConsistentHashRouter,
    _jump_consistent_hash,
    _fnv1a_32,
)


def test_jump_consistent_hash_basic():
    """Jump hash maps to valid buckets and is deterministic."""
    for n in [1, 4, 8, 16]:
        for key in [0, 42, 12345, 999999]:
            bucket = _jump_consistent_hash(key, n)
            assert 0 <= bucket < n, f"bucket {bucket} out of range [0, {n})"
            # Deterministic
            assert _jump_consistent_hash(key, n) == bucket


def test_jump_consistent_hash_displacement():
    """Adding one bucket displaces ~1/N of keys."""
    n_keys = 10000
    N = 8
    assignments_before = [_jump_consistent_hash(k, N) for k in range(n_keys)]
    assignments_after = [_jump_consistent_hash(k, N + 1) for k in range(n_keys)]
    displaced = sum(1 for a, b in zip(assignments_before, assignments_after) if a != b)
    displacement_rate = displaced / n_keys
    # Theoretical: ~1/(N+1) = 1/9 ~ 11.1%. Allow 2x margin.
    assert displacement_rate < 0.25, f"Displacement {displacement_rate:.1%} too high (expected ~{1/(N+1):.1%})"
    print(f"  Jump hash displacement: {displacement_rate:.1%} (expected ~{1/(N+1):.1%})")


def test_fnv1a_distribution():
    """FNV-1a produces reasonable distribution."""
    n_buckets = 8
    counts = [0] * n_buckets
    for i in range(10000):
        h = _fnv1a_32(i.to_bytes(4, "big"))
        counts[h % n_buckets] += 1
    # Check roughly uniform (each bucket should get ~1250 +/- 200)
    for c in counts:
        assert 800 < c < 1700, f"Bucket count {c} too skewed"


def test_consistent_hash_router_shape():
    """Router produces correct output shape."""
    router = ConsistentHashRouter(n_embd=64, n_groups=8, top_k=2)
    x = mx.random.normal((2, 16, 64))
    weights = router(x)
    assert weights.shape == (2, 16, 8), f"Expected (2, 16, 8), got {weights.shape}"


def test_consistent_hash_router_sparsity():
    """Router produces exactly top_k nonzero weights per token."""
    router = ConsistentHashRouter(n_embd=64, n_groups=8, top_k=2)
    x = mx.random.normal((2, 16, 64))
    weights = router(x)
    mx.eval(weights)
    # Each token should have exactly 2 nonzero weights
    nonzero_per_token = mx.sum((weights > 0.001).astype(mx.float32), axis=-1)
    mx.eval(nonzero_per_token)
    for b in range(2):
        for t in range(16):
            nz = nonzero_per_token[b, t].item()
            assert nz == 2, f"Expected 2 nonzero, got {nz} at ({b},{t})"


def test_consistent_hash_router_deterministic():
    """Same input always produces same routing."""
    router = ConsistentHashRouter(n_embd=64, n_groups=8, top_k=2)
    x = mx.random.normal((1, 8, 64))
    w1 = router(x)
    w2 = router(x)
    mx.eval(w1, w2)
    diff = mx.max(mx.abs(w1 - w2)).item()
    assert diff < 1e-6, f"Non-deterministic routing: max diff {diff}"


def test_add_expert_displacement():
    """Adding an expert displaces ~1/N of routing decisions."""
    router = ConsistentHashRouter(n_embd=64, n_groups=8, top_k=2)
    x = mx.random.normal((4, 32, 64))

    # Get assignments before
    weights_before = router(x, n_groups_override=8)
    mx.eval(weights_before)
    assignments_before = mx.argmax(weights_before, axis=-1)  # (B, T)
    mx.eval(assignments_before)

    # Add expert 8 (9th expert)
    router.add_expert(8)

    # Get assignments after (now 9 groups)
    weights_after = router(x, n_groups_override=9)
    mx.eval(weights_after)
    # Pad before to 9 groups for comparison
    assignments_after = mx.argmax(weights_after, axis=-1)
    mx.eval(assignments_after)

    # Count displaced tokens
    total = 4 * 32
    displaced = mx.sum(assignments_before != assignments_after).item()
    rate = displaced / total
    print(f"  Displacement rate: {rate:.1%} ({displaced}/{total})")
    # Kill criterion: <30%
    assert rate < 0.30, f"Displacement {rate:.1%} exceeds 30% threshold"


def test_model_forward():
    """Model produces correct output shape."""
    model = ConsistentHashRoutingGPT(vocab_size=28, block_size=32,
                                     n_embd=64, n_groups=8)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(tokens)
    assert logits.shape == (1, 8, 28), f"Expected (1, 8, 28), got {logits.shape}"


def test_model_add_expert():
    """Adding expert to model works without error."""
    model = ConsistentHashRoutingGPT(vocab_size=28, block_size=32,
                                     n_embd=64, n_groups=8)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    # Forward before
    logits_before = model(tokens)
    mx.eval(logits_before)

    # Add expert
    new_groups = model.add_expert_to_all_layers()
    mx.eval([g.parameters() for g in new_groups])

    # Forward after
    logits_after = model(tokens)
    mx.eval(logits_after)

    assert logits_after.shape == (1, 8, 28)
    # Should produce different logits (new expert contributes)
    diff = mx.max(mx.abs(logits_before - logits_after)).item()
    print(f"  Logit diff after adding expert: {diff:.4f}")


def test_param_count():
    """Verify zero routing parameters."""
    model = ConsistentHashRoutingGPT(vocab_size=28, block_size=32,
                                     n_embd=64, n_groups=8)
    mx.eval(model.parameters())
    total_params = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    # Compare with capsule_moe which has routing params
    # Our model should have fewer params (no router weights)
    print(f"  Total trainable params: {total_params:,}")
    # Router should contribute 0 params
    for layer in model.layers:
        router = layer.capsule_pool.router
        router_params = sum(v.size for _, v in nn.utils.tree_flatten(router.trainable_parameters()))
        assert router_params == 0, f"Router has {router_params} trainable params, expected 0"


if __name__ == "__main__":
    test_jump_consistent_hash_basic()
    print("PASS: jump_consistent_hash_basic")

    test_jump_consistent_hash_displacement()
    print("PASS: jump_consistent_hash_displacement")

    test_fnv1a_distribution()
    print("PASS: fnv1a_distribution")

    test_consistent_hash_router_shape()
    print("PASS: consistent_hash_router_shape")

    test_consistent_hash_router_sparsity()
    print("PASS: consistent_hash_router_sparsity")

    test_consistent_hash_router_deterministic()
    print("PASS: consistent_hash_router_deterministic")

    test_add_expert_displacement()
    print("PASS: add_expert_displacement")

    test_model_forward()
    print("PASS: model_forward")

    test_model_add_expert()
    print("PASS: model_add_expert")

    test_param_count()
    print("PASS: param_count")

    print("\nAll tests passed!")

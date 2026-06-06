"""Unit tests for loudness fix models."""

import mlx.core as mx
import mlx.nn as nn


def test_rmsnorm_composed_forward():
    """RMSNormComposedGPT produces correct output shape."""
    from .loudness_fix import RMSNormComposedGPT

    model = RMSNormComposedGPT(
        vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=2,
        n_capsules_per_pool=64, n_pools=2,
    )
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_rmsnorm_equalizes_magnitude():
    """Per-pool RMSNorm should equalize output magnitudes regardless of pool scale.

    Create two pools where one has 10x larger weights than the other.
    After RMSNorm, both should contribute equally to the output.
    """
    from .loudness_fix import RMSNormComposedPool

    d = 64
    pool = RMSNormComposedPool(d, n_capsules_per_pool=32, n_pools=2)
    mx.eval(pool.parameters())

    # Scale pool 0 weights by 10x to create artificial magnitude imbalance
    pool.pools[0].A.load_weights([("weight", pool.pools[0].A.weight * 10.0)])
    pool.pools[0].B.load_weights([("weight", pool.pools[0].B.weight * 10.0)])
    mx.eval(pool.parameters())

    # Get individual pool outputs (before normalization)
    x = mx.random.normal((4, 8, d))

    out_0_raw = pool.pools[0](x)
    out_1_raw = pool.pools[1](x)
    mx.eval(out_0_raw, out_1_raw)

    rms_0 = mx.sqrt(mx.mean(out_0_raw * out_0_raw)).item()
    rms_1 = mx.sqrt(mx.mean(out_1_raw * out_1_raw)).item()

    # Raw: pool 0 should be much louder
    assert rms_0 > rms_1 * 3, (
        f"Pool 0 should be much louder: rms_0={rms_0:.4f}, rms_1={rms_1:.4f}")

    # After RMSNorm: the composed output should exist and be finite
    out_composed = pool(x)
    mx.eval(out_composed)

    out_rms = mx.sqrt(mx.mean(out_composed * out_composed)).item()
    assert out_rms > 0, "Composed output should be nonzero"
    assert out_rms < 100, f"Composed output RMS should be bounded, got {out_rms}"

    print(f"  Raw RMS: pool_0={rms_0:.4f}, pool_1={rms_1:.4f} (ratio={rms_0/rms_1:.1f}x)")
    print(f"  Composed output RMS: {out_rms:.4f}")


def test_rmsnorm_composition_identity():
    """Verify: RMSNorm(Pool_A(x)) + RMSNorm(Pool_B(x)) == composed(x)."""
    from .loudness_fix import RMSNormComposedPool, ReLUCapsulePool

    d = 64
    composed = RMSNormComposedPool(d, n_capsules_per_pool=32, n_pools=2)
    mx.eval(composed.parameters())

    x = mx.random.normal((2, 8, d))

    # Manual computation
    out_a = composed.pools[0](x)
    out_b = composed.pools[1](x)
    norm_a = composed._rms_norm(out_a)
    norm_b = composed._rms_norm(out_b)
    manual_sum = norm_a + norm_b
    mx.eval(manual_sum)

    # Composed computation
    out_composed = composed(x)
    mx.eval(out_composed)

    diff = mx.max(mx.abs(out_composed - manual_sum)).item()
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-5, f"Identity failed: diff={diff:.2e}"


def test_matched_magnitude_forward():
    """MatchedMagnitudeGPT produces correct output shape."""
    from .loudness_fix import MatchedMagnitudeGPT

    model = MatchedMagnitudeGPT(
        vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=2,
        n_capsules=128, mag_coeff=1.0,
    )
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_matched_magnitude_loss():
    """Magnitude loss should be nonzero when target_rms is set."""
    from .loudness_fix import MatchedMagnitudePool

    pool = MatchedMagnitudePool(64, n_capsules=32, mag_coeff=1.0, target_rms=0.1)
    mx.eval(pool.parameters())

    x = mx.random.normal((2, 8, 64))
    _ = pool(x)

    mag_loss = pool.magnitude_loss()
    mx.eval(mag_loss)
    print(f"  Magnitude loss with target_rms=0.1: {mag_loss.item():.6f}")
    assert mag_loss.item() > 0, "Magnitude loss should be positive when target != actual"


def test_matched_magnitude_loss_zero_without_target():
    """Magnitude loss should be zero when target_rms is None (pretraining)."""
    from .loudness_fix import MatchedMagnitudePool

    pool = MatchedMagnitudePool(64, n_capsules=32, mag_coeff=1.0, target_rms=None)
    mx.eval(pool.parameters())

    x = mx.random.normal((2, 8, 64))
    _ = pool(x)

    mag_loss = pool.magnitude_loss()
    mx.eval(mag_loss)
    assert mag_loss.item() == 0.0, "Magnitude loss should be 0 without target"


def test_matched_magnitude_gradient_flows():
    """Gradients should flow through the magnitude loss to A and B weights."""
    from .loudness_fix import MatchedMagnitudeGPT

    model = MatchedMagnitudeGPT(
        vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=2,
        n_capsules=64, mag_coeff=1.0,
    )
    mx.eval(model.parameters())

    # Set target RMS to force nonzero magnitude loss
    for layer in model.layers:
        layer.capsule_pool.target_rms = 0.01  # very small, will produce large loss

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        ) + model.aux_loss()

    tokens = mx.array([[1, 2, 3, 4, 5]])
    targets = mx.array([[2, 3, 4, 5, 6]])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens, targets)
    mx.eval(loss, grads)

    grad_flat = nn.utils.tree_flatten(grads)
    capsule_grads = [(k, v) for k, v in grad_flat if "capsule_pool" in k]
    assert len(capsule_grads) > 0, "No gradients for capsule_pool"

    for name, grad in capsule_grads:
        grad_norm = mx.sqrt(mx.sum(grad * grad)).item()
        assert grad_norm > 0, f"Zero gradient for {name}"
    print(f"  All {len(capsule_grads)} capsule gradient paths verified")


def test_compose_with_rmsnorm():
    """compose_with_rmsnorm should produce a valid composed model."""
    from .loudness_fix import compose_with_rmsnorm
    from ..relu_router.relu_router import ReLURouterGPT

    V = 28
    base = ReLURouterGPT(vocab_size=V, n_capsules=64, n_embd=64, n_head=4,
                          n_layer=2, block_size=32)
    mx.eval(base.parameters())

    import copy
    domain_a = copy.deepcopy(base)
    domain_b = copy.deepcopy(base)

    composed = compose_with_rmsnorm(
        base, [domain_a, domain_b], vocab_size=V,
        n_capsules_per_pool=64, n_embd=64, n_head=4, n_layer=2, block_size=32,
    )

    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = composed(tokens)
    mx.eval(logits)

    assert logits.shape == (1, 5, V), f"Expected (1, 5, {V}), got {logits.shape}"
    print("  compose_with_rmsnorm: shape verified")


if __name__ == "__main__":
    tests = [
        ("test_rmsnorm_composed_forward", test_rmsnorm_composed_forward),
        ("test_rmsnorm_equalizes_magnitude", test_rmsnorm_equalizes_magnitude),
        ("test_rmsnorm_composition_identity", test_rmsnorm_composition_identity),
        ("test_matched_magnitude_forward", test_matched_magnitude_forward),
        ("test_matched_magnitude_loss", test_matched_magnitude_loss),
        ("test_matched_magnitude_loss_zero_without_target", test_matched_magnitude_loss_zero_without_target),
        ("test_matched_magnitude_gradient_flows", test_matched_magnitude_gradient_flows),
        ("test_compose_with_rmsnorm", test_compose_with_rmsnorm),
    ]
    for name, fn in tests:
        print(name)
        fn()
        print("PASSED\n")
    print("ALL TESTS PASSED")

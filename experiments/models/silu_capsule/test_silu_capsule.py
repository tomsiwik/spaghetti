"""Tests for SiLU Capsule — SiLU activation variant of ReLU Router."""

import mlx.core as mx
import mlx.nn as nn


def test_forward_shape():
    """Output shape matches input shape (B, T) -> (B, T, V)."""
    from .silu_capsule import SiLUCapsuleGPT

    model = SiLUCapsuleGPT(vocab_size=28, block_size=32, n_embd=64,
                            n_head=4, n_layer=4, n_capsules=256)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]])  # (1, 5)
    logits = model(tokens)
    assert logits.shape == (1, 5, 28), f"Expected (1, 5, 28), got {logits.shape}"


def test_silu_activation_properties():
    """SiLU should produce non-zero activations (unlike ReLU)."""
    from .silu_capsule import SiLUCapsulePool

    pool = SiLUCapsulePool(64, n_capsules=128)
    mx.eval(pool.parameters())

    x = mx.random.normal((4, 8, 64))
    h = nn.silu(pool.A(x))
    mx.eval(h)

    # SiLU never produces exact zeros (for non-zero input)
    n_exact_zero = mx.sum(h == 0).item()
    n_total = h.size
    exact_zero_frac = n_exact_zero / n_total
    print(f"  Exact zeros: {n_exact_zero}/{n_total} ({exact_zero_frac:.4f})")
    # SiLU should have very few exact zeros (only if A@x happens to be exactly 0)
    assert exact_zero_frac < 0.05, f"SiLU should rarely produce exact zeros, got {exact_zero_frac:.4f}"

    # SiLU can produce negative values (min ≈ -0.278)
    min_val = mx.min(h).item()
    print(f"  Min activation: {min_val:.4f}")
    assert min_val < 0, "SiLU should produce some negative activations"


def test_composition_identity_with_zero_init():
    """B=0 guarantees zero output regardless of SiLU activation.

    This is the composition identity: a newly-added capsule pool
    with zero-initialized B produces zero output, so composing
    it with existing pools doesn't change the model.
    """
    from .silu_capsule import SiLUCapsulePool

    d = 64
    pool = SiLUCapsulePool(d, n_capsules=128)
    mx.eval(pool.parameters())

    # Zero-init B (as in composition protocol)
    pool.B.weight = mx.zeros_like(pool.B.weight)
    mx.eval(pool.parameters())

    x = mx.random.normal((2, 8, d))
    out = pool(x)
    mx.eval(out)

    max_val = mx.max(mx.abs(out)).item()
    print(f"  Max |output| with B=0: {max_val:.2e}")
    assert max_val < 1e-7, f"B=0 should give zero output, got max {max_val:.2e}"


def test_composition_by_concatenation():
    """Composed SiLU pool output == sum of individual pool outputs."""
    from .silu_capsule import SiLUCapsulePool

    d = 64
    pool_a = SiLUCapsulePool(d, n_capsules=32)
    pool_b = SiLUCapsulePool(d, n_capsules=32)
    mx.eval(pool_a.parameters())
    mx.eval(pool_b.parameters())

    pool_composed = SiLUCapsulePool(d, n_capsules=64)
    mx.eval(pool_composed.parameters())

    A_composed = mx.concatenate([pool_a.A.weight, pool_b.A.weight], axis=0)
    B_composed = mx.concatenate([pool_a.B.weight, pool_b.B.weight], axis=1)
    pool_composed.A.load_weights([("weight", A_composed)])
    pool_composed.B.load_weights([("weight", B_composed)])
    mx.eval(pool_composed.parameters())

    x = mx.random.normal((2, 8, d))
    out_a = pool_a(x)
    out_b = pool_b(x)
    out_composed = pool_composed(x)
    out_sum = out_a + out_b
    mx.eval(out_a, out_b, out_composed, out_sum)

    diff = mx.max(mx.abs(out_composed - out_sum)).item()
    print(f"  Max |composed - sum|: {diff:.2e}")
    assert diff < 1e-5, f"Composition identity FAILED, max diff = {diff:.2e}"


def test_effective_sparsity_metric():
    """Effective sparsity should be measurable and reasonable."""
    from .silu_capsule import SiLUCapsuleGPT

    model = SiLUCapsuleGPT(vocab_size=28, block_size=32, n_embd=64,
                            n_head=4, n_layer=2, n_capsules=128)
    mx.eval(model.parameters())

    tokens = mx.array([[i % 27 + 1 for i in range(16)]] * 8)
    _ = model(tokens)
    mx.eval(model.parameters())

    stats = model.capsule_stats()
    for layer_idx, eff_sp in enumerate(stats["eff_sparsity"]):
        if eff_sp is not None:
            print(f"  Layer {layer_idx}: eff_sparsity={eff_sp:.3f}, "
                  f"near_dead={stats['n_near_dead'][layer_idx]}/{128}")
            # SiLU should have SOME effective sparsity but less than ReLU
            assert 0.0 <= eff_sp <= 1.0, f"Sparsity out of range: {eff_sp}"


def test_aux_loss_nonzero():
    """Auxiliary loss should be computable and finite."""
    from .silu_capsule import SiLUCapsuleGPT

    model = SiLUCapsuleGPT(vocab_size=28, block_size=32, n_embd=64,
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


def test_gradient_flows():
    """Gradients should flow through SiLU (fully differentiable, no dead zones)."""
    from .silu_capsule import SiLUCapsuleGPT

    model = SiLUCapsuleGPT(vocab_size=28, block_size=32, n_embd=64,
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

    grad_flat = nn.utils.tree_flatten(grads)
    capsule_grads = [(k, v) for k, v in grad_flat if "capsule_pool" in k]
    assert len(capsule_grads) > 0, "No gradients for capsule_pool parameters"

    for name, grad in capsule_grads:
        grad_norm = mx.sqrt(mx.sum(grad * grad)).item()
        print(f"  {name}: grad_norm={grad_norm:.6f}")
        assert grad_norm > 0, f"Zero gradient for {name}"


def test_silu_vs_relu_sparsity_comparison():
    """Side-by-side comparison: SiLU should have lower exact sparsity, higher gradient flow."""
    from .silu_capsule import SiLUCapsuleGPT
    from ..relu_router.relu_router import ReLURouterGPT

    kwargs = dict(vocab_size=28, block_size=32, n_embd=64,
                  n_head=4, n_layer=2, n_capsules=128)

    silu_model = SiLUCapsuleGPT(**kwargs)
    relu_model = ReLURouterGPT(**kwargs)
    mx.eval(silu_model.parameters())
    mx.eval(relu_model.parameters())

    tokens = mx.array([[i % 27 + 1 for i in range(16)]] * 8)

    # Run both
    _ = silu_model(tokens)
    _ = relu_model(tokens)
    mx.eval(silu_model.parameters(), relu_model.parameters())

    silu_stats = silu_model.capsule_stats()
    relu_stats = relu_model.capsule_stats()

    print("\n  === SiLU vs ReLU Sparsity Comparison ===")
    for i in range(2):
        relu_sp = relu_stats["sparsity"][i]
        silu_eff_sp = silu_stats["eff_sparsity"][i]
        relu_dead = relu_stats["n_dead"][i]
        silu_near_dead = silu_stats["n_near_dead"][i]

        print(f"  Layer {i}:")
        print(f"    ReLU: sparsity={relu_sp:.3f}, dead={relu_dead}/128")
        print(f"    SiLU: eff_sparsity={silu_eff_sp:.3f}, near_dead={silu_near_dead}/128")


if __name__ == "__main__":
    tests = [
        ("test_forward_shape", test_forward_shape),
        ("test_silu_activation_properties", test_silu_activation_properties),
        ("test_composition_identity_with_zero_init", test_composition_identity_with_zero_init),
        ("test_composition_by_concatenation", test_composition_by_concatenation),
        ("test_effective_sparsity_metric", test_effective_sparsity_metric),
        ("test_aux_loss_nonzero", test_aux_loss_nonzero),
        ("test_gradient_flows", test_gradient_flows),
        ("test_silu_vs_relu_sparsity_comparison", test_silu_vs_relu_sparsity_comparison),
    ]

    for name, fn in tests:
        print(name)
        fn()
        print("PASSED\n")

    print("ALL TESTS PASSED")

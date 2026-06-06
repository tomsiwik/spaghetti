"""Tests for TrainingCompatGPT: verify aux losses, snapshot, composition."""

import mlx.core as mx
import mlx.nn as nn


def test_model_creation():
    """Model creates with correct parameter count."""
    from experiments.models.training_compat.training_compat import TrainingCompatGPT

    model = TrainingCompatGPT(vocab_size=28, n_capsules=128, n_embd=64)
    mx.eval(model.parameters())

    tokens = mx.zeros((2, 32), dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (2, 32, 28), f"Expected (2, 32, 28), got {logits.shape}"
    print("PASS: model creation")


def test_aux_loss_zero_before_snapshot():
    """Aux losses should be zero before snapshot is set."""
    from experiments.models.training_compat.training_compat import TrainingCompatGPT

    model = TrainingCompatGPT(vocab_size=28, n_capsules=128, n_embd=64,
                              ortho_coeff=0.1, norm_coeff=0.1)
    mx.eval(model.parameters())

    tokens = mx.zeros((2, 32), dtype=mx.int32)
    _ = model(tokens)

    # Before snapshot, ortho loss should be 0 (no base reference)
    for layer in model.layers:
        ortho = layer.capsule_pool.ortho_loss()
        mx.eval(ortho)
        assert ortho.item() == 0.0, f"Ortho loss should be 0 before snapshot, got {ortho.item()}"

    print("PASS: aux loss zero before snapshot")


def test_aux_loss_nonzero_after_snapshot():
    """After snapshot + weight change, ortho loss should be nonzero."""
    from experiments.models.training_compat.training_compat import TrainingCompatGPT

    model = TrainingCompatGPT(vocab_size=28, n_capsules=128, n_embd=64,
                              ortho_coeff=0.1, norm_coeff=0.1)
    mx.eval(model.parameters())

    # Snapshot
    model.snapshot_base()

    # Perturb weights
    for layer in model.layers:
        pool = layer.capsule_pool
        new_A = pool.A.weight + mx.random.normal(pool.A.weight.shape) * 0.1
        pool.A.load_weights([("weight", new_A)])
        mx.eval(pool.A.weight)

    # Forward pass to populate norm stats
    tokens = mx.zeros((2, 32), dtype=mx.int32)
    _ = model(tokens)

    # Now ortho loss should be nonzero
    total_ortho = 0.0
    for layer in model.layers:
        ortho = layer.capsule_pool.ortho_loss()
        mx.eval(ortho)
        total_ortho += ortho.item()

    assert total_ortho > 0.0, f"Ortho loss should be > 0 after perturbation, got {total_ortho}"
    print(f"PASS: aux loss nonzero after snapshot (ortho={total_ortho:.4f})")


def test_composition():
    """Compose two domain models by weight concatenation."""
    from experiments.models.training_compat.training_compat import TrainingCompatGPT
    from experiments.models.training_compat.run_experiment import compose_models

    V = 28
    model_a = TrainingCompatGPT(vocab_size=V, n_capsules=64, n_embd=64)
    model_b = TrainingCompatGPT(vocab_size=V, n_capsules=64, n_embd=64)
    mx.eval(model_a.parameters())
    mx.eval(model_b.parameters())

    composed = compose_models(model_a, [model_a, model_b], V)

    # Check composed capsule count
    for layer in composed.layers:
        n_caps = layer.capsule_pool.n_capsules
        assert n_caps == 128, f"Expected 128 capsules, got {n_caps}"

    # Forward pass should work
    tokens = mx.zeros((2, 32), dtype=mx.int32)
    logits = composed(tokens)
    assert logits.shape == (2, 32, V)
    print("PASS: composition works")


def test_norm_loss():
    """Norm loss should be nonzero when output norm deviates from target."""
    from experiments.models.training_compat.training_compat import TrainingCompatGPT

    model = TrainingCompatGPT(vocab_size=28, n_capsules=128, n_embd=64,
                              ortho_coeff=0.0, norm_coeff=0.1)
    mx.eval(model.parameters())

    # Set extreme target ratio
    for layer in model.layers:
        layer.capsule_pool.target_norm_ratio = 100.0

    tokens = mx.zeros((2, 32), dtype=mx.int32)
    _ = model(tokens)

    total_norm_loss = 0.0
    for layer in model.layers:
        nl = layer.capsule_pool.norm_loss()
        mx.eval(nl)
        total_norm_loss += nl.item()

    assert total_norm_loss > 0.0, f"Norm loss should be > 0 with extreme target, got {total_norm_loss}"
    print(f"PASS: norm loss nonzero with mismatched target ({total_norm_loss:.4f})")


def test_gradient_flows():
    """Verify gradients flow through aux losses."""
    from experiments.models.training_compat.training_compat import TrainingCompatGPT

    model = TrainingCompatGPT(vocab_size=28, n_capsules=64, n_embd=64,
                              ortho_coeff=0.1, norm_coeff=0.1)
    mx.eval(model.parameters())
    model.snapshot_base()

    # Perturb to ensure nonzero loss
    for layer in model.layers:
        pool = layer.capsule_pool
        new_A = pool.A.weight + mx.random.normal(pool.A.weight.shape) * 0.1
        pool.A.load_weights([("weight", new_A)])
    mx.eval(model.parameters())

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        ce = nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean")
        return ce + model.aux_loss()

    tokens = mx.ones((2, 32), dtype=mx.int32)
    targets = mx.ones((2, 32), dtype=mx.int32)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens, targets)
    mx.eval(loss, grads)

    # Check that gradients exist for capsule weights
    grad_dict = dict(nn.utils.tree_flatten(grads))
    capsule_grads = {k: v for k, v in grad_dict.items() if "capsule_pool" in k}
    assert len(capsule_grads) > 0, "No capsule pool gradients found"

    nonzero_count = 0
    for k, v in capsule_grads.items():
        if mx.any(v != 0).item():
            nonzero_count += 1

    assert nonzero_count > 0, "All capsule pool gradients are zero"
    print(f"PASS: gradients flow ({nonzero_count}/{len(capsule_grads)} nonzero capsule grad tensors)")


if __name__ == "__main__":
    test_model_creation()
    test_aux_loss_zero_before_snapshot()
    test_aux_loss_nonzero_after_snapshot()
    test_norm_loss()
    test_composition()
    test_gradient_flows()
    print("\nAll tests passed!")

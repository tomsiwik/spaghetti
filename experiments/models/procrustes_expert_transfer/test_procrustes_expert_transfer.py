"""Tests for Procrustes Expert Transfer experiment."""

import numpy as np
import torch
import pytest


def test_orthogonal_procrustes_identity():
    """Procrustes of A with itself should give zero residual."""
    from experiments.models.procrustes_expert_transfer.procrustes_expert_transfer import (
        orthogonal_procrustes,
    )
    np.random.seed(42)
    # Use square matrix so R is unique (identity)
    A = np.random.randn(8, 8)
    R, residual = orthogonal_procrustes(A, A)
    assert residual < 1e-6, f"Residual should be ~0, got {residual}"
    # R @ A should equal A
    assert np.allclose(R @ A, A, atol=1e-5), "R @ A should equal A"


def test_orthogonal_procrustes_known_rotation():
    """Procrustes should recover a known rotation."""
    from experiments.models.procrustes_expert_transfer.procrustes_expert_transfer import (
        orthogonal_procrustes,
    )
    np.random.seed(42)
    d = 8
    A = np.random.randn(d, d)

    # Create a known rotation
    theta = 0.3
    R_true = np.eye(d)
    R_true[0, 0] = np.cos(theta)
    R_true[0, 1] = -np.sin(theta)
    R_true[1, 0] = np.sin(theta)
    R_true[1, 1] = np.cos(theta)

    B = R_true @ A
    R_recovered, residual = orthogonal_procrustes(A, B)

    assert residual < 1e-6, f"Residual should be ~0, got {residual}"
    assert np.allclose(R_recovered, R_true, atol=1e-5), "Should recover the rotation"


def test_procrustes_is_orthogonal():
    """Procrustes result should be orthogonal (R^T R = I)."""
    from experiments.models.procrustes_expert_transfer.procrustes_expert_transfer import (
        orthogonal_procrustes,
    )
    np.random.seed(42)
    A = np.random.randn(8, 5)
    B = np.random.randn(8, 5)
    R, _ = orthogonal_procrustes(A, B)

    assert np.allclose(R @ R.T, np.eye(8), atol=1e-6), "R should be orthogonal"
    assert np.linalg.det(R) > 0, "R should be a proper rotation (det > 0)"


def test_transform_deltas_identity():
    """Identity alignment should not change deltas."""
    from experiments.models.procrustes_expert_transfer.procrustes_expert_transfer import (
        transform_expert_deltas,
    )
    d = 8
    deltas = [
        (0, "fc1", torch.randn(d, 4 * d)),
        (0, "fc2", torch.randn(4 * d, d)),
    ]
    alignments = {0: {"R": np.eye(d), "residual": 0.0, "n_samples": 100}}

    transformed = transform_expert_deltas(deltas, alignments, d)

    for (_, _, orig), (_, _, trans) in zip(deltas, transformed):
        assert torch.allclose(orig, trans, atol=1e-5), "Identity alignment should preserve deltas"


def test_naive_vs_procrustes_comparison():
    """Activation Procrustes should not be worse than naive on average."""
    import json
    import os

    agg_path = os.path.join(
        os.path.dirname(__file__), "results_aggregate.json"
    )
    if not os.path.exists(agg_path):
        pytest.skip("Run experiment first to generate results_aggregate.json")

    with open(agg_path) as f:
        agg = json.load(f)

    naive_ratio = agg["methods"]["naive_transfer"]["mean_ratio"]
    act_ratio = agg["methods"]["procrustes_activation"]["mean_ratio"]

    # Activation Procrustes should improve over naive
    assert act_ratio <= naive_ratio + 0.01, (
        f"Activation Procrustes ({act_ratio:.4f}) should not be significantly "
        f"worse than naive ({naive_ratio:.4f})"
    )


def test_k1_survives():
    """K1: transferred expert PPL should be <20% worse than native."""
    import json
    import os

    agg_path = os.path.join(
        os.path.dirname(__file__), "results_aggregate.json"
    )
    if not os.path.exists(agg_path):
        pytest.skip("Run experiment first")

    with open(agg_path) as f:
        agg = json.load(f)

    for method in ["naive_transfer", "procrustes_activation"]:
        ratio = agg["methods"][method]["mean_ratio"]
        assert ratio < 1.20, f"{method} ratio {ratio:.4f} exceeds K1 threshold 1.20"


def test_experiment_runs():
    """Smoke test: experiment completes without errors."""
    from experiments.models.procrustes_expert_transfer.procrustes_expert_transfer import (
        run_experiment,
    )
    r = run_experiment(
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=16,
        lora_rank=4,
        pretrain_steps=100,
        expert_train_steps=50,
        n_experts=2,
        seed=99,
    )
    assert r is not None
    assert "KILLED" in r.verdict or "SURVIVES" in r.verdict or "INCONCLUSIVE" in r.verdict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

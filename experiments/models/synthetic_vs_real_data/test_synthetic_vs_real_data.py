#!/usr/bin/env python3
"""Tests for synthetic_vs_real_data simulation."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_vs_real_data import (
    SYNTHETIC_CFG, REAL_CFG, D_MODEL, RANK,
    make_mode_centers, generate_inputs, generate_labels,
    train_lora, quality, effective_rank, mean_cos, contamination_stats,
)


def test_mode_centers():
    """Mode centers should be unit norm."""
    rng = np.random.default_rng(42)
    centers = make_mode_centers(10, D_MODEL, rng)
    norms = np.linalg.norm(centers, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_generate_inputs_shape():
    """Inputs should have correct shape."""
    rng = np.random.default_rng(42)
    modes = make_mode_centers(SYNTHETIC_CFG.n_input_modes, D_MODEL, rng)
    X = generate_inputs(SYNTHETIC_CFG, 100, D_MODEL, rng, modes)
    assert X.shape == (100, D_MODEL)


def test_label_noise_levels():
    """Synthetic labels should have lower noise than real."""
    rng = np.random.default_rng(42)
    W_star = rng.standard_normal((D_MODEL, D_MODEL)) * 0.1
    X = rng.standard_normal((5000, D_MODEL))

    y_synth = generate_labels(X, W_star, SYNTHETIC_CFG.label_noise_std, rng)
    y_real = generate_labels(X, W_star, REAL_CFG.label_noise_std, rng)
    y_true = X @ W_star

    noise_synth = np.std(y_synth - y_true)
    noise_real = np.std(y_real - y_true)

    # Real noise should be ~6x higher
    assert noise_real / noise_synth > 4.0, \
        f"Expected real noise >> synth noise, got ratio {noise_real/noise_synth:.1f}"


def test_effective_rank():
    """Effective rank should be near d for broad distribution."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, D_MODEL))
    er = effective_rank(X)
    # Should be close to D_MODEL for Gaussian data
    assert er > D_MODEL * 0.9, f"Expected eff_rank > {D_MODEL*0.9}, got {er}"


def test_lora_learns():
    """LoRA should reduce error compared to zero initialization."""
    rng = np.random.default_rng(42)
    U = rng.standard_normal((D_MODEL, RANK))
    V = rng.standard_normal((RANK, D_MODEL))
    W_star = U @ V * 0.1

    X = rng.standard_normal((500, D_MODEL))
    y = X @ W_star + rng.standard_normal(X.shape) * 0.05

    A, B = train_lora(X, y, D_MODEL, RANK, 300, 0.01, rng)
    q = quality(A, B, X, W_star)

    # Should achieve some positive quality
    assert q > 0.0, f"Expected quality > 0, got {q}"


def test_contamination_stats():
    """Contamination stats should be consistent."""
    stats = contamination_stats(0.10, 1000, 164)
    assert stats["expected_overlap"] == 100.0
    assert stats["p_any"] >= 0.99
    assert stats["boost_pct"] > 0


def test_orthogonality_computable():
    """Should compute cosine between expert pairs."""
    rng = np.random.default_rng(42)
    experts = []
    for _ in range(3):
        A = rng.standard_normal((RANK, D_MODEL)) * 0.01
        B = rng.standard_normal((D_MODEL, RANK)) * 0.01
        experts.append((A, B))

    cos = mean_cos(experts)
    assert 0 <= cos <= 1, f"Expected cos in [0,1], got {cos}"


if __name__ == "__main__":
    tests = [test_mode_centers, test_generate_inputs_shape,
             test_label_noise_levels, test_effective_rank,
             test_lora_learns, test_contamination_stats,
             test_orthogonality_computable]

    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            sys.exit(1)

    print(f"\nAll {len(tests)} tests passed.")

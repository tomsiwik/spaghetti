#!/usr/bin/env python3
"""Tests for quality degradation detection experiment."""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def test_cosine_sim():
    """Test cosine similarity computation."""
    from experiments.models.quality_degradation_detection.quality_degradation_detection import cosine_sim

    # Identical vectors
    a = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_sim(a, a) - 1.0) < 1e-6

    # Orthogonal vectors
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_sim(a, b)) < 1e-6

    # Opposite vectors
    c = np.array([-1.0, 0.0, 0.0])
    assert abs(cosine_sim(a, c) - (-1.0)) < 1e-6

    # Zero vector
    z = np.array([0.0, 0.0, 0.0])
    assert cosine_sim(a, z) == 0.0


def test_model_init_and_forward():
    """Test model initialization and forward pass."""
    from experiments.models.quality_degradation_detection.quality_degradation_detection import (
        init_model, forward, CharTokenizer
    )

    tok = CharTokenizer()
    params = init_model(tok.vocab_size, d=16, H=2, L=1, max_T=8, seed=42)

    # Forward pass on dummy input
    inp = np.array([[2, 3, 4, 5]], dtype=np.int32)
    logits = forward(params, inp, tok.pad_id)

    assert logits.shape == (1, 4, tok.vocab_size)
    assert not np.any(np.isnan(logits))


def test_delta_operations():
    """Test compute_delta and apply_delta."""
    from experiments.models.quality_degradation_detection.quality_degradation_detection import (
        init_model, compute_delta, apply_delta, apply_deltas, flatten_delta
    )

    params1 = init_model(32, d=8, H=2, L=1, max_T=4, seed=42)
    params2 = init_model(32, d=8, H=2, L=1, max_T=4, seed=99)

    delta = compute_delta(params1, params2)

    # apply_delta should recover params2
    recovered = apply_delta(params1, delta)
    for k in params1:
        if k == '_config':
            continue
        np.testing.assert_allclose(recovered[k], params2[k], atol=1e-6)

    # apply_deltas with single delta should equal apply_delta
    recovered2 = apply_deltas(params1, [delta])
    for k in params1:
        if k == '_config':
            continue
        np.testing.assert_allclose(recovered2[k], params2[k], atol=1e-6)

    # flatten_delta should be a 1D vector
    flat = flatten_delta(delta)
    assert flat.ndim == 1
    assert flat.shape[0] > 0


def test_data_generators():
    """Test that all domain data generators produce valid data."""
    from experiments.models.quality_degradation_detection.quality_degradation_detection import (
        DOMAIN_GENERATORS, CharTokenizer
    )

    tok = CharTokenizer()
    rng = np.random.RandomState(42)

    for name, gen in DOMAIN_GENERATORS.items():
        data = gen(10, rng)
        assert len(data) == 10, f"Generator {name} produced {len(data)} samples, expected 10"
        for s in data:
            encoded = tok.encode(s)
            assert len(encoded) > 1, f"Generator {name} produced empty encoding for '{s}'"
            assert all(0 <= idx < tok.vocab_size for idx in encoded), \
                f"Generator {name} produced out-of-vocab token"


def test_run_experiment_fast():
    """Smoke test: run experiment with minimal config."""
    from experiments.models.quality_degradation_detection.quality_degradation_detection import (
        run_experiment, CONFIGS
    )

    # Override fast config for even faster test
    CONFIGS['smoke'] = {
        'd': 16, 'H': 2, 'L': 1, 'max_T': 16, 'rank': 4,
        'n_experts': 3, 'n_train': 50, 'n_test': 20,
        'epochs_base': 3, 'epochs_expert': 3, 'batch': 16,
        'lr_base': 0.002, 'lr_expert': 0.002,
    }

    # Temporarily reduce seeds
    import experiments.models.quality_degradation_detection.quality_degradation_detection as mod
    original_seeds = mod.N_SEEDS
    mod.N_SEEDS = 1

    try:
        result = run_experiment('smoke')
        assert 'kill_criteria' in result
        assert result['kill_criteria']['k2_kill'] == False  # Time should be fast
        assert result['total_pairs'] > 0
    finally:
        mod.N_SEEDS = original_seeds


if __name__ == '__main__':
    test_cosine_sim()
    print("PASS: test_cosine_sim")

    test_model_init_and_forward()
    print("PASS: test_model_init_and_forward")

    test_delta_operations()
    print("PASS: test_delta_operations")

    test_data_generators()
    print("PASS: test_data_generators")

    test_run_experiment_fast()
    print("PASS: test_run_experiment_fast")

    print("\nAll tests passed!")

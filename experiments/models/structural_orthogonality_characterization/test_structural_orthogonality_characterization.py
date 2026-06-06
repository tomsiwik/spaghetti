#!/usr/bin/env python3
"""Tests for the structural orthogonality characterization experiment."""

import numpy as np
import pytest

from experiments.models.structural_orthogonality_characterization.structural_orthogonality_characterization import (
    random_vector_expected_cos,
    random_subspace_expected_cos,
    concentration_tail_bound,
    grassmann_packing_bound,
    predict_d_crit,
    MicroMLP,
    init_lora,
    lora_to_delta_vector,
    random_lora_delta_vector,
    train_lora,
    generate_domain_data,
    cosine_sim,
    bootstrap_power_law_ci,
)


class TestTheory:
    def test_random_cos_decreases_with_d(self):
        vals = [random_vector_expected_cos(d) for d in [64, 256, 1024]]
        assert vals[0] > vals[1] > vals[2]

    def test_random_cos_matches_formula(self):
        d = 1000
        expected = np.sqrt(2.0 / (np.pi * d))
        assert abs(random_vector_expected_cos(d) - expected) < 1e-10

    def test_subspace_bound_decreases_with_d(self):
        vals = [random_subspace_expected_cos(d, 8) for d in [64, 256, 1024]]
        assert vals[0] > vals[1] > vals[2]

    def test_subspace_bound_formula(self):
        assert abs(random_subspace_expected_cos(64, 8) - np.sqrt(8/64)) < 1e-10

    def test_tail_bound_decreases_with_d(self):
        vals = [concentration_tail_bound(d, 0.1) for d in [64, 256, 1024]]
        assert vals[0] > vals[1] > vals[2]

    def test_packing_bound(self):
        assert grassmann_packing_bound(64, 8) == 64.0
        assert grassmann_packing_bound(1024, 8) == 16384.0

    def test_d_crit_formula(self):
        assert predict_d_crit(8, 0.01) == 80000.0
        assert predict_d_crit(16, 0.01) == 160000.0


class TestModel:
    def test_model_forward(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng, 64, 256, 4, 32)
        x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        logits, _ = model.forward(x)
        assert logits.shape == (1, 32)

    def test_lora_forward(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng, 64, 256, 4, 32)
        lora = init_lora(rng, 64, 256, 4, 8)
        x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        logits, _ = model.forward(x, lora, 8, 8)
        assert logits.shape == (1, 32)

    def test_compute_loss(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng, 64, 256, 4, 32)
        x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        y = np.array([5])
        loss = model.compute_loss(x, y)
        assert loss > 0

    def test_delta_vector_shape(self):
        rng = np.random.RandomState(42)
        lora = init_lora(rng, 64, 256, 4, 8)
        v = lora_to_delta_vector(lora, 4)
        assert v.shape == (4 * 2 * 64 * 256,)
        assert np.allclose(v, 0)

    def test_random_delta_nonzero(self):
        rng = np.random.RandomState(42)
        v = random_lora_delta_vector(rng, 64, 256, 4, 8)
        assert v.shape == (4 * 2 * 64 * 256,)
        assert np.linalg.norm(v) > 0

    def test_cosine_sim(self):
        assert abs(cosine_sim(np.array([1, 0]), np.array([0, 1]))) < 1e-10
        assert abs(cosine_sim(np.array([1, 0]), np.array([1, 0])) - 1.0) < 1e-10


class TestTraining:
    def test_training_returns_loss(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng, 64, 256, 2, 32)
        x, y = generate_domain_data(rng, 32, 16, 0, 2, n_sequences=100)
        lora, final_loss = train_lora(model, x, y, rng, rank=8, alpha=8,
                                       steps=50, lr=0.01, batch_size=32)
        assert isinstance(final_loss, float)
        assert final_loss > 0
        v = lora_to_delta_vector(lora, 2)
        assert np.linalg.norm(v) > 0


class TestBootstrap:
    def test_bootstrap_ci_basic(self):
        d_values = [64, 128, 256]
        per_d = {
            64: [0.01, 0.008, 0.012, 0.009, 0.011],
            128: [0.005, 0.004, 0.006, 0.0045, 0.0055],
            256: [0.002, 0.0018, 0.0025, 0.002, 0.0022],
        }
        result = bootstrap_power_law_ci(d_values, per_d, n_bootstrap=500)
        assert 'ci_lo' in result
        assert 'ci_hi' in result
        assert result['ci_lo'] < result['mean'] < result['ci_hi']
        # Should be a negative exponent
        assert result['mean'] < 0

    def test_bootstrap_ci_includes_flag(self):
        d_values = [64, 128, 256]
        per_d = {
            64: [0.01] * 5,
            128: [0.005] * 5,
            256: [0.0025] * 5,
        }
        result = bootstrap_power_law_ci(d_values, per_d, n_bootstrap=500)
        assert isinstance(result['includes_minus_half'], bool)


class TestSmoke:
    """Quick smoke test of the full experiment at tiny scale."""

    def test_experiment_runs(self):
        from experiments.models.structural_orthogonality_characterization.structural_orthogonality_characterization import run_experiment
        result = run_experiment(
            d_values=[64, 128],
            rank=8,
            n_pairs=2,
            n_seeds=1,
            n_random_pairs=10,
            train_steps={64: 50, 128: 50},
            tau=0.01,
        )
        assert 'kill_criteria_dimensional' in result
        assert 'convergence_diagnostics' in result
        assert 'scaling' in result
        assert 'gradient_bootstrap_ci' in result['scaling']
        assert '64' in result['per_d_results']
        assert '128' in result['per_d_results']
        # Check convergence diagnostics present
        assert '64' in result['convergence_diagnostics']
        assert 'mean_adapter_loss' in result['convergence_diagnostics']['64']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3
"""Tests for collision scaling experiment."""

import numpy as np
import pytest

from experiments.models.collision_scaling.experiment import (
    build_domain_config,
    compute_collision_rates,
    compute_pairwise_cosines,
    cosine_sim,
    fit_growth_models,
    MicroMLP,
    init_lora,
    lora_to_delta_vector,
    train_lora,
    generate_cluster_prototypes,
    generate_domain_data,
    D_MODEL, D_FF, N_LAYERS, LORA_RANK,
)


class TestDomainConfig:
    """Test domain configuration for various N values."""

    def test_n5_one_cluster(self):
        clusters, domains, d2c = build_domain_config(5)
        assert len(domains) == 5
        assert len(clusters) == 1

    def test_n10_two_clusters(self):
        clusters, domains, d2c = build_domain_config(10)
        assert len(domains) == 10
        assert len(clusters) == 2

    def test_n20_four_clusters(self):
        clusters, domains, d2c = build_domain_config(20)
        assert len(domains) == 20
        assert len(clusters) == 4

    def test_n50_ten_clusters(self):
        clusters, domains, d2c = build_domain_config(50)
        assert len(domains) == 50
        assert len(clusters) == 10

    def test_all_domains_have_cluster(self):
        for n in [5, 10, 15, 20, 30, 50]:
            _, domains, d2c = build_domain_config(n)
            for d in domains:
                assert d in d2c, f"Domain {d} has no cluster at N={n}"

    def test_unique_domains(self):
        for n in [5, 10, 15, 20, 30, 50]:
            _, domains, _ = build_domain_config(n)
            assert len(set(domains)) == n, f"Duplicate domains at N={n}"


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        v = np.random.randn(100)
        assert abs(cosine_sim(v, v) - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        v1 = np.array([1, 0, 0, 0])
        v2 = np.array([0, 1, 0, 0])
        assert abs(cosine_sim(v1, v2)) < 1e-10

    def test_opposite_vectors(self):
        v = np.random.randn(100)
        assert abs(cosine_sim(v, -v) + 1.0) < 1e-10

    def test_zero_vector(self):
        v = np.random.randn(100)
        assert cosine_sim(v, np.zeros(100)) == 0.0


class TestCollisionRates:
    """Test collision rate computation."""

    def test_no_collisions(self):
        cosines = {
            'all_cos': [0.01, 0.02, 0.05, 0.08],
            'within_cos': [0.05, 0.08],
            'cross_cos': [0.01, 0.02],
        }
        rates = compute_collision_rates(cosines, threshold=0.1)
        assert rates['total_collision_rate'] == 0.0
        assert rates['within_collision_rate'] == 0.0
        assert rates['cross_collision_rate'] == 0.0

    def test_all_collisions(self):
        cosines = {
            'all_cos': [0.5, 0.3, 0.2, 0.15],
            'within_cos': [0.5, 0.3],
            'cross_cos': [0.2, 0.15],
        }
        rates = compute_collision_rates(cosines, threshold=0.1)
        assert rates['total_collision_rate'] == 1.0
        assert rates['within_collision_rate'] == 1.0
        assert rates['cross_collision_rate'] == 1.0

    def test_partial_collisions(self):
        cosines = {
            'all_cos': [0.5, 0.01, 0.2, 0.05],
            'within_cos': [0.5, 0.01],
            'cross_cos': [0.2, 0.05],
        }
        rates = compute_collision_rates(cosines, threshold=0.1)
        assert rates['total_collision_rate'] == 0.5
        assert rates['n_collisions'] == 2


class TestGrowthModels:
    """Test growth model fitting."""

    def test_linear_data(self):
        n_values = [5, 10, 15, 20, 25]
        rates = [0.05, 0.10, 0.15, 0.20, 0.25]
        fits = fit_growth_models(n_values, rates)
        assert fits['linear']['r2'] > 0.99

    def test_decreasing_data(self):
        n_values = [5, 10, 20, 50]
        rates = [0.10, 0.05, 0.025, 0.01]
        fits = fit_growth_models(n_values, rates)
        assert fits['power_law']['beta'] < 0, "Decreasing data should have negative beta"

    def test_zero_data(self):
        n_values = [5, 10, 15]
        rates = [0.0, 0.0, 0.0]
        fits = fit_growth_models(n_values, rates)
        # All zeros -> insufficient data for log fit (can't take log of 0)
        assert fits['best_model'] in ('constant', 'insufficient_data')

    def test_superlinear_detection(self):
        n_values = [5, 10, 20, 50]
        rates = [0.01, 0.05, 0.25, 1.5]  # quadratic-ish growth
        fits = fit_growth_models(n_values, rates)
        assert fits['power_law']['beta'] > 1.0, "Should detect superlinear growth"
        assert fits['superlinear_test']['power_law_beta_gt_1']


class TestModelMechanics:
    """Test that model, LoRA, and data generation work correctly."""

    def test_model_forward(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng)
        x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        logits, _, _ = model.forward(x)
        assert logits.shape == (1, 32)

    def test_lora_delta_dimension(self):
        rng = np.random.RandomState(42)
        lora = init_lora(rng)
        v = lora_to_delta_vector(lora)
        expected_dim = N_LAYERS * 2 * D_MODEL * D_FF
        assert v.shape == (expected_dim,), f"Expected {expected_dim}, got {v.shape}"

    def test_lora_init_b_zero(self):
        rng = np.random.RandomState(42)
        lora = init_lora(rng)
        for l in range(N_LAYERS):
            assert np.allclose(lora['B1'][l], 0), "B1 should be initialized to zero"
            assert np.allclose(lora['B2'][l], 0), "B2 should be initialized to zero"

    def test_lora_training_changes_b(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng)
        prototypes = generate_cluster_prototypes(rng, 1)
        x, y = generate_domain_data(rng, prototypes, 'python', 'code',
                                    ['python'], n_sequences=50)
        lora = train_lora(model, x, y, rng, steps=10, lr=0.01)
        v = lora_to_delta_vector(lora)
        assert np.linalg.norm(v) > 0, "Training should produce non-zero deltas"

    def test_different_domains_different_vectors(self):
        rng = np.random.RandomState(42)
        model = MicroMLP(rng)
        prototypes = generate_cluster_prototypes(rng, 2)

        rng1 = np.random.RandomState(100)
        x1, y1 = generate_domain_data(rng1, prototypes, 'python', 'code',
                                      ['python', 'math'], n_sequences=100)
        lora1 = train_lora(model, x1, y1, rng1, steps=50, lr=0.01)
        v1 = lora_to_delta_vector(lora1)

        rng2 = np.random.RandomState(200)
        x2, y2 = generate_domain_data(rng2, prototypes, 'math', 'reasoning',
                                      ['python', 'math'], n_sequences=100)
        lora2 = train_lora(model, x2, y2, rng2, steps=50, lr=0.01)
        v2 = lora_to_delta_vector(lora2)

        cos = abs(cosine_sim(v1, v2))
        assert cos < 0.99, f"Different domains should produce different vectors (cos={cos:.4f})"


class TestPairwiseCosines:
    """Test pairwise cosine computation with cluster decomposition."""

    def test_pair_count(self):
        n = 10
        vectors = {f'd{i}': np.random.randn(100) for i in range(n)}
        d2c = {f'd{i}': 'c0' if i < 5 else 'c1' for i in range(n)}
        domains = [f'd{i}' for i in range(n)]

        result = compute_pairwise_cosines(vectors, d2c, domains)
        assert result['n_pairs'] == n * (n-1) // 2
        assert result['n_within'] + result['n_cross'] == result['n_pairs']
        assert result['n_within'] == 2 * 10  # C(5,2) * 2 clusters = 20

    def test_all_within_single_cluster(self):
        n = 5
        vectors = {f'd{i}': np.random.randn(100) for i in range(n)}
        d2c = {f'd{i}': 'c0' for i in range(n)}
        domains = [f'd{i}' for i in range(n)]

        result = compute_pairwise_cosines(vectors, d2c, domains)
        assert result['n_cross'] == 0
        assert result['n_within'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

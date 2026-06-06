"""Tests for adaptive rank selection experiment (v2, post-review fixes)."""

import numpy as np
import pytest


def test_effective_rank_identity():
    """Effective rank of a rank-k matrix should be close to k."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        effective_rank_rv
    )
    sigmas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    eff = effective_rank_rv(sigmas)
    assert abs(eff - 5.0) < 0.01, f"Expected ~5.0, got {eff}"


def test_effective_rank_one():
    """Single dominant singular value -> effective rank ~1."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        effective_rank_rv
    )
    sigmas = np.array([10.0, 0.001, 0.001, 0.001])
    eff = effective_rank_rv(sigmas)
    assert eff < 1.5, f"Expected ~1.0, got {eff}"


def test_stable_rank():
    """Stable rank of identity-like spectrum = n."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        stable_rank
    )
    sigmas = np.ones(8)
    sr = stable_rank(sigmas)
    assert abs(sr - 8.0) < 0.01


def test_energy_rank():
    """Energy rank should capture the right number of components."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        energy_rank
    )
    sigmas = np.array([10.0, 9.0, 8.0, 7.0, 0.01, 0.01, 0.01, 0.01])
    e95 = energy_rank(sigmas, 0.95)
    assert e95 <= 4, f"Expected <=4, got {e95}"


def test_snr_control():
    """Generated domains should have approximately the requested SNR."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        generate_exact_rank_domain
    )
    rng = np.random.default_rng(42)
    target_snr = 20.0
    Delta = generate_exact_rank_domain(64, 64, 8, target_snr, rng)

    U, s, Vt = np.linalg.svd(Delta, full_matrices=False)
    signal = U[:, :8] @ np.diag(s[:8]) @ Vt[:8, :]
    noise = Delta - signal
    actual_snr = np.linalg.norm(signal, 'fro') / np.linalg.norm(noise, 'fro')
    assert actual_snr > target_snr * 0.5, f"SNR {actual_snr} too low"


def test_threshold_detector_exact_rank():
    """Threshold detector should find optimal rank near true rank for exact-rank domains."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        find_optimal_rank_threshold
    )
    # Simulate error curve for exact-rank-8 matrix: sharp drop at r=8
    ranks = np.arange(1, 65)
    errors = np.zeros(64)
    for i, r in enumerate(ranks):
        if r < 8:
            errors[i] = 0.5 * (1 - r / 8)  # linearly decreasing above threshold
        else:
            errors[i] = 0.01 * (64 - r) / 56  # below threshold

    # Error at rank 7 should be > 0.05, rank 8 should be < 0.05
    errors[6] = 0.06  # rank 7
    errors[7] = 0.01  # rank 8

    opt = find_optimal_rank_threshold(ranks, errors, threshold=0.05)
    assert opt == 8, f"Expected optimal rank 8, got {opt}"


def test_kneedle_basic():
    """Kneedle should detect knee for a clear L-shaped curve."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        find_optimal_rank_kneedle
    )
    ranks = np.arange(1, 21)
    # L-shaped: steep drop then flat
    errors = np.concatenate([
        np.linspace(1.0, 0.1, 5),   # steep from rank 1-5
        np.linspace(0.09, 0.01, 15)  # flat from rank 6-20
    ])
    knee = find_optimal_rank_kneedle(ranks, errors)
    # Knee should be near the transition (rank 4-6)
    assert 3 <= knee <= 7, f"Expected knee near 5, got {knee}"


def test_kneedle_nonuniform_grid():
    """Kneedle should handle non-uniform rank grids (the v1 bug)."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        find_optimal_rank_kneedle
    )
    ranks = np.array([1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    # Error curve for a rank-8 matrix
    errors = np.array([0.9, 0.7, 0.5, 0.05, 0.03, 0.02, 0.01, 0.008, 0.005, 0.003])
    knee = find_optimal_rank_kneedle(ranks, errors)
    # Should detect knee around rank 4-8, NOT at rank 48 like v1
    assert knee <= 16, f"Kneedle picked rank {knee}, expected <= 16"


def test_experiment_runs():
    """Smoke test: experiment runs without error at small scale."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        run_experiment, Config
    )
    cfg = Config(d_in=32, d_out=32, n_seeds=2,
                 lora_ranks=(2, 4, 8, 16, 32),
                 domain_true_ranks=(2, 4, 8, 16),
                 decay_rates=(0.5, 0.9))
    results = run_experiment(cfg)
    assert 'kill_criteria' in results
    assert 'correlations_per_domain' in results
    assert 'null_baseline' in results
    assert 'threshold_vs_kneedle' in results


def test_null_baseline_reported():
    """Null baseline must be computed and reported."""
    from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
        run_experiment, Config
    )
    cfg = Config(d_in=32, d_out=32, n_seeds=2,
                 lora_ranks=(2, 4, 8, 16, 32),
                 domain_true_ranks=(2, 4, 8, 16),
                 decay_rates=(0.5, 0.9))
    results = run_experiment(cfg)
    assert 'null_baseline' in results
    assert 'within_2x_domain' in results['null_baseline']
    assert 0 <= results['null_baseline']['within_2x_domain'] <= 1


if __name__ == '__main__':
    test_effective_rank_identity()
    test_effective_rank_one()
    test_stable_rank()
    test_energy_rank()
    test_snr_control()
    test_threshold_detector_exact_rank()
    test_kneedle_basic()
    test_kneedle_nonuniform_grid()
    test_experiment_runs()
    test_null_baseline_reported()
    print("All tests passed!")

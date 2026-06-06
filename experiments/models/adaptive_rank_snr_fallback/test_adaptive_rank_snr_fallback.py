"""Tests for SNR-aware rank predictor with fallback heuristic."""

import numpy as np
import pytest


def test_compound_r95_no_fallback():
    """When r_99/r_95 <= 2.0, compound returns r_99."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        compound_r95_predictor
    )
    metrics = {'energy_rank_99': 10.0, 'energy_rank_95': 8.0}
    result = compound_r95_predictor(metrics, ratio_threshold=2.0)
    assert result == 10.0, f"Expected r_99=10, got {result}"


def test_compound_r95_with_fallback():
    """When r_99/r_95 > 2.0, compound falls back to r_95."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        compound_r95_predictor
    )
    metrics = {'energy_rank_99': 30.0, 'energy_rank_95': 10.0}
    result = compound_r95_predictor(metrics, ratio_threshold=2.0)
    assert result == 10.0, f"Expected r_95=10, got {result}"


def test_compound_r95_boundary():
    """At exactly 2.0 ratio, should NOT trigger fallback."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        compound_r95_predictor
    )
    metrics = {'energy_rank_99': 20.0, 'energy_rank_95': 10.0}
    result = compound_r95_predictor(metrics, ratio_threshold=2.0)
    assert result == 20.0, f"Expected r_99=20, got {result}"


def test_compound_effrank_no_fallback():
    """When r_99/eff_rank <= 2.0, compound returns r_99."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        compound_effrank_predictor
    )
    metrics = {'energy_rank_99': 10.0, 'effective_rank': 7.0}
    result = compound_effrank_predictor(metrics, ratio_threshold=2.0)
    assert result == 10.0


def test_compound_effrank_with_fallback():
    """When r_99/eff_rank > 2.0, compound falls back to eff_rank."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        compound_effrank_predictor
    )
    metrics = {'energy_rank_99': 50.0, 'effective_rank': 8.0}
    result = compound_effrank_predictor(metrics, ratio_threshold=2.0)
    assert result == 8.0


def test_experiment_runs():
    """Smoke test: experiment runs at small scale."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        run_full_experiment, FallbackConfig
    )
    cfg = FallbackConfig(
        dimensions=(32,),
        snr_values=(5.0, 20.0),
        n_seeds=2,
        domain_true_ranks_base=(2, 4, 8, 16),
        decay_rates=(0.5, 0.9),
        ratio_thresholds=(2.0,),
    )
    results = run_full_experiment(cfg)
    assert 'kill_criteria' in results
    assert 'conditions' in results
    assert len(results['conditions']) == 2  # 1 dim x 2 SNR


def test_kill_criteria_structure():
    """Kill criteria output has required fields."""
    from experiments.models.adaptive_rank_snr_fallback.adaptive_rank_snr_fallback import (
        run_full_experiment, FallbackConfig
    )
    cfg = FallbackConfig(
        dimensions=(32,),
        snr_values=(5.0, 20.0),
        n_seeds=2,
        domain_true_ranks_base=(2, 4, 8, 16),
        decay_rates=(0.5, 0.9),
        ratio_thresholds=(2.0,),
    )
    results = run_full_experiment(cfg)
    kc = results['kill_criteria']
    assert 'K1_pass' in kc
    assert 'K2_pass' in kc
    assert 'overall' in kc
    assert bool(kc['K1_pass']) in (True, False)
    assert bool(kc['K2_pass']) in (True, False)


if __name__ == '__main__':
    test_compound_r95_no_fallback()
    test_compound_r95_with_fallback()
    test_compound_r95_boundary()
    test_compound_effrank_no_fallback()
    test_compound_effrank_with_fallback()
    test_experiment_runs()
    test_kill_criteria_structure()
    print("All tests passed!")

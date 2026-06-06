"""Smoke tests for coverage vs noise disentangle experiment."""

import numpy as np
from coverage_vs_noise_disentangle import (
    make_conditions, decompose_effects, run_seed,
    make_mode_centers, generate_inputs, generate_labels,
    train_lora, quality, D_MODEL, RANK
)


def test_conditions_are_complete():
    """All 4 cells of the 2x2 design exist."""
    conds = make_conditions()
    assert len(conds) == 4
    expected = {"low_cov_low_noise", "low_cov_high_noise",
                "high_cov_low_noise", "high_cov_high_noise"}
    assert set(conds.keys()) == expected


def test_conditions_match_design():
    """Each condition has the correct factor levels."""
    conds = make_conditions()
    assert conds["low_cov_low_noise"].n_modes == 5
    assert conds["low_cov_low_noise"].label_noise_std == 0.05
    assert conds["low_cov_high_noise"].n_modes == 5
    assert conds["low_cov_high_noise"].label_noise_std == 0.30
    assert conds["high_cov_low_noise"].n_modes == 20
    assert conds["high_cov_low_noise"].label_noise_std == 0.05
    assert conds["high_cov_high_noise"].n_modes == 20
    assert conds["high_cov_high_noise"].label_noise_std == 0.30


def test_decompose_sums_to_100():
    """Variance explained percentages must sum to 100%."""
    cell_means = {
        "low_cov_low_noise": 0.025,
        "low_cov_high_noise": 0.023,
        "high_cov_low_noise": 0.060,
        "high_cov_high_noise": 0.058,
    }
    d = decompose_effects(cell_means)
    total = (d["variance_explained_pct"]["coverage"] +
             d["variance_explained_pct"]["noise"] +
             d["variance_explained_pct"]["interaction"])
    assert abs(total - 100.0) < 0.1, f"Total = {total}, should be ~100"


def test_decompose_known_values():
    """Test ANOVA decomposition with known symmetric case."""
    # Pure coverage effect, no noise, no interaction
    cell_means = {
        "low_cov_low_noise": 0.02,
        "low_cov_high_noise": 0.02,
        "high_cov_low_noise": 0.06,
        "high_cov_high_noise": 0.06,
    }
    d = decompose_effects(cell_means)
    assert d["variance_explained_pct"]["coverage"] > 99.0
    assert d["variance_explained_pct"]["noise"] < 1.0
    assert d["variance_explained_pct"]["interaction"] < 1.0
    assert abs(d["main_effects"]["coverage"] - 0.04) < 0.001
    assert abs(d["main_effects"]["noise"]) < 0.001


def test_single_seed_runs():
    """run_seed produces valid output."""
    result = run_seed(42)
    assert "conditions" in result
    assert len(result["conditions"]) == 4
    for key, val in result["conditions"].items():
        assert 0 <= val["quality_mean"] <= 1
        assert val["effective_rank_mean"] > 0


def test_coverage_effect_direction():
    """Verify that high coverage produces higher quality (basic sanity)."""
    result = run_seed(42)
    q_low_cov = (result["conditions"]["low_cov_low_noise"]["quality_mean"] +
                 result["conditions"]["low_cov_high_noise"]["quality_mean"]) / 2
    q_high_cov = (result["conditions"]["high_cov_low_noise"]["quality_mean"] +
                  result["conditions"]["high_cov_high_noise"]["quality_mean"]) / 2
    assert q_high_cov > q_low_cov, (
        f"High coverage ({q_high_cov:.4f}) should beat low ({q_low_cov:.4f})")


if __name__ == "__main__":
    test_conditions_are_complete()
    print("PASS: test_conditions_are_complete")
    test_conditions_match_design()
    print("PASS: test_conditions_match_design")
    test_decompose_sums_to_100()
    print("PASS: test_decompose_sums_to_100")
    test_decompose_known_values()
    print("PASS: test_decompose_known_values")
    test_single_seed_runs()
    print("PASS: test_single_seed_runs")
    test_coverage_effect_direction()
    print("PASS: test_coverage_effect_direction")
    print("\nAll tests passed.")

"""Tests for Shannon Channel Capacity Bound experiment (REVISED).

Test 1: Channel model math correctness
Test 2: R^2 computation correctness
Test 3: Model fitting produces valid parameters
Test 4: Baseline comparison (linear, power-law)
Test 5: Sensitivity analysis
Test 6: Full experiment run (collects new data + validates)
"""

import math
import numpy as np

from .channel_capacity_bound import (
    ChannelModel,
    compute_r_squared,
    compute_aic,
    compute_bic,
    fit_channel_model_analytical,
    fit_linear,
    fit_power_law,
    predict_gaps_shannon,
    predict_linear,
    predict_power_law,
    sensitivity_analysis,
    run_experiment,
    EMPIRICAL_GAPS_TRAIN,
    D_MODEL,
)


def test_channel_model_math():
    """Verify channel model equations are correct."""
    model = ChannelModel(
        d=64, P=10.0, sigma2=1.0,
        rho_mean=0.1, calibration_gain=0.0,
    )

    # N=1: no interference, SNR = P/sigma^2 = 10
    assert model.snr_single() == 10.0
    assert model.snr_effective(1) == 10.0
    assert model.composition_gap_theory(1) == 0.0

    # N=2: interference = 1 * rho^2 * P = 0.01 * 10 = 0.1
    # SNR_eff = 10 / (1 + 0.1) = 9.09
    snr_2 = model.snr_effective(2)
    assert abs(snr_2 - 10.0 / 1.1) < 1e-6, f"SNR at N=2: {snr_2}"

    # Capacity should decrease with N
    c1 = model.capacity_per_expert(1)
    c2 = model.capacity_per_expert(2)
    c5 = model.capacity_per_expert(5)
    c10 = model.capacity_per_expert(10)
    assert c1 > c2 > c5 > c10, "Capacity not monotonically decreasing"

    # Gap should increase with N
    g2 = model.composition_gap_theory(2)
    g5 = model.composition_gap_theory(5)
    g10 = model.composition_gap_theory(10)
    assert 0 < g2 < g5 < g10, "Gap not monotonically increasing"

    # Verify no rho_at_n method exists (dead code removed)
    assert not hasattr(model, 'rho_at_n'), "rho_at_n should be removed (Fix #6)"

    print(f"  N=1: C={c1:.4f}, gap=0.0%")
    print(f"  N=2: C={c2:.4f}, gap={g2:.2f}%")
    print(f"  N=5: C={c5:.4f}, gap={g5:.2f}%")
    print(f"  N=10: C={c10:.4f}, gap={g10:.2f}%")
    print("  All channel model equations verified.")
    print("  Dead code (rho_at_n) confirmed removed.")


def test_r_squared():
    """Verify R^2 computation."""
    assert compute_r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    obs = [1.0, 2.0, 3.0]
    pred = [2.0, 2.0, 2.0]
    r2 = compute_r_squared(obs, pred)
    assert abs(r2 - 0.0) < 1e-10

    obs = [1.0, 2.0, 3.0]
    pred = [1.1, 1.9, 3.1]
    r2 = compute_r_squared(obs, pred)
    assert r2 > 0.9
    print(f"  R^2 computation verified (perfect=1.0, good={r2:.4f})")


def test_model_fit():
    """Verify model fitting produces reasonable parameters."""
    model, offset, mse = fit_channel_model_analytical(EMPIRICAL_GAPS_TRAIN)

    assert model.P > 0, f"SNR_0 should be positive, got {model.P}"
    assert 0 <= model.rho_mean <= 1.0, f"rho should be in [0,1], got {model.rho_mean}"

    alpha = model.rho_mean**2 * model.P
    assert alpha > 0, f"alpha should be positive, got {alpha}"
    assert mse < 1.0, f"Fit MSE should be < 1.0, got {mse}"

    print(f"  SNR_0 = {model.P:.4f}")
    print(f"  rho   = {model.rho_mean:.6f}")
    print(f"  alpha = {alpha:.6f}")
    print(f"  offset = {offset:+.4f}%")
    print(f"  MSE   = {mse:.6f}")


def test_baseline_comparison():
    """Verify baseline model fitting works."""
    lin_params, lin_mse, _ = fit_linear(EMPIRICAL_GAPS_TRAIN)
    assert "a" in lin_params and "b" in lin_params
    assert lin_mse >= 0

    pow_params, pow_mse, _ = fit_power_law(EMPIRICAL_GAPS_TRAIN)
    assert "a" in pow_params and "b" in pow_params and "c" in pow_params
    assert pow_mse >= 0

    # AIC/BIC should be computable
    aicc_lin = compute_aic(3, 2, lin_mse)
    aicc_pow = compute_aic(3, 3, pow_mse)
    bic_lin = compute_bic(3, 2, lin_mse)
    bic_pow = compute_bic(3, 3, pow_mse)

    print(f"  Linear: a={lin_params['a']:.4f}, b={lin_params['b']:.4f}, MSE={lin_mse:.6f}")
    print(f"  Power:  a={pow_params['a']:.4f}, b={pow_params['b']:.4f}, c={pow_params['c']:.4f}, MSE={pow_mse:.6f}")
    print(f"  AICc:   linear={aicc_lin:.2f}, power={aicc_pow:.2f}")
    print(f"  BIC:    linear={bic_lin:.2f}, power={bic_pow:.2f}")


def test_sensitivity():
    """Verify sensitivity analysis runs correctly."""
    sens = sensitivity_analysis(EMPIRICAL_GAPS_TRAIN, perturbation_pct=1.0)

    assert "n_max_min" in sens
    assert "n_max_max" in sens
    assert sens["n_max_min"] > 0
    assert sens["n_max_max"] >= sens["n_max_min"]
    assert sens["n_perturbations"] == 27  # 3^3

    print(f"  N_max range: [{sens['n_max_min']}, {sens['n_max_max']}]")
    print(f"  Perturbations tested: {sens['n_perturbations']}")
    spread = sens['n_max_max'] - sens['n_max_min']
    robust = "ROBUST" if spread <= 5 else "NOT ROBUST"
    print(f"  Spread = {spread} ({robust})")


def test_full_experiment():
    """Run the full revised experiment with data collection."""
    results = run_experiment(collect_new_data=True)

    # Check structure
    assert "revision" in results
    assert results["revision"] == "v2_held_out_validation"
    assert "validation_data" in results
    assert "baseline_comparison" in results
    assert "sensitivity" in results

    # Check kill criteria
    assert results["kill_criteria"]["status"] == "killed"

    # Check validation data was collected
    val_data = results["validation_data"]
    print(f"\n  Validation data collected: {val_data}")
    print(f"  Train R^2: {results['fit_quality']['train_r2']:.4f}")
    print(f"  Val R^2:   {results['fit_quality']['val_r2']}")
    print(f"  All R^2:   {results['fit_quality']['all_r2']:.4f}")
    print(f"  Verdict:   {results['kill_criteria']['verdict']}")
    print(f"  Status:    {results['kill_criteria']['status']}")

    # Report baseline comparison
    bc = results["baseline_comparison"]
    print(f"\n  Baseline comparison (val MSE):")
    print(f"    Shannon:   {bc['shannon']['val_mse']}")
    print(f"    Linear:    {bc['linear']['val_mse']}")
    print(f"    Power-law: {bc['power_law']['val_mse']}")


if __name__ == "__main__":
    print("=== test_channel_model_math ===")
    test_channel_model_math()
    print("\n=== test_r_squared ===")
    test_r_squared()
    print("\n=== test_model_fit ===")
    test_model_fit()
    print("\n=== test_baseline_comparison ===")
    test_baseline_comparison()
    print("\n=== test_sensitivity ===")
    test_sensitivity()
    print("\n=== test_full_experiment ===")
    test_full_experiment()

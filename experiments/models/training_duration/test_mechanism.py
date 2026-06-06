"""Unit tests for training duration experiment mechanism.

Validates that:
1. TrainingDurationGPT is registered and instantiable
2. Death profiling works at different step counts
3. Monotonicity checker detects violations correctly
4. Exponential curve fitting produces reasonable parameters
"""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model, MODEL_REGISTRY
from .test_training_duration import (
    check_monotonicity, fit_exponential,
)


def test_registration():
    """TrainingDurationGPT is registered with correct parent."""
    assert "training_duration" in MODEL_REGISTRY
    assert MODEL_REGISTRY["training_duration"]["parent"] == "pruning_controls"


def test_instantiation():
    """Model instantiates and produces correct output shape."""
    model = get_model("training_duration", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())
    tokens = mx.zeros((2, 16), dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (2, 16, 28)


def test_monotonicity_checker_positive():
    """Monotonicity checker accepts monotonically increasing data."""
    data = [
        {"steps": 0, "death_rate": 0.10},
        {"steps": 100, "death_rate": 0.30},
        {"steps": 200, "death_rate": 0.50},
        {"steps": 400, "death_rate": 0.55},
    ]
    is_mono, violations = check_monotonicity(data)
    assert is_mono
    assert len(violations) == 0


def test_monotonicity_checker_negative():
    """Monotonicity checker detects decreases > 0.5pp tolerance."""
    data = [
        {"steps": 0, "death_rate": 0.10},
        {"steps": 100, "death_rate": 0.50},
        {"steps": 200, "death_rate": 0.40},  # decrease > 0.5pp
        {"steps": 400, "death_rate": 0.55},
    ]
    is_mono, violations = check_monotonicity(data)
    assert not is_mono
    assert len(violations) == 1
    assert violations[0][0] == 100  # from step 100
    assert violations[0][1] == 200  # to step 200


def test_monotonicity_checker_tolerance():
    """Small decreases within 0.5pp tolerance are accepted."""
    data = [
        {"steps": 0, "death_rate": 0.50},
        {"steps": 100, "death_rate": 0.504},  # tiny decrease from noise
        {"steps": 200, "death_rate": 0.500},  # within tolerance
    ]
    is_mono, violations = check_monotonicity(data)
    assert is_mono


def test_exponential_fit():
    """Exponential fit recovers approximately correct parameters from synthetic data."""
    import math
    # Synthetic data: delta_0=0.1, delta_inf=0.5, tau=100
    synthetic = []
    for S in [0, 50, 100, 200, 400, 800, 1600, 3200]:
        dr = 0.1 + 0.5 * (1 - math.exp(-S / 100)) if S > 0 else 0.1
        synthetic.append({"steps": S, "death_rate": dr})

    params = fit_exponential(synthetic)
    assert abs(params["delta_0"] - 0.1) < 0.05, f"delta_0={params['delta_0']}"
    assert abs(params["asymptotic_death"] - 0.6) < 0.1, f"asymptotic={params['asymptotic_death']}"
    assert params["r_squared"] > 0.95, f"R2={params['r_squared']}"


if __name__ == "__main__":
    test_registration()
    print("PASS: test_registration")
    test_instantiation()
    print("PASS: test_instantiation")
    test_monotonicity_checker_positive()
    print("PASS: test_monotonicity_checker_positive")
    test_monotonicity_checker_negative()
    print("PASS: test_monotonicity_checker_negative")
    test_monotonicity_checker_tolerance()
    print("PASS: test_monotonicity_checker_tolerance")
    test_exponential_fit()
    print("PASS: test_exponential_fit")
    print("\nAll mechanism tests passed.")

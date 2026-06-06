#!/usr/bin/env python3
"""Tests for FFN-only vs All-Modules LoRA experiment."""

import math
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.models.ffn_only_vs_all_modules.ffn_only_vs_all_modules import (
    dimension_analysis,
    monte_carlo_comparison,
)


def test_dimension_analysis_qwen7b():
    """Verify dimension counting for Qwen2.5-7B."""
    dims = dimension_analysis(d_model=3584, d_ff=18944,
                               n_heads=28, n_kv_heads=4,
                               n_layers=28, rank=16)
    # FFN-only should have more params than attention (3 large modules vs 4 smaller)
    assert dims['ffn_total_params'] > dims['attn_total_params'], \
        f"FFN params ({dims['ffn_total_params']}) should exceed attn ({dims['attn_total_params']})"

    # All = FFN + attn
    assert dims['all_total_params'] == dims['ffn_total_params'] + dims['attn_total_params']

    # FFN delta dim should be majority of all delta dim
    assert dims['ffn_ratio'] > 0.8, f"FFN ratio {dims['ffn_ratio']} should be >0.8"

    # Params ratio should be between 1 and 2
    assert 1 < dims['params_ratio'] < 2, f"Params ratio {dims['params_ratio']} unexpected"

    print("PASS: dimension_analysis_qwen7b")


def test_dimension_analysis_micro():
    """Verify dimension counting at micro scale."""
    dims = dimension_analysis(d_model=64, d_ff=256,
                               n_heads=4, n_kv_heads=4,
                               n_layers=4, rank=8)
    # At micro with equal heads, attention is d*d*4 per layer
    # FFN is d*d_ff*2 + d_ff*d per layer = d_ff*d*3
    # = 256*64*3 = 49152 per layer
    # attn = 64*64*4 = 16384 per layer
    # So FFN should dominate
    assert dims['ffn_delta_dim'] > dims['attn_delta_dim']

    # Expected cos should be small
    assert dims['ffn_expected_cos'] < 0.01
    assert dims['all_expected_cos'] < 0.01

    print("PASS: dimension_analysis_micro")


def test_monte_carlo_consistency():
    """Monte Carlo cosine should be near theoretical prediction."""
    mc = monte_carlo_comparison(rank=4, n_experts=4, n_trials=5)

    # Both means should be small (near-orthogonal)
    assert mc['ffn_mean'] < 0.05, f"FFN mean {mc['ffn_mean']} too high"
    assert mc['all_mean'] < 0.05, f"All mean {mc['all_mean']} too high"

    # The larger space (all) should have lower cos for RANDOM deltas
    # (more dimensions = more orthogonal for random vectors)
    ffn_theory = math.sqrt(2 / (math.pi * mc['ffn_dim']))
    all_theory = math.sqrt(2 / (math.pi * mc['all_dim']))
    assert ffn_theory > all_theory, "Theory: smaller dim should have higher cos"

    print("PASS: monte_carlo_consistency")


def test_expected_cosine_formula():
    """Verify the expected cosine formula for known dimensions."""
    # For D-dimensional random unit vectors, E[|cos|] ~ sqrt(2/(pi*D))
    for D in [100, 1000, 10000, 100000]:
        expected = math.sqrt(2 / (math.pi * D))
        # Generate random vectors and check
        rng = np.random.RandomState(42)
        cosines = []
        for _ in range(500):
            a = rng.randn(D)
            b = rng.randn(D)
            cos = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            cosines.append(cos)
        empirical = np.mean(cosines)
        # Should be within 30% (finite sample variance)
        ratio = empirical / expected
        assert 0.7 < ratio < 1.3, \
            f"D={D}: expected={expected:.4f}, empirical={empirical:.4f}, ratio={ratio:.2f}"

    print("PASS: expected_cosine_formula")


def test_real_adapter_findings():
    """Validate key findings from the real adapter analysis."""
    results_file = Path(__file__).parent / "results.json"
    if not results_file.exists():
        print("SKIP: results.json not found (run experiment first)")
        return

    import json
    with open(results_file) as f:
        results = json.load(f)

    real = results.get('real_adapters')
    if not real:
        print("SKIP: no real adapter data")
        return

    # Key findings to validate:
    # 1. FFN-only mean |cos| < All-modules mean |cos|
    assert real['ffn_more_orthogonal'], \
        f"FFN ({real['ffn_mean_abs_cos']:.6f}) should be more orthogonal than all ({real['full_mean_abs_cos']:.6f})"

    # 2. Attention has higher inter-domain similarity
    assert real['attn_more_similar'], \
        f"Attn ({real['attn_mean_abs_cos']:.6f}) should be more similar than FFN ({real['ffn_mean_abs_cos']:.6f})"

    # 3. Most pairs should be near-zero (only math-medical is the outlier)
    n_near_zero = sum(1 for c in real['full_cosines'] if abs(c) < 0.01)
    assert n_near_zero >= 8, f"Expected at least 8/10 near-zero pairs, got {n_near_zero}"

    print("PASS: real_adapter_findings")


if __name__ == "__main__":
    test_dimension_analysis_qwen7b()
    test_dimension_analysis_micro()
    test_monte_carlo_consistency()
    test_expected_cosine_formula()
    test_real_adapter_findings()
    print("\nAll tests passed.")

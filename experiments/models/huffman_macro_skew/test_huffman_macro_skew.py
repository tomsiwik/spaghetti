"""Tests for Huffman macro routing skew analysis."""

import sys
import math
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

from experiments.models.huffman_macro_skew.run_experiment import (
    zipf_distribution, mixture_zipf, frequency_entropy, normalized_entropy,
    gini_coefficient, huffman_analysis, deepseek_v3_model, mixtral_model,
    qwen3_coder_model,
)
from experiments.models.huffman_tree.huffman_tree import (
    build_huffman_tree, get_huffman_codes, huffman_expected_depth,
)


def test_zipf_sums_to_one():
    for n in [8, 64, 256, 512]:
        for alpha in [0.5, 1.0, 1.5]:
            freqs = zipf_distribution(n, alpha)
            assert abs(sum(freqs) - 1.0) < 1e-10, f"Zipf({n}, {alpha}) doesn't sum to 1"


def test_mixture_zipf_sums_to_one():
    for uw in [0.0, 0.3, 0.5, 1.0]:
        freqs = mixture_zipf(64, 1.0, uw)
        assert abs(sum(freqs) - 1.0) < 1e-10


def test_uniform_entropy_is_maximal():
    n = 8
    freqs = [1.0 / n] * n
    h = frequency_entropy(freqs)
    assert abs(h - math.log2(n)) < 1e-10, f"Uniform entropy should be log2({n})"
    assert abs(normalized_entropy(freqs) - 1.0) < 1e-10


def test_uniform_gives_zero_reduction():
    n = 8
    freqs = [1.0 / n] * n
    result = huffman_analysis(freqs, "uniform_8")
    assert result["depth_reduction_pct"] < 0.01, "Uniform should give 0% reduction"


def test_skewed_gives_positive_reduction():
    freqs = [0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.02, 0.02]
    result = huffman_analysis(freqs, "skewed_8")
    assert result["depth_reduction_pct"] > 5.0, "Heavy skew should give >5% reduction"


def test_gini_uniform_is_near_zero():
    freqs = [1.0 / 8] * 8
    g = gini_coefficient(freqs)
    assert abs(g) < 0.01, f"Gini of uniform should be ~0, got {g}"


def test_gini_increases_with_skew():
    g_uniform = gini_coefficient([1.0 / 8] * 8)
    g_skewed = gini_coefficient(zipf_distribution(8, 1.0))
    g_very_skewed = gini_coefficient(zipf_distribution(8, 2.0))
    assert g_uniform < g_skewed < g_very_skewed


def test_huffman_respects_shannon_bound():
    """E[depth] >= H(f) for all distributions (Shannon's lower bound)."""
    for n in [8, 16, 64, 256]:
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            freqs = zipf_distribution(n, alpha)
            h = frequency_entropy(freqs)
            root = build_huffman_tree(freqs)
            codes = get_huffman_codes(root)
            ed = huffman_expected_depth(freqs, codes)
            assert ed >= h - 1e-10, (
                f"Huffman E[d]={ed:.4f} < H={h:.4f} for Zipf({n},{alpha})"
            )


def test_huffman_within_one_bit_of_entropy():
    """E[depth] < H(f) + 1 for Huffman codes."""
    for n in [8, 16, 64, 256]:
        for alpha in [0.5, 1.0, 1.5]:
            freqs = zipf_distribution(n, alpha)
            h = frequency_entropy(freqs)
            root = build_huffman_tree(freqs)
            codes = get_huffman_codes(root)
            ed = huffman_expected_depth(freqs, codes)
            assert ed < h + 1.0 + 1e-10, (
                f"Huffman E[d]={ed:.4f} >= H+1={h+1:.4f} for Zipf({n},{alpha})"
            )


def test_production_models_return_correct_sizes():
    for label, freqs in deepseek_v3_model(256).items():
        assert len(freqs) == 256, f"{label} has {len(freqs)} experts"
    for label, freqs in mixtral_model(8).items():
        assert len(freqs) == 8
    for label, freqs in qwen3_coder_model(512).items():
        assert len(freqs) == 512


def test_kill_criteria_logic():
    # Near-uniform: should be killed
    uniform_result = huffman_analysis([1.0/8]*8, "uniform")
    assert uniform_result["kill_uniform"] is True
    assert uniform_result["kill_insufficient"] is True

    # Heavy skew: should survive both
    heavy_result = huffman_analysis([0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.02, 0.02], "heavy")
    assert heavy_result["kill_uniform"] is False
    assert heavy_result["kill_insufficient"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
    print(f"\nRan {len(tests)} tests.")

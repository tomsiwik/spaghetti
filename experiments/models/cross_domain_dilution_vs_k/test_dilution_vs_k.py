#!/usr/bin/env python3
"""Tests for cross-domain dilution vs top-k experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np


def test_softmax_weights():
    """Test softmax weight computation."""
    from experiments.models.cross_domain_dilution_vs_k.cross_domain_dilution_vs_k import softmax_weights

    # Equal scores -> equal weights
    scores = {'a': 1.0, 'b': 1.0}
    w = softmax_weights(scores, temperature=1.0)
    assert abs(w['a'] - 0.5) < 1e-6
    assert abs(w['b'] - 0.5) < 1e-6

    # Unequal scores -> unequal weights
    scores = {'a': 2.0, 'b': 1.0}
    w = softmax_weights(scores, temperature=1.0)
    assert w['a'] > w['b']

    # Sharp temperature -> near-hard selection
    w_sharp = softmax_weights(scores, temperature=0.01)
    assert w_sharp['a'] > 0.99

    print("test_softmax_weights PASSED")


def test_weighted_merge():
    """Test weighted merge produces correct interpolation."""
    from experiments.models.cross_domain_dilution_vs_k.cross_domain_dilution_vs_k import (
        weighted_merge, init_model
    )
    from experiments.models.cross_domain_composition.cross_domain_composition import CharTokenizer

    tok = CharTokenizer()
    base = init_model(tok.vocab_size, d=16, H=2, L=1, max_T=16, seed=42)

    # Create simple deltas
    delta_a = {k: np.ones_like(v) * 0.1 for k, v in base.items() if k != '_config'}
    delta_b = {k: np.ones_like(v) * 0.2 for k, v in base.items() if k != '_config'}

    expert_deltas = {'a': delta_a, 'b': delta_b}

    # Equal weight merge
    w_eq = {'a': 0.5, 'b': 0.5}
    merged_eq = weighted_merge(base, expert_deltas, w_eq, ['a', 'b'])

    # All-a merge
    w_a = {'a': 1.0, 'b': 0.0}
    merged_a = weighted_merge(base, expert_deltas, w_a, ['a', 'b'])

    # Check that equal-weight interpolates between the two
    for k in base:
        if k == '_config':
            continue
        expected_eq = base[k] + 0.5 * 0.1 + 0.5 * 0.2
        assert np.allclose(merged_eq[k], expected_eq, atol=1e-6), f"Failed on {k}"

        expected_a = base[k] + 0.1
        assert np.allclose(merged_a[k], expected_a, atol=1e-6), f"Failed on {k}"

    print("test_weighted_merge PASSED")


def test_results_exist():
    """Test that results.json was generated and contains expected fields."""
    import json

    results_path = Path(__file__).parent / 'results.json'
    if not results_path.exists():
        print("test_results_exist SKIPPED (run experiment first)")
        return

    with open(results_path) as f:
        results = json.load(f)

    assert 'strategy_aggregate' in results
    assert 'kill_criteria' in results
    assert 'equal_weight' in results['strategy_aggregate']
    assert 'ppl_probe_weighted' in results['strategy_aggregate']

    # Verify kill criteria fields
    kc = results['kill_criteria']
    assert 'k1_improvement_pp' in kc
    assert 'k2_arith_reverse_best' in kc

    print("test_results_exist PASSED")


if __name__ == '__main__':
    test_softmax_weights()
    test_weighted_merge()
    test_results_exist()
    print("\nAll tests PASSED")

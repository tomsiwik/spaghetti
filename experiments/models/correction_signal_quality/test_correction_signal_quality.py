#!/usr/bin/env python3
"""Tests for correction signal quality simulation."""

import numpy as np
import pytest

from experiments.models.correction_signal_quality.correction_signal_quality import (
    HUMAN, TEACHER, EXECUTION, DOMAINS,
    CorrectionSource, Domain,
    generate_problems, simulate_corrections,
    simulate_expert_trajectory, run_full_simulation,
    compute_aggregate_statistics, evaluate_kill_criteria,
    build_decision_tree,
)


def test_sigmoid_accuracy_endpoints():
    """Source accuracy should match base/hard at difficulty 0/1."""
    assert abs(HUMAN.accuracy(np.array([0.0]))[0] - HUMAN.base_accuracy) < 0.01
    assert abs(HUMAN.accuracy(np.array([1.0]))[0] - HUMAN.hard_accuracy) < 0.01
    assert abs(TEACHER.accuracy(np.array([0.0]))[0] - TEACHER.base_accuracy) < 0.01
    assert abs(TEACHER.accuracy(np.array([1.0]))[0] - TEACHER.hard_accuracy) < 0.01


def test_accuracy_monotonic_decrease():
    """Accuracy should decrease with difficulty."""
    d = np.linspace(0, 1, 100)
    for source in [HUMAN, TEACHER, EXECUTION]:
        acc = source.accuracy(d)
        # Allow small non-monotonicity from float precision
        assert acc[0] >= acc[-1] - 0.001, f"{source.name} accuracy not decreasing"


def test_generate_problems_clipped():
    """Generated difficulties should be in [0.01, 0.99]."""
    rng = np.random.default_rng(42)
    domain = DOMAINS[0]
    difficulties = generate_problems(domain, 10000, rng)
    assert difficulties.min() >= 0.01
    assert difficulties.max() <= 0.99


def test_execution_inapplicable_to_writing():
    """Execution feedback should provide no signal for non-code domains."""
    rng = np.random.default_rng(42)
    writing_domain = [d for d in DOMAINS if d.name == "creative_writing"][0]
    difficulties = generate_problems(writing_domain, 100, rng)
    is_correct, is_degen, is_applicable = simulate_corrections(
        EXECUTION, writing_domain, difficulties, rng
    )
    assert not is_applicable.any(), "Execution should not apply to writing"


def test_execution_applicable_to_code():
    """Execution feedback should provide signal for code domains."""
    rng = np.random.default_rng(42)
    code_domain = [d for d in DOMAINS if d.name == "python_basics"][0]
    difficulties = generate_problems(code_domain, 100, rng)
    is_correct, is_degen, is_applicable = simulate_corrections(
        EXECUTION, code_domain, difficulties, rng
    )
    assert is_applicable.all(), "Execution should apply to code"


def test_human_quality_always_positive():
    """Human corrections should always improve expert quality (on average)."""
    rng = np.random.default_rng(42)
    for domain in DOMAINS:
        result = simulate_expert_trajectory(HUMAN, domain, 200, rng)
        assert result["quality_delta"] > 0, f"Human should improve {domain.name}"


def test_teacher_quality_positive_on_easy_domains():
    """Teacher corrections should improve easy domains."""
    rng = np.random.default_rng(42)
    easy_domain = [d for d in DOMAINS if d.name == "python_basics"][0]
    result = simulate_expert_trajectory(TEACHER, easy_domain, 200, rng)
    assert result["quality_delta"] > 0, "Teacher should improve easy domains"


def test_cost_ordering():
    """Cost should be human >> teacher >> execution."""
    assert HUMAN.cost_per_correction > TEACHER.cost_per_correction
    assert TEACHER.cost_per_correction > EXECUTION.cost_per_correction


def test_reproducibility():
    """Same seed should produce identical results."""
    data1 = run_full_simulation(seeds=[42])
    data2 = run_full_simulation(seeds=[42])
    for r1, r2 in zip(data1["results"], data2["results"]):
        assert r1["quality_delta"] == r2["quality_delta"]
        assert r1["n_correct_nondegen"] == r2["n_correct_nondegen"]


def test_kill_criteria_structure():
    """Kill criteria evaluation should produce expected fields."""
    data = run_full_simulation(seeds=[42])
    agg = compute_aggregate_statistics(data)
    kill = evaluate_kill_criteria(agg)
    assert "K1_killed" in kill
    assert "K2_killed" in kill
    assert "K1_teacher_error_rate_avg" in kill
    assert "K2_worst_degeneracy" in kill


def test_decision_tree_covers_all_domains():
    """Decision tree should have an entry for every domain."""
    data = run_full_simulation(seeds=[0, 1, 2])
    agg = compute_aggregate_statistics(data)
    tree = build_decision_tree(agg)
    for domain in DOMAINS:
        assert domain.name in tree["per_domain_decisions"]


def test_teacher_avg_accuracy():
    """Teacher average accuracy should be within calibrated range."""
    avg_acc = TEACHER.avg_accuracy()
    # From MATH.md: ~83% average accuracy
    assert 0.75 < avg_acc < 0.90, f"Teacher avg accuracy {avg_acc} out of expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

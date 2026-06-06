#!/usr/bin/env python3
"""Tests for correction routing sensitivity analysis."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from correction_routing_sensitivity.correction_routing_sensitivity import (
    analytical_teacher_error_rate,
    analytical_harmful_rate,
    find_k1_breakpoint_hard_acc,
    find_k1_breakpoint_harmful,
    find_k2_coverage_breakpoint,
    compute_analytical_breakpoints,
    make_sources,
    make_domains,
    BASELINE_DOMAINS,
    BASELINE_TEACHER_HARD,
)


def test_analytical_error_rate_monotone():
    """Error rate should decrease as teacher_hard_accuracy increases."""
    errors = [analytical_teacher_error_rate(h, 0.5, 0.2)
              for h in np.linspace(0.55, 0.90, 20)]
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1], \
            f"Error rate not monotone decreasing: {errors[i]:.4f} < {errors[i+1]:.4f}"


def test_analytical_error_rate_increases_with_difficulty():
    """Error rate should increase as difficulty_mean increases."""
    errors = [analytical_teacher_error_rate(0.70, dm, 0.15)
              for dm in np.linspace(0.2, 0.8, 20)]
    for i in range(len(errors) - 1):
        assert errors[i] <= errors[i + 1], \
            f"Error rate not monotone increasing with difficulty"


def test_harmful_rate_exceeds_error_rate():
    """Harmful rate (wrong + degen) should always exceed error rate alone."""
    for h in np.linspace(0.55, 0.90, 10):
        for dm in np.linspace(0.2, 0.8, 10):
            err = analytical_teacher_error_rate(h, dm, 0.15)
            harm = analytical_harmful_rate(h, 0.08, dm, 0.15)
            assert harm >= err, \
                f"Harmful ({harm:.4f}) < error ({err:.4f}) at h={h}, dm={dm}"


def test_k2_coverage_breakpoint_closed_form():
    """K2 coverage breakpoint should be exactly 1 - threshold/gamma."""
    bp = find_k2_coverage_breakpoint(gamma=0.30, threshold=0.10)
    assert abs(bp - 2/3) < 1e-10, f"Expected 0.6667, got {bp}"


def test_k1_breakpoint_consistency():
    """K1 breakpoint should be consistent: error at breakpoint ~ threshold."""
    for d in BASELINE_DOMAINS:
        bp = find_k1_breakpoint_hard_acc(d.difficulty_mean, d.difficulty_std, 0.20)
        if bp is not None:
            err = analytical_teacher_error_rate(bp, d.difficulty_mean, d.difficulty_std)
            assert abs(err - 0.20) < 0.001, \
                f"At breakpoint {bp:.4f}, error is {err:.4f}, expected ~0.20"


def test_harmful_breakpoint_exceeds_error_breakpoint():
    """Harmful breakpoint should require higher teacher accuracy than error breakpoint."""
    for d in BASELINE_DOMAINS:
        bp_err = find_k1_breakpoint_hard_acc(d.difficulty_mean, d.difficulty_std, 0.20)
        bp_harm = find_k1_breakpoint_harmful(d.difficulty_mean, d.difficulty_std, 0.08, 0.20)
        if bp_err is not None and bp_harm is not None:
            assert bp_harm > bp_err, \
                f"Harmful breakpoint ({bp_harm:.4f}) <= error breakpoint ({bp_err:.4f}) for {d.name}"


def test_make_sources_parametric():
    """make_sources should create teacher with specified hard accuracy."""
    sources = make_sources(0.75)
    teacher = [s for s in sources if s.name == "teacher_70b"][0]
    assert teacher.hard_accuracy == 0.75


def test_make_domains_perturbation():
    """make_domains should shift difficulty means correctly."""
    domains = make_domains(+0.05)
    for orig, perturbed in zip(BASELINE_DOMAINS, domains):
        expected = min(0.95, max(0.05, orig.difficulty_mean + 0.05))
        assert abs(perturbed.difficulty_mean - expected) < 1e-10


def test_baseline_error_rate_matches_parent():
    """Analytical baseline error rate should be close to parent simulation (~19.6%)."""
    errors = []
    for d in BASELINE_DOMAINS:
        errors.append(analytical_teacher_error_rate(0.70, d.difficulty_mean, d.difficulty_std))
    avg = np.mean(errors)
    # Parent reported 19.6%
    assert abs(avg - 0.184) < 0.02, f"Average error {avg:.4f} far from expected ~0.184"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

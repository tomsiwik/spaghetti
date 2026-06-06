#!/usr/bin/env python3
"""Tests for Frechet merge implementation."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from frechet_merge import (
    random_subspace, chordal_frechet_mean, geodesic_karcher_mean,
    grassmannian_log, grassmannian_exp, chordal_distance,
    naive_addition, make_delta_from_subspace, subspace_preservation,
)


def test_chordal_mean_of_identical_subspaces():
    """Chordal mean of N copies of the same subspace should return that subspace."""
    d, r, N = 64, 4, 5
    rng = np.random.RandomState(42)
    U = random_subspace(d, r, rng)
    A_list = [U.copy() for _ in range(N)]
    merged = chordal_frechet_mean(A_list, r)
    dist = chordal_distance(U, merged)
    assert dist < 1e-6, f"Distance should be ~0, got {dist}"
    print("  PASS: chordal mean of identical subspaces")


def test_geodesic_mean_of_identical_subspaces():
    """Geodesic mean of identical subspaces should return that subspace."""
    d, r, N = 64, 4, 5
    rng = np.random.RandomState(42)
    U = random_subspace(d, r, rng)
    A_list = [U.copy() for _ in range(N)]
    merged, info = geodesic_karcher_mean(A_list, r)
    dist = chordal_distance(U, merged)
    assert dist < 1e-6, f"Distance should be ~0, got {dist}"
    assert info['converged'], "Should converge immediately"
    print("  PASS: geodesic mean of identical subspaces")


def test_log_exp_roundtrip():
    """Exp(Log(Y)) should return Y (up to sign ambiguity of orthonormal frames)."""
    d, r = 64, 4
    rng = np.random.RandomState(42)
    X = random_subspace(d, r, rng)
    Y = random_subspace(d, r, rng)
    Delta = grassmannian_log(X, Y)
    Y_recovered = grassmannian_exp(X, Delta)
    dist = chordal_distance(Y, Y_recovered)
    assert dist < 1e-8, f"Roundtrip distance should be ~0, got {dist}"
    print("  PASS: Log/Exp roundtrip")


def test_chordal_beats_naive_at_n10():
    """At N=10, chordal should beat naive on subspace preservation."""
    d, r, N = 128, 8, 10
    rng = np.random.RandomState(42)
    A_list = [random_subspace(d, r, rng) for _ in range(N)]
    B_list = [rng.randn(r, d) * 0.1 for _ in range(N)]

    frames = [np.linalg.qr(A)[0][:, :r] for A in A_list]

    # Naive
    delta_naive = naive_addition(A_list, B_list, 1.0, r)
    U_naive, _, _ = np.linalg.svd(delta_naive, full_matrices=False)
    pres_naive = np.mean(subspace_preservation(U_naive[:, :r], frames, r))

    # Chordal
    merged_chordal = chordal_frechet_mean(A_list, r)
    pres_chordal = np.mean(subspace_preservation(merged_chordal, frames, r))

    assert pres_chordal > pres_naive, \
        f"Chordal ({pres_chordal:.4f}) should beat naive ({pres_naive:.4f})"
    print(f"  PASS: chordal ({pres_chordal:.4f}) > naive ({pres_naive:.4f}) at N=10")


def test_chordal_agreement_at_n2():
    """At N=2, chordal and geodesic should agree closely."""
    d, r = 128, 8
    rng = np.random.RandomState(42)
    A_list = [random_subspace(d, r, rng) for _ in range(2)]
    merged_chordal = chordal_frechet_mean(A_list, r)
    merged_geodesic, _ = geodesic_karcher_mean(A_list, r)
    dist = chordal_distance(merged_chordal, merged_geodesic)
    assert dist < 0.1, f"At N=2, chordal-geodesic distance should be small, got {dist}"
    print(f"  PASS: chordal-geodesic agreement at N=2: dist={dist:.6f}")


def test_advantage_grows_with_n():
    """The chordal-over-naive advantage should increase with N."""
    d, r = 256, 8
    advantages = []
    for N in [2, 5, 10, 25]:
        rng = np.random.RandomState(42)
        A_list = [random_subspace(d, r, rng) for _ in range(N)]
        B_list = [rng.randn(r, d) * 0.1 for _ in range(N)]
        frames = [np.linalg.qr(A)[0][:, :r] for A in A_list]

        delta_naive = naive_addition(A_list, B_list, 1.0, r)
        U_naive, _, _ = np.linalg.svd(delta_naive, full_matrices=False)
        pres_naive = np.mean(subspace_preservation(U_naive[:, :r], frames, r))

        merged = chordal_frechet_mean(A_list, r)
        pres_chordal = np.mean(subspace_preservation(merged, frames, r))

        adv = (pres_chordal - pres_naive) / pres_naive
        advantages.append(adv)

    # Advantage should be monotonically increasing (allowing small noise)
    for i in range(1, len(advantages)):
        assert advantages[i] > advantages[i-1] - 0.02, \
            f"Advantage should grow: {advantages}"
    print(f"  PASS: advantage grows with N: {[f'{a:.2%}' for a in advantages]}")


if __name__ == '__main__':
    print("Running Frechet merge tests...")
    test_chordal_mean_of_identical_subspaces()
    test_geodesic_mean_of_identical_subspaces()
    test_log_exp_roundtrip()
    test_chordal_beats_naive_at_n10()
    test_chordal_agreement_at_n2()
    test_advantage_grows_with_n()
    print("\nAll tests passed.")

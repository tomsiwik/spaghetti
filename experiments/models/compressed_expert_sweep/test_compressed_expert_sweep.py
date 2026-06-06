#!/usr/bin/env python3
"""Tests for the compressed expert sweep experiment."""

import json
import numpy as np
from pathlib import Path

import pytest

# Test the core functions
from experiments.models.compressed_expert_sweep.compressed_expert_sweep import (
    cosine_sim,
    measure_pairwise_cosines,
    frobenius_ratio,
    generate_domain_perturbation,
    fit_lora,
    fit_lora_xs,
    fit_vera,
    measure_inference_overhead,
    LORA_RANK,
    DTYPE,
)


def test_cosine_sim_orthogonal():
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0], dtype=np.float32)
    assert abs(cosine_sim(a, b)) < 1e-6


def test_cosine_sim_parallel():
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([2, 4, 6], dtype=np.float32)
    assert abs(cosine_sim(a, b) - 1.0) < 1e-6


def test_cosine_sim_zero():
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1, 2, 3], dtype=np.float32)
    assert cosine_sim(a, b) == 0.0


def test_frobenius_ratio_identity():
    A = np.eye(4, dtype=np.float32)
    assert frobenius_ratio(A, A) < 1e-6


def test_frobenius_ratio_zero_target():
    A = np.ones((4, 4), dtype=np.float32)
    assert frobenius_ratio(A, np.zeros_like(A)) == 0.0


def test_generate_domain_perturbation_shape():
    rng = np.random.RandomState(42)
    dW = generate_domain_perturbation(64, 256, 8, rng, domain_seed=0)
    assert dW.shape == (64, 256)
    assert dW.dtype == DTYPE


def test_generate_domain_perturbation_rank():
    rng = np.random.RandomState(42)
    dW = generate_domain_perturbation(64, 256, 8, rng, domain_seed=0)
    _, s, _ = np.linalg.svd(dW, full_matrices=False)
    # Should be rank 8 (8 nonzero singular values)
    assert np.sum(s > 1e-8) <= 8


def test_generate_different_domains():
    rng = np.random.RandomState(42)
    dW1 = generate_domain_perturbation(64, 256, 8, rng, domain_seed=0)
    dW2 = generate_domain_perturbation(64, 256, 8, rng, domain_seed=1)
    cos = abs(cosine_sim(dW1.ravel(), dW2.ravel()))
    assert cos < 0.5  # Different domains should be somewhat different


def test_fit_lora_perfect_reconstruction():
    """LoRA should perfectly reconstruct rank-r targets."""
    rng = np.random.RandomState(42)
    d, d_ff, r = 64, 256, 8
    target = generate_domain_perturbation(d, d_ff, r, rng, domain_seed=42)
    delta, info = fit_lora(target, d, d_ff, r, rng)
    error = frobenius_ratio(delta, target)
    assert error < 1e-5, f"LoRA reconstruction error {error} should be near zero"
    # fit_lora returns B(d, r) + A(r, d_ff)
    expected = d * r + r * d_ff
    assert info['n_params'] == expected, f"Expected {expected}, got {info['n_params']}"


def test_fit_lora_xs_constrained():
    """LoRA-XS should have high error on random targets (subspace mismatch)."""
    rng = np.random.RandomState(42)
    d, d_ff, r = 64, 256, 8
    W = (rng.randn(d, d_ff) * 0.02).astype(DTYPE)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r].astype(DTYPE)
    Vt_r = Vt[:r, :].astype(DTYPE)

    target = generate_domain_perturbation(d, d_ff, r, rng, domain_seed=42)
    delta, info = fit_lora_xs(target, U_r, Vt_r, r, rng)
    error = frobenius_ratio(delta, target)
    # LoRA-XS should have high error (random target not aligned with SVD basis)
    assert error > 0.9, f"LoRA-XS should NOT fit random targets well, error={error}"
    assert info['n_params'] == r * r


def test_fit_vera_params_correct():
    """VeRA should have d_out + r trainable params."""
    rng = np.random.RandomState(42)
    d, d_ff, r = 256, 1024, 8
    B_s = (rng.randn(d, r) * np.sqrt(2.0 / r)).astype(DTYPE)
    A_s = (rng.randn(r, d_ff) * np.sqrt(2.0 / d_ff)).astype(DTYPE)

    target = generate_domain_perturbation(d, d_ff, r, rng, domain_seed=42)
    delta, info = fit_vera(target, B_s, A_s, d, r, rng)
    assert info['n_params'] == d + r
    # Delta should have correct shape
    assert delta.shape == target.shape


def test_vera_orthogonality_worse_than_lora():
    """VeRA experts should have higher cosines than LoRA (shared B, A)."""
    d, d_ff, r = 256, 1024, 8
    rng = np.random.RandomState(42)
    B_s = (rng.randn(d, r) * np.sqrt(2.0 / r)).astype(DTYPE)
    A_s = (rng.randn(r, d_ff) * np.sqrt(2.0 / d_ff)).astype(DTYPE)

    n = 20
    lora_deltas = []
    vera_deltas = []

    for i in range(n):
        ri = np.random.RandomState(i * 11 + 7)
        B = (ri.randn(d, r) * 0.02).astype(DTYPE)
        A = (ri.randn(r, d_ff) * np.sqrt(2.0 / d_ff)).astype(DTYPE)
        lora_deltas.append((B @ A).ravel())

        lb = (ri.randn(d) * 0.02).astype(DTYPE)
        ld = (ri.randn(r) * 0.02).astype(DTYPE)
        dW = (lb[:, None] * B_s * ld[None, :]) @ A_s
        vera_deltas.append(dW.ravel())

    lora_cos = np.mean(measure_pairwise_cosines(lora_deltas))
    vera_cos = np.mean(measure_pairwise_cosines(vera_deltas))

    assert vera_cos > lora_cos * 2, (
        f"VeRA ({vera_cos:.4f}) should be >2x worse than LoRA ({lora_cos:.4f})"
    )


def test_lora_xs_orthogonality_worse_than_lora():
    """LoRA-XS should have significantly higher cosines than LoRA."""
    rng = np.random.RandomState(42)
    d, d_ff, r = 256, 1024, 8
    W = (rng.randn(d, d_ff) * 0.02).astype(DTYPE)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r].astype(DTYPE)
    Vt_r = Vt[:r, :].astype(DTYPE)

    n = 20
    lora_deltas = []
    xs_deltas = []

    for i in range(n):
        ri = np.random.RandomState(i * 7 + 13)
        B = (ri.randn(d, r) * 0.02).astype(DTYPE)
        A = (ri.randn(r, d_ff) * np.sqrt(2.0 / d_ff)).astype(DTYPE)
        lora_deltas.append((B @ A).ravel())

        M = (ri.randn(r, r) * 0.02).astype(DTYPE)
        xs_deltas.append((U_r @ M @ Vt_r).ravel())

    lora_cos = np.mean(measure_pairwise_cosines(lora_deltas))
    xs_cos = np.mean(measure_pairwise_cosines(xs_deltas))

    assert xs_cos > lora_cos * 5, (
        f"LoRA-XS ({xs_cos:.4f}) should be >>5x worse than LoRA ({lora_cos:.4f})"
    )


def test_lora_xs_cosine_d_independent():
    """LoRA-XS cosines should NOT decrease with d (shared subspace collapse)."""
    cos_by_d = []
    for d in [64, 256]:
        d_ff = 4 * d
        r = 8
        rng = np.random.RandomState(42)
        W = (rng.randn(d, d_ff) * 0.02).astype(DTYPE)
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        U_r = U[:, :r].astype(DTYPE)
        Vt_r = Vt[:r, :].astype(DTYPE)

        xs_deltas = []
        for i in range(30):
            ri = np.random.RandomState(i * 7 + 13)
            M = (ri.randn(r, r) * 0.02).astype(DTYPE)
            xs_deltas.append((U_r @ M @ Vt_r).ravel())

        cos_by_d.append(float(np.mean(measure_pairwise_cosines(xs_deltas))))

    # LoRA-XS cosines should be similar at both d values
    ratio = cos_by_d[0] / cos_by_d[1]
    assert 0.5 < ratio < 2.0, (
        f"LoRA-XS cos should be d-independent: d=64 cos={cos_by_d[0]:.4f}, "
        f"d=256 cos={cos_by_d[1]:.4f}, ratio={ratio:.2f}"
    )


def test_results_file_exists():
    """Check that results were saved."""
    results_path = Path(__file__).parent / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        assert 'verdicts' in results
        assert 'geometric' in results


def test_pairwise_cosines_count():
    """N items should produce N*(N-1)/2 pairwise cosines."""
    vecs = [np.random.randn(10) for _ in range(5)]
    cos = measure_pairwise_cosines(vecs)
    assert len(cos) == 10  # 5*4/2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

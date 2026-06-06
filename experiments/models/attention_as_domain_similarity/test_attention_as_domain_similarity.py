#!/usr/bin/env python3
"""Tests for attention_as_domain_similarity experiment."""

import numpy as np
import pytest

from experiments.models.attention_as_domain_similarity.attention_as_domain_similarity import (
    ALL_DOMAINS,
    CLUSTERS,
    CLUSTER_SIMILARITY,
    DOMAIN_TO_CLUSTER,
    MicroTransformer,
    build_ground_truth_matrix,
    compute_cosine_matrices,
    extract_upper_triangle,
    generate_cluster_prototypes,
    generate_domain_data,
    init_lora,
    lora_to_delta_vectors,
)


def test_ground_truth_matrix_properties():
    """Ground truth matrix should be symmetric, have 1s on diagonal."""
    gt = build_ground_truth_matrix()
    N = len(ALL_DOMAINS)
    assert gt.shape == (N, N)
    # Symmetric
    np.testing.assert_array_almost_equal(gt, gt.T)
    # Diagonal = 1
    np.testing.assert_array_equal(np.diag(gt), np.ones(N))
    # Within-cluster = 0.7
    for cluster, domains in CLUSTERS.items():
        for i, d1 in enumerate(domains):
            for j, d2 in enumerate(domains):
                if d1 != d2:
                    idx1 = ALL_DOMAINS.index(d1)
                    idx2 = ALL_DOMAINS.index(d2)
                    assert gt[idx1, idx2] == 0.7


def test_ground_truth_graduated():
    """Ground truth should have graduated cross-cluster similarities."""
    gt = build_ground_truth_matrix()
    # code-reasoning should be higher than code-creative
    py_idx = ALL_DOMAINS.index('python')
    math_idx = ALL_DOMAINS.index('math')
    poetry_idx = ALL_DOMAINS.index('poetry')
    assert gt[py_idx, math_idx] > gt[py_idx, poetry_idx]


def test_transformer_forward():
    """Transformer forward pass should produce correct shapes."""
    rng = np.random.RandomState(42)
    model = MicroTransformer(rng)
    lora = init_lora(rng)

    B, T = 4, 16
    x = rng.randint(0, 32, size=(B, T))
    logits, inter, h = model.forward(x, lora)

    assert logits.shape == (B, 32)  # (B, V)
    assert len(inter) == 2  # 2 layers
    assert h.shape == (B, T, 64)  # (B, T, d)


def test_lora_delta_dimensions():
    """Delta vectors should have expected dimensions."""
    rng = np.random.RandomState(42)
    lora = init_lora(rng)
    attn_d, ffn_d, full_d = lora_to_delta_vectors(lora)

    # Attention: 2 layers * 4 matrices * d*d = 2 * 4 * 64*64 = 32768
    assert attn_d.shape == (32768,)
    # FFN: 2 layers * (d*d_ff + d_ff*d) = 2 * 2 * 64*256 = 65536
    assert ffn_d.shape == (65536,)
    # Full: attn + ffn
    assert full_d.shape == (32768 + 65536,)


def test_extract_upper_triangle():
    """Should extract correct number of elements."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    upper = extract_upper_triangle(mat)
    assert len(upper) == 3  # N*(N-1)/2 = 3
    np.testing.assert_array_equal(upper, [2, 3, 6])


def test_cosine_matrix_diagonal():
    """Self-cosine should be 1.0."""
    rng = np.random.RandomState(42)
    lora1 = init_lora(rng)
    lora2 = init_lora(rng)
    # Make B non-zero
    for key in ['Bq', 'Bk', 'Bv', 'Bo', 'B1', 'B2']:
        for l in range(2):
            lora1[key][l] = rng.randn(*lora1[key][l].shape) * 0.01
            lora2[key][l] = rng.randn(*lora2[key][l].shape) * 0.01

    d1 = lora_to_delta_vectors(lora1)
    d2 = lora_to_delta_vectors(lora2)
    attn_cos, ffn_cos, full_cos = compute_cosine_matrices([d1, d2])

    np.testing.assert_almost_equal(attn_cos[0, 0], 1.0, decimal=5)
    np.testing.assert_almost_equal(ffn_cos[0, 0], 1.0, decimal=5)
    np.testing.assert_almost_equal(full_cos[0, 0], 1.0, decimal=5)


def test_data_generation():
    """Data generation should produce correct shapes."""
    rng = np.random.RandomState(42)
    prototypes = generate_cluster_prototypes(rng)
    x, y = generate_domain_data(rng, prototypes, 'python', 'code', n_sequences=50)
    assert x.shape == (50, 16)
    assert y.shape == (50,)
    assert x.min() >= 0
    assert x.max() < 32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

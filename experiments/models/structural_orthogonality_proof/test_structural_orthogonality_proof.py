#!/usr/bin/env python3
"""Tests for structural orthogonality proof experiment."""

import numpy as np
from pathlib import Path
from structural_orthogonality_proof import (
    MicroMLP, init_lora, lora_delta_vec, train_lora,
    generate_domain_data, random_subspace_cosine, cosine_sim,
    D_CONFIG, LORA_RANK,
)


def test_model_forward():
    """Model produces valid logits."""
    rng = np.random.RandomState(42)
    model = MicroMLP(64, 2, 4, rng)
    x = np.random.randint(0, 32, (4, 16))
    # Simple forward without LoRA
    h = model.wte[x].mean(axis=1)
    for l in range(model.n_layers):
        h_in = h
        z1 = h @ model.layers[l]['W1']
        a1 = np.maximum(z1, 0)
        z2 = a1 @ model.layers[l]['W2']
        h = h_in + z2
    logits = h @ model.W_out
    assert logits.shape == (4, 32), f"Expected (4, 32), got {logits.shape}"
    print("PASS: model forward")


def test_lora_init():
    """LoRA init produces correct shapes."""
    A1, B1, A2, B2 = init_lora(128, 512, 2, np.random.RandomState(42))
    assert len(A1) == 2
    assert A1[0].shape == (128, 8)
    assert B1[0].shape == (8, 512)
    assert A2[0].shape == (512, 8)
    assert B2[0].shape == (8, 128)
    print("PASS: lora init shapes")


def test_delta_vector_dim():
    """Delta vector has correct dimension."""
    d, d_ff, nl = 64, 256, 2
    rng = np.random.RandomState(42)
    A1, B1, A2, B2 = init_lora(d, d_ff, nl, rng)
    v = lora_delta_vec(A1, B1, A2, B2)
    expected_D = nl * 2 * d * d_ff
    assert v.shape[0] == expected_D, f"Expected {expected_D}, got {v.shape[0]}"
    print(f"PASS: delta vector dim = {expected_D}")


def test_domain_data():
    """Domain data has correct shape and token range."""
    x, y = generate_domain_data(0, 50)
    assert x.shape == (50, 16)
    assert y.shape == (50,)
    assert x.min() >= 0 and x.max() < 32
    assert y.min() >= 0 and y.max() < 32
    print("PASS: domain data")


def test_different_domains_different_data():
    """Different domain IDs produce different data."""
    x0, y0 = generate_domain_data(0, 100)
    x1, y1 = generate_domain_data(1, 100)
    # Token distributions should differ
    assert not np.array_equal(y0, y1), "Same targets for different domains"
    print("PASS: different domains produce different data")


def test_random_cosine_bounded():
    """Random subspace cosines should be small."""
    cosines = random_subspace_cosine(256, 128, 2, 20)
    assert all(0 <= c <= 1 for c in cosines)
    mean_cos = np.mean(cosines)
    assert mean_cos < 0.01, f"Random cos mean {mean_cos} too high"
    print(f"PASS: random cos mean = {mean_cos:.6f} < 0.01")


def test_training_produces_nonzero_delta():
    """Training produces non-zero B matrices."""
    rng = np.random.RandomState(42)
    model = MicroMLP(64, 2, 4, rng)
    x, y = generate_domain_data(0, 100)
    A1, B1, A2, B2, loss = train_lora(model, x, y, rng, steps=50, lr=0.01, batch_size=32)
    v = lora_delta_vec(A1, B1, A2, B2)
    assert np.linalg.norm(v) > 1e-6, "Delta vector is zero after training"
    print(f"PASS: delta norm = {np.linalg.norm(v):.6f}")


def test_results_file_exists():
    """Results file should exist after running."""
    p = Path(__file__).parent / 'results.json'
    if p.exists():
        import json
        with open(p) as f:
            r = json.load(f)
        assert 'aggregate' in r
        assert 'kill_criteria' in r
        print(f"PASS: results.json exists, overall={r['kill_criteria']['overall']}")
    else:
        print("SKIP: results.json not yet generated")


if __name__ == '__main__':
    test_model_forward()
    test_lora_init()
    test_delta_vector_dim()
    test_domain_data()
    test_different_domains_different_data()
    test_random_cosine_bounded()
    test_training_produces_nonzero_delta()
    test_results_file_exists()
    print("\nAll tests passed.")

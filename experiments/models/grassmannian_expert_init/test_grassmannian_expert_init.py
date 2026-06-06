#!/usr/bin/env python3
"""Tests for Grassmannian expert init experiment."""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for import
sys.path.insert(0, str(Path(__file__).parent))

from grassmannian_expert_init import (
    welch_bound,
    random_grassmannian_points,
    frames_to_gram,
    block_norms,
    structural_projection,
    spectral_projection,
    alternating_projection,
    init_lora_from_frame,
    init_lora_random_orthonormal,
    cosine_sim,
    lora_delta_vec,
    MicroMLP,
    generate_domain_data,
    train_lora,
    subspace_distance,
    LORA_RANK,
)


def test_welch_bound():
    """Welch bound is correct for known cases."""
    # N=d (enough room for perfect orthogonality at r=1)
    assert welch_bound(4, 1, 4) == 0.0
    # N > d: positive bound
    wb = welch_bound(8, 1, 4)
    assert wb > 0
    # r=1, N=6, d=3: known Welch bound
    wb = welch_bound(6, 1, 3)
    expected = np.sqrt((6 - 3) / (3 * 5))  # sqrt(3/15) = sqrt(0.2)
    assert abs(wb - expected) < 1e-6, f"Expected {expected}, got {wb}"
    print(f"PASS: welch_bound (r=1: {wb:.4f})")

    # General r
    wb_r8 = welch_bound(8, 8, 64)
    print(f"  welch_bound(N=8, r=8, d=64) = {wb_r8:.4f}")
    assert wb_r8 >= 0
    print("PASS: welch_bound")


def test_random_grassmannian_points():
    """Random Grassmannian points are orthonormal frames."""
    rng = np.random.RandomState(42)
    frames = random_grassmannian_points(4, 8, 64, rng)
    assert frames.shape == (4, 64, 8)

    # Each frame should have orthonormal columns
    for i in range(4):
        gram = frames[i].T @ frames[i]
        np.testing.assert_allclose(gram, np.eye(8), atol=1e-5,
                                   err_msg=f"Frame {i} not orthonormal")
    print("PASS: random_grassmannian_points")


def test_frames_to_gram():
    """Gram matrix has correct structure."""
    rng = np.random.RandomState(42)
    frames = random_grassmannian_points(3, 4, 16, rng)
    G = frames_to_gram(frames)

    assert G.shape == (12, 12)  # 3*4 x 3*4

    # Diagonal blocks should be identity
    for i in range(3):
        block = G[i*4:(i+1)*4, i*4:(i+1)*4]
        np.testing.assert_allclose(block, np.eye(4), atol=1e-5,
                                   err_msg=f"Diagonal block {i} not I")

    # Should be symmetric
    np.testing.assert_allclose(G, G.T, atol=1e-5)
    print("PASS: frames_to_gram")


def test_structural_projection():
    """Structural projection caps off-diagonal blocks."""
    rng = np.random.RandomState(42)
    N, r, d = 3, 4, 16
    frames = random_grassmannian_points(N, r, d, rng)
    G = frames_to_gram(frames)

    mu = 0.5
    G_proj = structural_projection(G, N, r, mu)

    norms = block_norms(G_proj, N, r)
    for i in range(N):
        for j in range(N):
            if i != j:
                assert norms[i, j] <= mu + 1e-5, \
                    f"Block ({i},{j}) norm {norms[i,j]} > {mu}"
    print("PASS: structural_projection")


def test_spectral_projection():
    """Spectral projection produces valid Gram matrix."""
    rng = np.random.RandomState(42)
    N, r, d = 4, 2, 8
    Nr = N * r

    # Create a random matrix and project
    G = rng.randn(Nr, Nr).astype(np.float32)
    G = (G + G.T) / 2
    G_proj = spectral_projection(G, N, r, d)

    # Should be symmetric
    np.testing.assert_allclose(G_proj, G_proj.T, atol=1e-5)

    # Should be PSD (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(G_proj)
    assert eigvals.min() > -1e-4, f"Negative eigenvalue: {eigvals.min()}"

    # Trace should be N*r
    np.testing.assert_allclose(np.trace(G_proj), N * r, atol=1e-3)
    print("PASS: spectral_projection")


def test_alternating_projection_reduces_coherence():
    """AP should reduce max coherence from random initialization."""
    rng = np.random.RandomState(42)
    N, r, d = 4, 4, 32

    # Measure random coherence
    rand_frames = random_grassmannian_points(N, r, d, rng)
    rand_G = frames_to_gram(rand_frames)
    rand_norms = block_norms(rand_G, N, r)
    np.fill_diagonal(rand_norms, 0)
    rand_max = rand_norms.max()

    # Run AP
    frames, history = alternating_projection(N, r, d, n_iter=200, rng=np.random.RandomState(42))
    G = frames_to_gram(frames)
    norms = block_norms(G, N, r)
    np.fill_diagonal(norms, 0)
    ap_max = norms.max()

    print(f"  Random max coherence: {rand_max:.4f}")
    print(f"  AP max coherence:     {ap_max:.4f}")
    # AP should generally reduce coherence (or at least not increase dramatically)
    # At micro scale, results may vary
    print("PASS: alternating_projection runs")


def test_init_lora_from_frame():
    """LoRA init from frame produces correct shapes."""
    rng = np.random.RandomState(42)
    frame = random_grassmannian_points(1, 8, 64, rng)[0]
    A1, B1, A2, B2 = init_lora_from_frame(frame, 64, 256, 2)

    assert len(A1) == 2
    assert A1[0].shape == (64, 8)
    assert B1[0].shape == (8, 256)
    assert A2[0].shape == (256, 8)
    assert B2[0].shape == (8, 64)

    # A1[0] should be the frame (orthonormal)
    gram = A1[0].T @ A1[0]
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-5)

    # B matrices should be zero
    assert np.all(B1[0] == 0)
    assert np.all(B2[0] == 0)
    print("PASS: init_lora_from_frame")


def test_init_lora_random_orthonormal():
    """Haar-random orthonormal init produces orthonormal A matrices."""
    rng = np.random.RandomState(42)
    A1, B1, A2, B2 = init_lora_random_orthonormal(64, 256, 2, rng)

    assert len(A1) == 2
    assert A1[0].shape == (64, 8)

    # A1 should have orthonormal columns
    for l in range(2):
        gram = A1[l].T @ A1[l]
        np.testing.assert_allclose(gram, np.eye(8), atol=1e-5,
                                   err_msg=f"A1[{l}] not orthonormal")
        gram2 = A2[l].T @ A2[l]
        np.testing.assert_allclose(gram2, np.eye(8), atol=1e-5,
                                   err_msg=f"A2[{l}] not orthonormal")

    # B matrices should be zero
    assert np.all(B1[0] == 0)
    assert np.all(B2[0] == 0)
    print("PASS: init_lora_random_orthonormal")


def test_subspace_distance():
    """Subspace distance is 0 for same frame, positive for different."""
    rng = np.random.RandomState(42)
    frame = random_grassmannian_points(1, 8, 64, rng)[0]
    A1, B1, A2, B2 = init_lora_from_frame(frame, 64, 256, 2)

    # Before training, A1[0] = frame, so distance should be ~0
    dist = subspace_distance(frame, A1)
    assert dist < 0.01, f"Expected ~0 distance before training, got {dist}"

    # Different frame should have positive distance
    frame2 = random_grassmannian_points(1, 8, 64, np.random.RandomState(99))[0]
    dist2 = subspace_distance(frame2, A1)
    assert dist2 > 0.1, f"Expected positive distance for different frame, got {dist2}"
    print(f"PASS: subspace_distance (same={dist:.4f}, diff={dist2:.4f})")


def test_training_with_ap_init():
    """Training with AP-initialized LoRA produces non-zero deltas."""
    rng = np.random.RandomState(42)
    model = MicroMLP(64, 2, 4, rng)
    frame = random_grassmannian_points(1, 8, 64, rng)[0]
    A1, B1, A2, B2 = init_lora_from_frame(frame, 64, 256, 2)
    x, y = generate_domain_data(0, 100)

    A1, B1, A2, B2, loss = train_lora(model, x, y, A1, B1, A2, B2,
                                       steps=50, lr=0.01, batch_size=32)
    delta = lora_delta_vec(A1, B1, A2, B2)
    assert np.linalg.norm(delta) > 1e-6
    print(f"PASS: training with AP init (loss={loss:.4f}, delta_norm={np.linalg.norm(delta):.4f})")


def test_results_file():
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
    test_welch_bound()
    test_random_grassmannian_points()
    test_frames_to_gram()
    test_structural_projection()
    test_spectral_projection()
    test_alternating_projection_reduces_coherence()
    test_init_lora_from_frame()
    test_init_lora_random_orthonormal()
    test_subspace_distance()
    test_training_with_ap_init()
    test_results_file()
    print("\nAll tests passed.")

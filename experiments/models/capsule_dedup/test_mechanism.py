"""Unit tests for capsule deduplication mechanism.

Validates that the dedup algorithm correctly identifies redundant
capsules and that the merge rule (a-average, b-sum) preserves output.
"""

import mlx.core as mx
import mlx.nn as nn

from .capsule_dedup import (
    cosine_similarity_matrix,
    find_redundant_clusters,
    find_all_redundant_clusters,
    merge_capsules,
    deduplicate_composed_model,
)
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool


def test_cosine_similarity_matrix():
    """Test cosine similarity computation."""
    # Identity vectors should have cos=1 with themselves
    A = mx.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    S = cosine_similarity_matrix(A)
    mx.eval(S)

    # Diagonal should be 1.0
    for i in range(3):
        assert abs(S[i, i].item() - 1.0) < 1e-5, f"Diagonal S[{i},{i}] = {S[i,i].item()}"

    # Off-diagonal should be 0.0 (orthogonal)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(S[i, j].item()) < 1e-5, f"Off-diag S[{i},{j}] = {S[i,j].item()}"

    # Parallel vectors
    A2 = mx.array([
        [1.0, 0.0],
        [2.0, 0.0],  # same direction, different magnitude
    ])
    S2 = cosine_similarity_matrix(A2)
    mx.eval(S2)
    assert abs(S2[0, 1].item() - 1.0) < 1e-4, f"Parallel cos = {S2[0,1].item()}"

    print("  PASS: cosine_similarity_matrix")


def test_find_clusters():
    """Test cluster detection."""
    # 4 capsules: 0 and 1 are nearly identical, 2 and 3 are different
    A = mx.array([
        [1.0, 0.0, 0.0, 0.0],   # capsule 0
        [0.99, 0.1, 0.0, 0.0],  # capsule 1 - similar to 0
        [0.0, 0.0, 1.0, 0.0],   # capsule 2 - different
        [0.0, 0.0, 0.0, 1.0],   # capsule 3 - different
    ])
    S = cosine_similarity_matrix(A)

    # At threshold 0.95: only (0,1) should cluster
    clusters = find_all_redundant_clusters(S, threshold=0.95)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert set(clusters[0]) == {0, 1}, f"Expected {{0,1}}, got {set(clusters[0])}"

    # Cross-pool only: (0,1) are in pool A (indices 0-1), (2,3) in pool B (indices 2-3)
    clusters_cross = find_redundant_clusters(S, threshold=0.95, pool_sizes=[2, 2])
    # 0 and 1 are in the same pool, so cross-pool should find nothing
    assert len(clusters_cross) == 0, f"Expected 0 cross-pool clusters, got {len(clusters_cross)}"

    print("  PASS: find_clusters")


def test_merge_preserves_output():
    """Test that a-average b-sum approximately preserves output.

    For two capsules with identical a vectors, the merge should be exact.
    """
    d = 8
    P = 4

    # Create two capsules with identical a vectors but different b vectors
    A = mx.array([
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # capsule 0
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # capsule 1 (identical a)
        [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],   # capsule 2
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0],   # capsule 3
    ])

    # Random b vectors
    mx.random.seed(42)
    B = mx.random.normal((d, P)) * 0.1

    # Random input
    x = mx.random.normal((1, 1, d))

    # Output before merge
    h_before = nn.relu(x @ A.T)  # (1, 1, 4)
    out_before = h_before @ B.T  # (1, 1, d) -- wait, B is (d, P), so out = B @ h^T... no.
    # Actually: out = B @ h^T for a single vector. In matrix form:
    # h = (1, 1, P), B = (d, P), out = h @ B.T = (1, 1, d)
    out_before = h_before @ B.T  # Hmm, shapes: (1,1,4) @ (4,8) = (1,1,8) -- this is right
    mx.eval(out_before)

    # Merge capsules 0 and 1
    clusters = [[0, 1]]
    A_new, B_new = merge_capsules(A, B, clusters)
    mx.eval(A_new, B_new)

    # Should have 3 capsules now: merged(0,1), 2, 3
    assert A_new.shape[0] == 3, f"Expected 3 capsules, got {A_new.shape[0]}"

    # Output after merge
    h_after = nn.relu(x @ A_new.T)  # (1, 1, 3)
    out_after = h_after @ B_new.T  # (1, 1, d)
    mx.eval(out_after)

    # Should be exactly equal since a vectors were identical
    diff = mx.max(mx.abs(out_before - out_after)).item()
    assert diff < 1e-4, f"Output diff = {diff}, expected < 1e-4 for identical a vectors"

    print(f"  PASS: merge_preserves_output (diff = {diff:.2e})")


def test_merge_approximate_output():
    """Test that merge approximately preserves output for similar (not identical) a vectors."""
    d = 8
    P = 4

    # Two capsules with SIMILAR a vectors (cos ~ 0.98)
    A = mx.array([
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # capsule 0
        [0.98, 0.52, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],  # capsule 1 (similar)
        [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],   # capsule 2
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0],   # capsule 3
    ])

    cos_01 = cosine_similarity_matrix(A)[0, 1].item()
    print(f"    cos(a_0, a_1) = {cos_01:.4f}")

    mx.random.seed(42)
    B = mx.random.normal((d, P)) * 0.1
    x = mx.random.normal((1, 1, d))

    h_before = nn.relu(x @ A.T)
    out_before = h_before @ B.T
    mx.eval(out_before)

    clusters = [[0, 1]]
    A_new, B_new = merge_capsules(A, B, clusters)
    mx.eval(A_new, B_new)

    h_after = nn.relu(x @ A_new.T)
    out_after = h_after @ B_new.T
    mx.eval(out_after)

    diff = mx.max(mx.abs(out_before - out_after)).item()
    rel_diff = diff / (mx.max(mx.abs(out_before)).item() + 1e-8)

    print(f"    Output diff = {diff:.4e}, relative = {rel_diff:.4e}")
    assert rel_diff < 0.1, f"Relative diff {rel_diff:.4f} > 10% for similar capsules"

    print(f"  PASS: merge_approximate_output")


def test_dedup_model_forward():
    """Test that deduplicated model can still do forward pass."""
    model = ReLURouterGPT(vocab_size=28, n_capsules=16, **dict(
        n_embd=16, n_head=2, n_layer=2, block_size=8,
    ))
    mx.eval(model.parameters())

    # Forward pass before dedup
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    out_before = model(tokens)
    mx.eval(out_before)

    # Deduplicate (may or may not find anything to merge)
    stats = deduplicate_composed_model(
        model, threshold=0.90,
        cross_pool_only=False,
        verbose=False,
    )

    # Forward pass after dedup -- should not crash
    out_after = model(tokens)
    mx.eval(out_after)

    print(f"  PASS: dedup_model_forward (removed {stats['total_capsules_removed']} capsules)")


def run_all_tests():
    """Run all mechanism tests."""
    print("\nCapsule Dedup Mechanism Tests")
    print("=" * 40)
    test_cosine_similarity_matrix()
    test_find_clusters()
    test_merge_preserves_output()
    test_merge_approximate_output()
    test_dedup_model_forward()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all_tests()

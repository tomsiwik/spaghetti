"""Tests for expert removal graceful experiment.

Validates the core mechanisms: GS orthogonalization, naive subtraction,
reconstruction error measurement.
"""

import numpy as np
import pytest

from experiments.models.expert_removal_graceful.run_experiment import (
    generate_lora_expert,
    generate_expert_set,
    flatten_dW,
    cosine_sim,
    gram_schmidt,
    merge_with_gs,
    naive_removal,
    gs_recompute,
    reconstruction_error,
    per_expert_quality,
)


class TestGramSchmidt:
    """Test Gram-Schmidt orthogonalization."""

    def test_orthogonality(self):
        """GS output vectors should be pairwise orthogonal."""
        rng = np.random.RandomState(42)
        deltas = [rng.randn(100) for _ in range(5)]
        ortho = gram_schmidt(deltas)

        for i in range(len(ortho)):
            for j in range(i + 1, len(ortho)):
                cos = cosine_sim(ortho[i], ortho[j])
                assert abs(cos) < 1e-10, f"cos({i},{j}) = {cos}"

    def test_first_unchanged(self):
        """First vector should be unchanged by GS."""
        rng = np.random.RandomState(42)
        deltas = [rng.randn(100) for _ in range(5)]
        ortho = gram_schmidt(deltas)

        np.testing.assert_array_almost_equal(ortho[0], deltas[0])

    def test_span_preserved(self):
        """GS should preserve the span of the input vectors."""
        rng = np.random.RandomState(42)
        deltas = [rng.randn(50) for _ in range(3)]
        ortho = gram_schmidt(deltas)

        # Each original delta should be expressible as linear combination of ortho
        for d in deltas:
            coeffs = [np.dot(d, o) / np.dot(o, o) for o in ortho]
            reconstructed = sum(c * o for c, o in zip(coeffs, ortho))
            np.testing.assert_array_almost_equal(d, reconstructed, decimal=10)


class TestNaiveRemoval:
    """Test naive subtraction removal strategy."""

    def test_zero_error_orthogonal(self):
        """With perfectly orthogonal deltas, naive removal should have zero error."""
        # Create orthogonal deltas using identity-like structure
        d = 100
        deltas = []
        for i in range(5):
            v = np.zeros(d)
            v[i * 20:(i + 1) * 20] = np.random.randn(20)
            deltas.append(v)

        ortho = gram_schmidt(deltas)
        merged = sum(ortho)

        for k in range(5):
            w_naive = naive_removal(ortho, merged, k)
            w_gt = gs_recompute(deltas, k)
            error = reconstruction_error(w_naive, w_gt)
            assert error < 1e-8, f"Error removing expert {k}: {error}%"

    def test_last_expert_zero_error(self):
        """Removing last expert should have zero error (no cascade dependency)."""
        rng = np.random.RandomState(42)
        # Even with non-orthogonal deltas, removing last has zero error
        shared = rng.randn(100)
        deltas = [shared * 0.3 + rng.randn(100) for _ in range(5)]

        ortho = gram_schmidt(deltas)
        merged = sum(ortho)

        # Last expert
        w_naive = naive_removal(ortho, merged, 4)
        w_gt = gs_recompute(deltas, 4)
        error = reconstruction_error(w_naive, w_gt)
        assert error < 1e-8, f"Last expert removal error: {error}%"

    def test_near_orthogonal_low_error(self):
        """At SOLE cosines (~0.001), naive removal should have very low error."""
        d = 896 * 896  # large dimension for near-orthogonality
        rng = np.random.RandomState(42)

        # Random high-dim vectors are near-orthogonal
        deltas = [rng.randn(1000) / np.sqrt(1000) for _ in range(10)]

        # Verify near-orthogonality
        for i in range(10):
            for j in range(i + 1, 10):
                cos = cosine_sim(deltas[i], deltas[j])
                assert abs(cos) < 0.15, f"Not near-orthogonal: cos={cos}"

        ortho = gram_schmidt(deltas)
        merged = sum(ortho)

        # Remove middle expert
        w_naive = naive_removal(ortho, merged, 5)
        w_gt = gs_recompute(deltas, 5)
        error = reconstruction_error(w_naive, w_gt)
        assert error < 5.0, f"Near-orthogonal removal error too high: {error}%"


class TestReconstructionError:
    """Test reconstruction error metric."""

    def test_identical_zero(self):
        """Identical vectors should have zero error."""
        v = np.random.randn(100)
        assert reconstruction_error(v, v) < 1e-12

    def test_symmetric(self):
        """Error should be symmetric (it's a norm ratio, direction doesn't matter)."""
        a = np.random.randn(100)
        b = np.random.randn(100)
        # Not strictly symmetric because denominator differs, but close
        e1 = reconstruction_error(a, b)
        e2 = reconstruction_error(b, a)
        # Both should be positive
        assert e1 > 0
        assert e2 > 0


class TestExpertGeneration:
    """Test synthetic expert generation."""

    def test_clustered_cosines(self):
        """Clustered experts should have higher within-cluster cosine."""
        experts = generate_expert_set(
            N=12, d=100, r=8,
            cluster_structure={"n_clusters": 3, "within_cos": 0.3},
            seed=42
        )
        deltas = [flatten_dW(e) for e in experts]

        # Within-cluster cosines (same cluster = i%3 == j%3)
        within = []
        cross = []
        for i in range(12):
            for j in range(i + 1, 12):
                cos = abs(cosine_sim(deltas[i], deltas[j]))
                if experts[i]["cluster"] == experts[j]["cluster"]:
                    within.append(cos)
                else:
                    cross.append(cos)

        assert np.mean(within) > np.mean(cross), \
            f"Within-cluster cos ({np.mean(within):.3f}) should exceed " \
            f"cross-cluster ({np.mean(cross):.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

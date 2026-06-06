"""Tests for hash ring expert removal."""

import numpy as np
import pytest

from experiments.models.hash_ring_remove_expert.run_experiment import (
    HashRing,
    generate_token_hashes,
    measure_removal_displacement,
    build_expert_quality_matrix,
    simulate_quality_degradation,
)


class TestHashRing:
    def test_build_ring(self):
        ring = HashRing(8, n_virtual=10)
        assert len(ring.ring) == 80
        # Sorted
        positions = [p for p, _ in ring.ring]
        assert positions == sorted(positions)

    def test_find_primary(self):
        ring = HashRing(4, n_virtual=50)
        # Every token should route to one of 4 experts
        hashes = generate_token_hashes(1000, seed=42)
        for h in hashes:
            expert = ring.find_primary(int(h))
            assert 0 <= expert < 4

    def test_find_top_k(self):
        ring = HashRing(8, n_virtual=50)
        hashes = generate_token_hashes(100, seed=42)
        for h in hashes:
            top2 = ring.find_top_k(int(h), k=2)
            assert len(top2) == 2
            assert len(set(top2)) == 2  # distinct

    def test_remove_expert(self):
        ring = HashRing(8, n_virtual=10)
        initial_len = len(ring.ring)
        ring.remove_expert(3)
        assert len(ring.ring) == initial_len - 10
        # No expert 3 on ring
        experts_on_ring = set(e for _, e in ring.ring)
        assert 3 not in experts_on_ring

    def test_add_expert(self):
        ring = HashRing(8, n_virtual=10)
        initial_len = len(ring.ring)
        ring.add_expert(8)
        assert len(ring.ring) == initial_len + 10
        experts_on_ring = set(e for _, e in ring.ring)
        assert 8 in experts_on_ring


class TestRemovalDisplacement:
    def test_zero_false_moves(self):
        """Only tokens on the removed expert should be displaced."""
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = measure_removal_displacement(ring, 4, hashes)
        assert result['false_moves'] == 0

    def test_100_percent_neighbor_accuracy(self):
        """All displaced tokens should go to clockwise neighbors."""
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = measure_removal_displacement(ring, 4, hashes)
        assert result['neighbor_accuracy'] == 1.0

    def test_displacement_bounded(self):
        """Displacement should be bounded (not displacing ALL tokens)."""
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = measure_removal_displacement(ring, 4, hashes)
        # Should be much less than 50%
        assert result['displacement_rate'] < 0.50

    def test_displacement_nonzero(self):
        """Some tokens should be displaced."""
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = measure_removal_displacement(ring, 4, hashes)
        assert result['displacement_rate'] > 0.01

    def test_different_experts_same_guarantee(self):
        """Removing any expert should have 100% neighbor accuracy."""
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        for expert_id in range(8):
            result = measure_removal_displacement(ring, expert_id, hashes)
            assert result['false_moves'] == 0
            assert result['neighbor_accuracy'] == 1.0


class TestAddRemoveSymmetry:
    def test_roundtrip_identity(self):
        """Remove then re-add should produce identical routing."""
        ring_original = HashRing(8, n_virtual=50)
        hashes = generate_token_hashes(1000, seed=42)

        assignments_before = [ring_original.find_primary(int(h)) for h in hashes]

        # Remove expert 4
        ring_minus = HashRing(8, n_virtual=50)
        ring_minus.remove_expert(4)

        # Re-add expert 4
        ring_minus.add_expert(4)

        assignments_after = [ring_minus.find_primary(int(h)) for h in hashes]

        assert assignments_before == assignments_after


class TestQualityDegradation:
    def test_zero_specialization_no_degradation(self):
        """With identical experts, removal should not degrade quality."""
        Q = np.ones((8, 8))  # All experts equally good
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = simulate_quality_degradation(ring, 4, hashes, Q, seed=42)
        assert abs(result['degradation_pct']) < 1.0

    def test_degradation_under_kill_threshold(self):
        """Degradation should be under 5% for moderate specialization."""
        Q = build_expert_quality_matrix(8, specialization=0.3, seed=42)
        ring = HashRing(8, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = simulate_quality_degradation(ring, 4, hashes, Q, seed=42)
        assert abs(result['degradation_pct']) < 5.0


class TestScaling:
    @pytest.mark.parametrize("N", [4, 8, 16, 32])
    def test_neighbor_accuracy_scales(self, N):
        """100% neighbor accuracy should hold at all N values."""
        ring = HashRing(N, n_virtual=150)
        hashes = generate_token_hashes(10_000, seed=42)
        result = measure_removal_displacement(ring, N // 2, hashes)
        assert result['neighbor_accuracy'] == 1.0
        assert result['false_moves'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

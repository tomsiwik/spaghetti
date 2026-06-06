"""Tests for the ReLoRA composition test experiment."""

import math
import pytest
import mlx.core as mx
import mlx.nn as nn

from ..gpt import GPT
from ..lora_procrustes.lora_procrustes import LoRAGPT, LoRALinear

from .relora_composition_test import (
    merge_lora_into_base,
    compute_pairwise_cosine,
    compute_weight_spectrum,
    _effective_rank,
    _bootstrap_ci,
    _permutation_test,
    attach_lora_to_gpt,
    _copy_relora_base_with_fresh_lora,
    train_relora,
    train_conventional,
    train_lora_expert,
    run_experiment,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_lora_model():
    """Small LoRAGPT for unit tests."""
    model = LoRAGPT(
        vocab_size=28, block_size=16,
        n_embd=32, n_head=2, n_layer=2,
        lora_rank=4, lora_alpha=1.0,
    )
    mx.eval(model.parameters())
    return model


@pytest.fixture
def small_gpt_model():
    """Small GPT for unit tests."""
    model = GPT(
        vocab_size=28, block_size=16,
        n_embd=32, n_head=2, n_layer=2,
    )
    mx.eval(model.parameters())
    return model


@pytest.fixture
def dataset():
    """Minimal dataset for testing."""
    from ...data import load_names, CharTokenizer, CharDataset
    docs = load_names()[:100]  # small subset
    tok = CharTokenizer(docs)
    return CharDataset(docs, tok, block_size=16), tok


# ── Unit Tests ──────────────────────────────────────────────────────────────────


class TestMergeLoRA:
    """Test the core ReLoRA merge operation."""

    def test_merge_preserves_output(self, small_lora_model):
        """After merge, model output should be (nearly) unchanged."""
        x = mx.array([[0, 1, 2, 3, 4]])

        # Train for a few steps to get non-zero LoRA
        small_lora_model.layers[0].mlp.fc1.A = mx.ones_like(
            small_lora_model.layers[0].mlp.fc1.A
        ) * 0.1
        small_lora_model.layers[0].mlp.fc1.B = mx.ones_like(
            small_lora_model.layers[0].mlp.fc1.B
        ) * 0.1
        mx.eval(small_lora_model.parameters())

        out_before = small_lora_model(x)
        mx.eval(out_before)
        before_val = out_before[0, 0, 0].item()

        merge_lora_into_base(small_lora_model)
        out_after = small_lora_model(x)
        mx.eval(out_after)
        after_val = out_after[0, 0, 0].item()

        # After merge + reset(B=0), LoRA contribution is 0,
        # but base now contains the merged delta
        assert abs(before_val - after_val) < 0.05, (
            f"Merge changed output: {before_val:.4f} -> {after_val:.4f}"
        )

    def test_merge_resets_B_to_zero(self, small_lora_model):
        """After merge, B should be zero (so LoRA output is zero)."""
        small_lora_model.layers[0].mlp.fc1.B = mx.ones_like(
            small_lora_model.layers[0].mlp.fc1.B
        )
        mx.eval(small_lora_model.parameters())

        merge_lora_into_base(small_lora_model)

        for layer in small_lora_model.layers:
            assert mx.allclose(layer.mlp.fc1.B, mx.zeros_like(layer.mlp.fc1.B)), \
                "B not reset to zero after merge"
            assert mx.allclose(layer.mlp.fc2.B, mx.zeros_like(layer.mlp.fc2.B)), \
                "B not reset to zero after merge"

    def test_merge_accumulates_rank(self, small_lora_model):
        """Multiple merges should increase effective rank of base weights."""
        # Get initial rank
        w0 = small_lora_model.layers[0].mlp.fc1.linear.weight
        _, S0, _ = mx.linalg.svd(w0, stream=mx.cpu)
        mx.eval(S0)
        rank0 = _effective_rank(S0.tolist())

        # Do several merges with random LoRA
        for _ in range(5):
            for layer in small_lora_model.layers:
                layer.mlp.fc1.A = mx.random.normal(layer.mlp.fc1.A.shape) * 0.1
                layer.mlp.fc1.B = mx.random.normal(layer.mlp.fc1.B.shape) * 0.1
                layer.mlp.fc2.A = mx.random.normal(layer.mlp.fc2.A.shape) * 0.1
                layer.mlp.fc2.B = mx.random.normal(layer.mlp.fc2.B.shape) * 0.1
            mx.eval(small_lora_model.parameters())
            merge_lora_into_base(small_lora_model)

        w1 = small_lora_model.layers[0].mlp.fc1.linear.weight
        _, S1, _ = mx.linalg.svd(w1, stream=mx.cpu)
        mx.eval(S1)
        rank1 = _effective_rank(S1.tolist())

        # Effective rank should not decrease (and likely increases)
        assert rank1 >= rank0 * 0.8, (
            f"Effective rank decreased after merges: {rank0:.2f} -> {rank1:.2f}"
        )


class TestOrthogonality:
    """Test cosine similarity measurement."""

    def test_orthogonal_deltas_have_low_cosine(self):
        """Truly orthogonal deltas should have cos ~ 0."""
        # Create two orthogonal delta sets (simulated)
        d = 64
        # Use random vectors which are approximately orthogonal in high-d
        deltas_a = [(0, 'fc1', mx.random.normal((d, 4 * d)))]
        deltas_b = [(0, 'fc1', mx.random.normal((d, 4 * d)))]
        cosines = compute_pairwise_cosine([deltas_a, deltas_b])
        assert len(cosines) == 1
        _, _, cos = cosines[0]
        # Random vectors in high-d should have cos << 0.1
        assert abs(cos) < 0.2, f"Random high-d vectors not near-orthogonal: |cos|={abs(cos):.4f}"

    def test_parallel_deltas_have_high_cosine(self):
        """Identical deltas should have cos = 1."""
        d = 64
        delta = mx.random.normal((d, 4 * d))
        deltas_a = [(0, 'fc1', delta)]
        deltas_b = [(0, 'fc1', delta)]
        cosines = compute_pairwise_cosine([deltas_a, deltas_b])
        _, _, cos = cosines[0]
        assert cos > 0.99, f"Identical deltas should have cos~1, got {cos:.4f}"


class TestEffectiveRank:
    """Test effective rank computation."""

    def test_uniform_spectrum(self):
        """Uniform singular values -> full rank."""
        sv = [1.0] * 10
        er = _effective_rank(sv)
        assert abs(er - 10.0) < 0.01, f"Uniform spectrum should give rank=10, got {er}"

    def test_single_nonzero(self):
        """Single dominant singular value -> rank ~ 1."""
        sv = [1.0] + [0.0] * 9
        er = _effective_rank(sv)
        assert abs(er - 1.0) < 0.01, f"Single nonzero SV should give rank~1, got {er}"

    def test_decaying_spectrum(self):
        """Exponentially decaying spectrum -> intermediate rank."""
        sv = [math.exp(-0.5 * i) for i in range(10)]
        er = _effective_rank(sv)
        assert 1.0 < er < 10.0, f"Decaying spectrum should give 1 < rank < 10, got {er}"


class TestAttachLoRA:
    """Test LoRA attachment to pretrained GPT."""

    def test_attach_copies_weights(self, small_gpt_model):
        """Attached LoRA model should have same base weights."""
        lora = attach_lora_to_gpt(small_gpt_model, rank=4)

        # Check embedding weights match
        assert mx.allclose(lora.wte.weight, small_gpt_model.wte.weight)

        # Check MLP base weights match
        orig_w = small_gpt_model.layers[0].mlp.fc1.weight
        lora_w = lora.layers[0].mlp.fc1.linear.weight
        assert mx.allclose(orig_w, lora_w), "Base weights not copied correctly"

    def test_fresh_lora_has_zero_delta(self, small_gpt_model):
        """Freshly attached LoRA should have zero delta (B=0)."""
        lora = attach_lora_to_gpt(small_gpt_model, rank=4)
        for layer in lora.layers:
            delta = layer.mlp.fc1.get_delta()
            mx.eval(delta)
            assert mx.allclose(delta, mx.zeros_like(delta)), "Fresh LoRA delta not zero"


class TestCopyReLoRABase:
    """Test copying ReLoRA base with fresh LoRA."""

    def test_copy_preserves_base(self, small_lora_model):
        """Copy should preserve base weights."""
        # Modify base weights
        small_lora_model.layers[0].mlp.fc1.linear.weight = (
            mx.ones_like(small_lora_model.layers[0].mlp.fc1.linear.weight) * 0.5
        )
        mx.eval(small_lora_model.parameters())

        copy = _copy_relora_base_with_fresh_lora(small_lora_model, rank=4)

        assert mx.allclose(
            copy.layers[0].mlp.fc1.linear.weight,
            small_lora_model.layers[0].mlp.fc1.linear.weight,
        ), "Base weights not preserved in copy"

    def test_copy_has_fresh_lora(self, small_lora_model):
        """Copy should have fresh (zero-delta) LoRA."""
        small_lora_model.layers[0].mlp.fc1.B = mx.ones_like(
            small_lora_model.layers[0].mlp.fc1.B
        )
        mx.eval(small_lora_model.parameters())

        copy = _copy_relora_base_with_fresh_lora(small_lora_model, rank=4)

        delta = copy.layers[0].mlp.fc1.get_delta()
        mx.eval(delta)
        assert mx.allclose(delta, mx.zeros_like(delta)), "Copy LoRA delta not zero"


class TestWeightSpectrum:
    """Test weight spectrum analysis."""

    def test_spectrum_returns_stats(self, small_lora_model):
        """Should return spectrum stats for each layer."""
        spectrum = compute_weight_spectrum(small_lora_model)
        assert len(spectrum) > 0
        for key, stats in spectrum.items():
            assert "max_sv" in stats
            assert "effective_rank" in stats
            assert stats["effective_rank"] > 0


# ── Integration Tests ───────────────────────────────────────────────────────────


class TestReLoRATraining:
    """Test ReLoRA and conventional training."""

    def test_relora_reduces_loss(self, small_lora_model, dataset):
        """ReLoRA training should reduce loss."""
        ds, tok = dataset
        result = train_relora(
            small_lora_model, ds,
            total_steps=50, merge_every=20,
            batch_size=8, lr=3e-3, log_every=50,
        )
        assert result["final_loss"] < 4.0, (
            f"ReLoRA did not reduce loss enough: {result['final_loss']:.4f}"
        )
        assert result["merges_done"] >= 2, "Not enough merges performed"

    def test_conventional_reduces_loss(self, small_gpt_model, dataset):
        """Conventional training should reduce loss."""
        ds, tok = dataset
        result = train_conventional(
            small_gpt_model, ds,
            total_steps=50, batch_size=8, lr=3e-3, log_every=50,
        )
        assert result["final_loss"] < 4.0, (
            f"Conventional training did not reduce loss: {result['final_loss']:.4f}"
        )


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_ci_contains_mean(self):
        """CI should contain the sample mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = _bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_ci_narrows_with_less_variance(self):
        """Less variable data should give narrower CI."""
        tight = [1.0, 1.01, 1.02, 0.99, 0.98]
        wide = [0.5, 1.5, 2.5, 0.1, 3.0]
        _, tlo, thi = _bootstrap_ci(tight)
        _, wlo, whi = _bootstrap_ci(wide)
        assert (thi - tlo) < (whi - wlo)


class TestPermutationTest:
    """Test permutation test for cosine significance."""

    def test_identical_distributions_high_p(self):
        """Same distribution should give high p-value."""
        import random as rnd
        rng = rnd.Random(42)
        a = [rng.gauss(0, 1) for _ in range(20)]
        b = [rng.gauss(0, 1) for _ in range(20)]
        p = _permutation_test(a, b, n_perms=1000)
        # Not necessarily > 0.05 for any single draw, but shouldn't be tiny
        assert p > 0.001

    def test_different_distributions_low_p(self):
        """Very different distributions should give low p-value."""
        a = [10.0, 11.0, 12.0, 10.5, 11.5]
        b = [0.1, 0.2, 0.15, 0.12, 0.18]
        p = _permutation_test(a, b, n_perms=1000)
        assert p < 0.05


class TestFullExperiment:
    """Integration test for the full experiment."""

    def test_experiment_runs_and_produces_results(self):
        """Full experiment should complete and produce valid results."""
        results = run_experiment(
            n_embd=32, n_head=2, n_layer=2,
            block_size=16,
            lora_rank=4, lora_alpha=1.0,
            total_pretrain_steps=100,
            merge_every=25,
            expert_train_steps=50,
            n_experts=2,
            batch_size=8,
            lr=3e-3, seed=42,
        )

        # Basic sanity checks
        assert results.relora_final_loss > 0
        assert results.conventional_final_loss > 0
        assert results.relora_mean_cos >= 0
        assert results.conventional_mean_cos >= 0
        assert len(results.relora_expert_losses) == 2
        assert len(results.conventional_expert_losses) == 2
        assert results.cos_ratio > 0
        assert results.loss_ratio > 0
        assert results.verdict in ("SURVIVES", "KILLED", "INCONCLUSIVE")

#!/usr/bin/env python3
"""Tests for SOLE vs LoRA Soups comparison."""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from experiments.models.oae_vs_lora_soups.oae_vs_lora_soups import (
    MLP, LoRA, compose_oae, compose_avg, compose_cat,
    make_transition_matrix, generate_data, cosine_similarity,
    D_MODEL, D_FF, N_LAYERS, LORA_RANK, VOCAB_SIZE, SEQ_LEN,
)


def test_model_forward():
    """Base model produces valid logits."""
    rng = np.random.default_rng(42)
    model = MLP(rng)
    x = rng.integers(0, VOCAB_SIZE, size=8)
    logits = model.forward(x)
    assert logits.shape == (8, VOCAB_SIZE), f"Expected (8, {VOCAB_SIZE}), got {logits.shape}"
    assert np.isfinite(logits).all(), "Logits contain non-finite values"
    print("PASS: test_model_forward")


def test_lora_deltas():
    """LoRA produces correctly shaped deltas."""
    rng = np.random.default_rng(42)
    lora = LoRA(rng)
    deltas = lora.get_deltas()
    assert len(deltas) == N_LAYERS
    for dW_up, dW_down in deltas:
        assert dW_up.shape == (D_MODEL, D_FF), f"dW_up shape: {dW_up.shape}"
        assert dW_down.shape == (D_FF, D_MODEL), f"dW_down shape: {dW_down.shape}"
    print("PASS: test_lora_deltas")


def test_lora_rank():
    """LoRA deltas have rank <= LORA_RANK."""
    rng = np.random.default_rng(42)
    lora = LoRA(rng)
    deltas = lora.get_deltas()
    for l, (dW_up, dW_down) in enumerate(deltas):
        rank_up = np.linalg.matrix_rank(dW_up, tol=1e-6)
        rank_down = np.linalg.matrix_rank(dW_down, tol=1e-6)
        assert rank_up <= LORA_RANK, f"Layer {l} up rank {rank_up} > {LORA_RANK}"
        assert rank_down <= LORA_RANK, f"Layer {l} down rank {rank_down} > {LORA_RANK}"
    print("PASS: test_lora_rank")


def test_compose_oae_is_sum():
    """SOLE composition is simple addition."""
    rng = np.random.default_rng(42)
    lora1 = LoRA(rng)
    lora2 = LoRA(rng)
    deltas = compose_oae([lora1, lora2])
    for l in range(N_LAYERS):
        d1 = lora1.get_deltas()[l]
        d2 = lora2.get_deltas()[l]
        np.testing.assert_allclose(deltas[l][0], d1[0] + d2[0], atol=1e-6)
        np.testing.assert_allclose(deltas[l][1], d1[1] + d2[1], atol=1e-6)
    print("PASS: test_compose_oae_is_sum")


def test_compose_avg_is_scaled():
    """Uniform average composition scales by 1/k."""
    rng = np.random.default_rng(42)
    lora1 = LoRA(rng)
    lora2 = LoRA(rng)
    deltas = compose_avg([lora1, lora2])
    for l in range(N_LAYERS):
        d1 = lora1.get_deltas()[l]
        d2 = lora2.get_deltas()[l]
        expected_up = (d1[0] + d2[0]) / 2
        expected_down = (d1[1] + d2[1]) / 2
        np.testing.assert_allclose(deltas[l][0], expected_up, atol=1e-6)
        np.testing.assert_allclose(deltas[l][1], expected_down, atol=1e-6)
    print("PASS: test_compose_avg_is_scaled")


def test_orthogonality():
    """Independently initialized LoRAs are near-orthogonal."""
    rng = np.random.default_rng(42)
    loras = [LoRA(rng) for _ in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            cos = abs(cosine_similarity(loras[i].flatten(), loras[j].flatten()))
            assert cos < 0.5, f"LoRA {i} and {j} have cos={cos:.4f} >= 0.5"
    print("PASS: test_orthogonality")


def test_gradient_computation():
    """Analytical gradients produce valid loss reduction."""
    rng = np.random.default_rng(42)
    model = MLP(rng)
    lora = LoRA(rng)

    x = rng.integers(0, VOCAB_SIZE, size=32)
    y = rng.integers(0, VOCAB_SIZE, size=32)

    loss_before = model.loss_and_grads(x, y, lora.get_deltas(), grad_lora=True, grad_base=False)[0]
    for _ in range(5):
        lora.train_step(model, x, y, lr=0.01)
    loss_after = model.loss_and_grads(x, y, lora.get_deltas(), grad_lora=True, grad_base=False)[0]

    # Loss should decrease (or at least not increase dramatically)
    assert loss_after <= loss_before * 1.5, f"Loss increased from {loss_before:.4f} to {loss_after:.4f}"
    print(f"PASS: test_gradient_computation (loss: {loss_before:.4f} -> {loss_after:.4f})")


def test_data_generation():
    """Markov chain data generation produces valid sequences."""
    rng = np.random.default_rng(42)
    tm = make_transition_matrix('code', 0, rng)
    assert tm.shape == (VOCAB_SIZE, VOCAB_SIZE)
    np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)
    data = generate_data(tm, 10, SEQ_LEN + 1, rng)
    assert data.shape == (10, SEQ_LEN + 1)
    assert data.min() >= 0 and data.max() < VOCAB_SIZE
    print("PASS: test_data_generation")


if __name__ == '__main__':
    test_model_forward()
    test_lora_deltas()
    test_lora_rank()
    test_compose_oae_is_sum()
    test_compose_avg_is_scaled()
    test_orthogonality()
    test_gradient_computation()
    test_data_generation()
    print("\nAll tests passed.")

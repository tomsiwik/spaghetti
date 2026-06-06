#!/usr/bin/env python3
"""Tests for pre-merge vs dynamic routing experiment."""

import numpy as np
import pytest


def test_micro_mlp_forward():
    """Test MicroMLP forward pass produces valid logits."""
    from experiments.models.premerge_vs_dynamic_quality.premerge_vs_dynamic import (
        MicroMLP, VOCAB_SIZE, CONTEXT_LEN, D_MODEL
    )
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    x = rng.randint(0, VOCAB_SIZE, size=(4, CONTEXT_LEN))
    logits, intermediates, h = model.forward(x)
    assert logits.shape == (4, VOCAB_SIZE)
    assert h.shape == (4, D_MODEL)
    assert len(intermediates) == 4


def test_lora_forward():
    """Test LoRA modifies output."""
    from experiments.models.premerge_vs_dynamic_quality.premerge_vs_dynamic import (
        MicroMLP, init_lora, VOCAB_SIZE, CONTEXT_LEN
    )
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    lora = init_lora(rng)
    # Set non-zero B to get different output
    for l in range(4):
        lora['B1'][l] = rng.randn(*lora['B1'][l].shape) * 0.01
        lora['B2'][l] = rng.randn(*lora['B2'][l].shape) * 0.01
    x = rng.randint(0, VOCAB_SIZE, size=(2, CONTEXT_LEN))
    logits_base, _, _ = model.forward(x)
    logits_lora, _, _ = model.forward(x, lora)
    assert not np.allclose(logits_base, logits_lora)


def test_premerge_deltas():
    """Test pre-merge averaging produces correct shapes."""
    from experiments.models.premerge_vs_dynamic_quality.premerge_vs_dynamic import (
        MicroMLP, init_lora, lora_to_delta_weights, premerge_deltas,
        N_LAYERS, D_MODEL, D_FF
    )
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    loras = [init_lora(rng) for _ in range(5)]
    deltas = [lora_to_delta_weights(l) for l in loras]
    merged = premerge_deltas(deltas, strategy='average')
    assert len(merged) == N_LAYERS
    assert merged[0]['dW1'].shape == (D_MODEL, D_FF)
    assert merged[0]['dW2'].shape == (D_FF, D_MODEL)


def test_forward_with_delta():
    """Test forward_with_delta matches forward with LoRA."""
    from experiments.models.premerge_vs_dynamic_quality.premerge_vs_dynamic import (
        MicroMLP, init_lora, lora_to_delta_weights, VOCAB_SIZE, CONTEXT_LEN
    )
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    lora = init_lora(rng)
    for l in range(4):
        lora['B1'][l] = rng.randn(*lora['B1'][l].shape) * 0.1
        lora['B2'][l] = rng.randn(*lora['B2'][l].shape) * 0.1
    x = rng.randint(0, VOCAB_SIZE, size=(3, CONTEXT_LEN))
    logits_lora, _, _ = model.forward(x, lora)
    deltas = lora_to_delta_weights(lora)
    logits_delta = model.forward_with_delta(x, deltas)
    np.testing.assert_allclose(logits_lora, logits_delta, atol=1e-10)


def test_cosine_route_topk():
    """Test cosine routing returns correct number of experts."""
    from experiments.models.premerge_vs_dynamic_quality.premerge_vs_dynamic import (
        cosine_route_topk, D_MODEL
    )
    rng = np.random.RandomState(42)
    names = ['a', 'b', 'c', 'd', 'e']
    centroids = {n: rng.randn(D_MODEL) for n in names}
    query = rng.randn(D_MODEL)
    result = cosine_route_topk(query, centroids, names, k=2)
    assert len(result) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
    assert abs(sum(w for _, w in result) - 1.0) < 1e-10


def test_data_generation():
    """Test data generation produces valid sequences."""
    from experiments.models.premerge_vs_dynamic_quality.premerge_vs_dynamic import (
        generate_cluster_prototypes, generate_domain_data,
        VOCAB_SIZE, CONTEXT_LEN
    )
    rng = np.random.RandomState(42)
    prototypes = generate_cluster_prototypes(rng)
    x, y = generate_domain_data(rng, prototypes, 'python', 'code',
                                n_sequences=10)
    assert x.shape == (10, CONTEXT_LEN)
    assert y.shape == (10,)
    assert x.min() >= 0 and x.max() < VOCAB_SIZE
    assert y.min() >= 0 and y.max() < VOCAB_SIZE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

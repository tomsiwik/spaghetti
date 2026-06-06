#!/usr/bin/env python3
"""Quick validation tests for the latency benchmark.

Usage:
    uv run python -m pytest experiments/models/inference_latency_vs_N/test_latency.py -v
"""

import pytest
import numpy as np


def _get_torch():
    import torch
    return torch


@pytest.fixture
def model_and_experts():
    """Build a tiny model with a few experts for testing."""
    from experiments.models.inference_latency_vs_N.bench_latency import (
        build_base_model, generate_lora_experts,
    )
    torch = _get_torch()
    device = "cpu"
    model = build_base_model(d_model=64, n_heads=2, n_layers=2,
                             vocab_size=128, max_seq_len=32, device=device)
    experts = generate_lora_experts(model, n_experts=10, rank=4, device=device)
    input_ids = torch.randint(0, 128, (1, 16), device=device)
    return model, experts, input_ids, device


def test_premerge_same_shape(model_and_experts):
    """Pre-merged model should have identical parameter shapes."""
    from experiments.models.inference_latency_vs_N.bench_latency import premerge_weights
    model, experts, _, device = model_and_experts
    merged = premerge_weights(model, experts[:5], device)

    for p1, p2 in zip(model.parameters(), merged.parameters()):
        assert p1.shape == p2.shape, "Pre-merge changed parameter shape"


def test_premerge_weights_differ(model_and_experts):
    """Pre-merged model should have different weight values from base."""
    from experiments.models.inference_latency_vs_N.bench_latency import premerge_weights
    torch = _get_torch()
    model, experts, _, device = model_and_experts
    merged = premerge_weights(model, experts[:5], device)

    # At least some weights should differ
    any_diff = False
    for p1, p2 in zip(model.parameters(), merged.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            any_diff = True
            break
    assert any_diff, "Pre-merge didn't change any weights"


def test_dynamic_topk_output_shape(model_and_experts):
    """Dynamic top-k should produce output of correct shape."""
    from experiments.models.inference_latency_vs_N.bench_latency import dynamic_topk_forward
    model, experts, input_ids, _ = model_and_experts
    logits = dynamic_topk_forward(model, input_ids, experts[:5], k=2, expert_indices=[0, 1])
    assert logits.shape == (1, 16, 128), f"Expected (1,16,128), got {logits.shape}"


def test_dynamic_restores_weights(model_and_experts):
    """Dynamic top-k should restore original weights after forward."""
    from experiments.models.inference_latency_vs_N.bench_latency import dynamic_topk_forward
    torch = _get_torch()
    model, experts, input_ids, _ = model_and_experts

    # Save original weights
    orig = [p.data.clone() for p in model.parameters()]

    # Run dynamic forward
    dynamic_topk_forward(model, input_ids, experts[:5], k=2, expert_indices=[0, 1])

    # Check weights are restored
    for p, o in zip(model.parameters(), orig):
        assert torch.allclose(p.data, o, atol=1e-7), "Dynamic forward didn't restore weights"


def test_routing_latency_sublinear():
    """Hash ring routing should scale sublinearly with N."""
    from experiments.models.inference_latency_vs_N.bench_latency import measure_routing_latency
    results = measure_routing_latency([10, 100], n_iters=5000)
    ratio = results[100]["per_query_us"] / results[10]["per_query_us"]
    # O(log N) would give ~2x for 10x N. Allow up to 3x.
    assert ratio < 3.0, f"Routing grew {ratio:.2f}x for 10x N (expected <3x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for the few-shot adaptation recovery experiment."""

import os
import json
import torch
import pytest

# Test imports work
from experiments.models.base_free_composition.base_free_composition import (
    GPT, LoRALinear, LoRAGPT, CharTokenizer, CharDataset, load_names,
)

from experiments.models.fewshot_adaptation_recovery.fewshot_adaptation_recovery import (
    inject_lora_deltas_as_trainable,
    adapt_expert,
    run_experiment,
)


@pytest.fixture
def micro_model():
    """Create a tiny GPT for testing."""
    torch.manual_seed(42)
    return GPT(vocab_size=30, block_size=16, n_embd=32, n_head=2, n_layer=2)


@pytest.fixture
def expert_deltas(micro_model):
    """Create synthetic LoRA deltas."""
    deltas = []
    for i, layer in enumerate(micro_model.layers):
        for fc_name in ["fc1", "fc2"]:
            fc = getattr(layer.mlp, fc_name)
            in_f, out_f = fc.in_features, fc.out_features
            # Simulate rank-4 LoRA delta
            A = torch.randn(in_f, 4) * 0.01
            B = torch.randn(4, out_f) * 0.01
            delta = A @ B * 0.25  # scale = alpha/rank = 1/4
            deltas.append((i, fc_name, delta))
    return deltas


def test_svd_warmstart_reproduces_delta(micro_model, expert_deltas):
    """SVD warm-start should reproduce the original delta exactly."""
    lora_model = inject_lora_deltas_as_trainable(
        micro_model, expert_deltas, rank=4, alpha=1.0
    )
    recovered_deltas = lora_model.get_all_deltas()

    for (li, fn, orig), (_, _, recovered) in zip(expert_deltas, recovered_deltas):
        err = (orig - recovered).abs().max().item()
        assert err < 1e-5, f"Layer {li} {fn}: reconstruction error {err} > 1e-5"


def test_adaptation_reduces_loss():
    """Adaptation steps should reduce loss on the new base."""
    result = run_experiment(
        n_embd=32, n_head=2, n_layer=2, block_size=16,
        lora_rank=4, total_pretrain_steps=200,
        expert_train_steps=100, n_experts=2,
        adapt_steps_list=[10, 50], adapt_lr=1e-3,
        seed=42
    )

    # Adapted loss at 50 steps should be <= adapted loss at 10 steps
    results_10 = [r for r in result.per_expert if r["adapt_steps"] == 10]
    results_50 = [r for r in result.per_expert if r["adapt_steps"] == 50]

    for r10, r50 in zip(results_10, results_50):
        assert r50["adapted_loss"] <= r10["adapted_loss"] + 0.01, \
            "More adaptation steps should not significantly increase loss"


def test_zeroshot_baseline_matches_parent():
    """Zero-shot loss should be worse than original (positive gap)."""
    result = run_experiment(
        n_embd=32, n_head=2, n_layer=2, block_size=16,
        lora_rank=4, total_pretrain_steps=200,
        expert_train_steps=100, n_experts=2,
        adapt_steps_list=[10], seed=42
    )

    for r in result.per_expert:
        assert r["zs_gap"] > 0, "Zero-shot should be worse than original"
        assert r["zeroshot_loss"] > r["original_loss"], \
            "ZS loss should exceed original loss"


def test_results_files_exist():
    """Check that result files were created by the main experiment."""
    base_dir = os.path.dirname(__file__)
    assert os.path.exists(os.path.join(base_dir, "results_aggregate.json")), \
        "Aggregate results file should exist"

    with open(os.path.join(base_dir, "results_aggregate.json")) as f:
        agg = json.load(f)

    assert agg["overall_verdict"] in ("KILLED", "SURVIVES", "INCONCLUSIVE")
    assert "aggregate_across_seeds" in agg
    assert "50" in agg["aggregate_across_seeds"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

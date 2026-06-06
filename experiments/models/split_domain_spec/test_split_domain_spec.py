"""Tests for split_domain_spec mechanism."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.models.split_domain_spec.split_domain_spec import (
    profile_leaf_activations,
    compute_domain_jaccard,
)
from experiments.models.split_leaf_actual.split_leaf_actual import split_leaf_into_tree


def test_model_registers():
    """Model registers and instantiates."""
    model = get_model("split_domain_spec", vocab_size=28, block_size=32)
    mx.eval(model.parameters())
    out = model(mx.zeros((1, 4), dtype=mx.int32))
    assert out.shape == (1, 4, 28)
    print("PASS: model registers and runs forward")


def test_profile_leaf_activations():
    """Profiling returns active capsule sets per leaf per layer."""
    mx.random.seed(42)
    model = get_model("split_domain_spec", vocab_size=28, block_size=32,
                       n_capsules_per_leaf=16)
    mx.eval(model.parameters())

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    train_docs, _ = train_val_split(docs, seed=42)
    ds = CharDataset(train_docs, tokenizer, 32)

    active = profile_leaf_activations(model, ds, [0, 1], n_batches=5, batch_size=16, seed=42)
    assert 0 in active and 1 in active
    assert len(active[0]) == 4  # 4 layers
    # At least some capsules should be active
    total_active = sum(len(v) for v in active[0].values())
    assert total_active > 0, "No active capsules found"
    print(f"PASS: profiling found {total_active} active capsule slots across layers for leaf 0")


def test_compute_domain_jaccard():
    """Jaccard computation works correctly."""
    # Perfect overlap
    active_A = {0: {0, 1, 2}, 1: {0, 1}}
    active_B = {0: {0, 1, 2}, 1: {0, 1}}
    per_layer, mean_j = compute_domain_jaccard(active_A, active_B, 2)
    assert abs(mean_j - 1.0) < 1e-6, f"Expected J=1.0 for identical sets, got {mean_j}"

    # No overlap
    active_A2 = {0: {0, 1}, 1: {0}}
    active_B2 = {0: {2, 3}, 1: {1}}
    per_layer2, mean_j2 = compute_domain_jaccard(active_A2, active_B2, 2)
    assert abs(mean_j2) < 1e-6, f"Expected J=0.0 for disjoint sets, got {mean_j2}"

    # Partial overlap
    active_A3 = {0: {0, 1, 2}}
    active_B3 = {0: {1, 2, 3}}
    per_layer3, mean_j3 = compute_domain_jaccard(active_A3, active_B3, 1)
    expected = 2.0 / 4.0  # |{1,2}| / |{0,1,2,3}|
    assert abs(mean_j3 - expected) < 1e-6, f"Expected J={expected}, got {mean_j3}"

    print("PASS: Jaccard computation correct")


def test_split_then_profile():
    """After splitting, can profile both children."""
    mx.random.seed(42)
    model = get_model("split_domain_spec", vocab_size=28, block_size=32,
                       n_capsules_per_leaf=32)
    mx.eval(model.parameters())

    split_leaf_into_tree(model, leaf_idx=0, noise_scale=0.001)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    train_docs, _ = train_val_split(docs, seed=42)
    ds = CharDataset(train_docs, tokenizer, 32)

    active = profile_leaf_activations(model, ds, [0, 1], n_batches=5, batch_size=16, seed=42)

    # After split, leaves 0 and 1 have 16 capsules each
    for li in [0, 1]:
        for l_idx in range(4):
            max_cap = max(active[li][l_idx]) if active[li][l_idx] else -1
            assert max_cap < 16, f"Leaf {li} layer {l_idx} has capsule idx {max_cap} >= 16"

    print("PASS: split children profiled with correct capsule indices")


if __name__ == "__main__":
    test_model_registers()
    test_profile_leaf_activations()
    test_compute_domain_jaccard()
    test_split_then_profile()
    print("\nAll tests passed!")

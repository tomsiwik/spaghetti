"""Smoke tests for N=5 pre-composition pruning pipeline.

Quick validation that the mechanism works before running the full
3-seed experiment.
"""

import copy

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    compose_relu_models,
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import (
    profile_activations,
    identify_dead_capsules,
    prune_model,
)
from ..prune_before_compose.prune_before_compose import compose_pruned_models


def test_n5_compose_pruned():
    """Test that compose_pruned_models works with 5 variable-sized pools."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    # Create 5 models with different capsule counts (simulating pruning)
    capsule_counts = [100, 80, 90, 70, 85]
    models = [_make_relu_model(V, n_capsules=c) for c in capsule_counts]
    base = _make_relu_model(V, n_capsules=N_CAPSULES)

    composed = compose_pruned_models(base, models)

    expected_total = sum(capsule_counts)
    for layer in composed.layers:
        assert layer.capsule_pool.n_capsules == expected_total, \
            f"Expected {expected_total}, got {layer.capsule_pool.n_capsules}"

    # Check forward pass works
    inputs = mx.ones((2, 32), dtype=mx.int32)
    logits = composed(inputs)
    assert logits.shape == (2, 32, V), f"Bad shape: {logits.shape}"
    print(f"  PASS: compose_pruned_models handles 5 variable-sized pools "
          f"({expected_total} total capsules)")


def test_n5_prune_before_compose_mechanism():
    """Test that pruning before composition produces a valid model at N=5."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="quintary")

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=42)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, all_docs_val = train_val_split(docs, seed=42)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    domain_names = list(domain_datasets.keys())
    assert len(domain_names) == 5, f"Expected 5 domains, got {len(domain_names)}"

    # Pretrain + fine-tune (short for smoke test)
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=50, batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)

    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=50,
              batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    # Pipeline A: compose then prune
    composed_A = compose_relu_models(base, [domain_models[d] for d in domain_names])
    freqs_A = profile_activations(composed_A, joint_val, n_batches=5, batch_size=16, seed=42)
    masks_A = identify_dead_capsules(freqs_A, threshold=0.0)
    prune_model(composed_A, masks_A, verbose=False)
    loss_A = evaluate(composed_A, joint_val, batch_size=16)

    # Pipeline B: prune then compose
    pruned_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(domain_models[d_name])
        freqs_d = profile_activations(
            model_d, domain_datasets[d_name][1],
            n_batches=5, batch_size=16, seed=42,
        )
        masks_d = identify_dead_capsules(freqs_d, threshold=0.0)
        prune_model(model_d, masks_d, verbose=False)
        pruned_models[d_name] = model_d

    composed_B = compose_pruned_models(base, [pruned_models[d] for d in domain_names])
    loss_B = evaluate(composed_B, joint_val, batch_size=16)

    # Both should produce finite losses
    assert loss_A < 10.0, f"Pipeline A loss too high: {loss_A}"
    assert loss_B < 10.0, f"Pipeline B loss too high: {loss_B}"

    # Print comparison
    delta = ((loss_B - loss_A) / loss_A) * 100
    print(f"  Pipeline A (compose-then-prune, N=5): {loss_A:.4f}")
    print(f"  Pipeline B (prune-then-compose, N=5): {loss_B:.4f}")
    print(f"  Delta: {delta:+.2f}%")

    # Count capsules
    caps_A = sum(layer.capsule_pool.n_capsules for layer in composed_A.layers)
    caps_B = sum(layer.capsule_pool.n_capsules for layer in composed_B.layers)
    print(f"  Pipeline A alive capsules (total across layers): {caps_A}")
    print(f"  Pipeline B alive capsules (total across layers): {caps_B}")
    print(f"  PASS: Both pipelines produce valid models with finite loss at N=5")


def main():
    """Run all mechanism tests."""
    print("\n=== Test 1: compose_pruned_models with 5 pools ===")
    test_n5_compose_pruned()

    print("\n=== Test 2: prune-before-compose mechanism at N=5 ===")
    test_n5_prune_before_compose_mechanism()


if __name__ == "__main__":
    main()

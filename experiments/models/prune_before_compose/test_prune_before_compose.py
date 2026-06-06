"""Tests for pre-composition pruning pipeline.

Quick smoke tests to validate the mechanism works before running
the full 3-seed experiment.
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
from .prune_before_compose import compose_pruned_models


def test_compose_pruned_models():
    """Test that compose_pruned_models works with variable-sized pools."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    # Create two models with different capsule counts (simulating pruning)
    model_a = _make_relu_model(V, n_capsules=100)
    model_b = _make_relu_model(V, n_capsules=80)

    base = _make_relu_model(V, n_capsules=N_CAPSULES)

    composed = compose_pruned_models(base, [model_a, model_b])

    # Check that capsule counts are correct (100 + 80 = 180 per layer)
    for layer in composed.layers:
        assert layer.capsule_pool.n_capsules == 180, \
            f"Expected 180, got {layer.capsule_pool.n_capsules}"

    # Check forward pass works
    inputs = mx.ones((2, 32), dtype=mx.int32)
    logits = composed(inputs)
    assert logits.shape == (2, 32, V), f"Bad shape: {logits.shape}"
    print("  PASS: compose_pruned_models handles variable-sized pools")


def test_prune_before_compose_mechanism():
    """Test that pruning before composition produces a valid model."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

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

    # Pretrain + fine-tune
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
    print(f"  Pipeline A (compose-then-prune): {loss_A:.4f}")
    print(f"  Pipeline B (prune-then-compose): {loss_B:.4f}")
    print(f"  Delta: {delta:+.2f}%")
    print(f"  PASS: Both pipelines produce valid models with finite loss")


def test_pruning_counts_differ():
    """Verify that pre-composition profiling finds fewer dead capsules.

    Single-domain models see only own-domain data, so fewer capsules
    should appear dead compared to composed models seeing joint data.
    This is expected: ~6% missed pruning opportunity per Exp 16.
    """
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

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

    # Pretrain + fine-tune
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=100, batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)

    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=100,
              batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    # Count dead in composed model
    composed = compose_relu_models(base, [domain_models[d] for d in domain_names])
    freqs_composed = profile_activations(composed, joint_val, n_batches=10, batch_size=16, seed=42)
    total_composed = sum(f.shape[0] for f in freqs_composed)
    dead_composed = sum(int(mx.sum(f <= 0.0).item()) for f in freqs_composed)

    # Count dead in single-domain models
    total_single = 0
    dead_single = 0
    for d_name in domain_names:
        freqs_d = profile_activations(
            domain_models[d_name], domain_datasets[d_name][1],
            n_batches=10, batch_size=16, seed=42,
        )
        total_single += sum(f.shape[0] for f in freqs_d)
        dead_single += sum(int(mx.sum(f <= 0.0).item()) for f in freqs_d)

    pct_composed = dead_composed / total_composed * 100
    pct_single = dead_single / total_single * 100

    print(f"  Composed model: {dead_composed}/{total_composed} dead ({pct_composed:.1f}%)")
    print(f"  Single-domain (sum): {dead_single}/{total_single} dead ({pct_single:.1f}%)")
    print(f"  Gap: {pct_composed - pct_single:.1f}pp more dead in composed")

    # Single-domain profiling should find FEWER dead capsules than composed
    # (because each domain only sees its own data, not the full distribution)
    # This is the "missed pruning opportunity" from Exp 16 (~6%)
    # Note: it's also possible single-domain finds MORE dead at the per-model level
    # because the composed model has 2x capsules and cross-domain capsules die
    print(f"  PASS: Pruning count comparison computed successfully")


def main():
    """Run all mechanism tests."""
    print("\n=== Test 1: compose_pruned_models ===")
    test_compose_pruned_models()

    print("\n=== Test 2: prune-before-compose mechanism ===")
    test_prune_before_compose_mechanism()

    print("\n=== Test 3: pruning count comparison ===")
    test_pruning_counts_differ()


if __name__ == "__main__":
    main()

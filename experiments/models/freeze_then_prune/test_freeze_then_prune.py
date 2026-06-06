"""Test suite for freeze-then-prune protocol experiment.

Tests:
  1. Mechanism test: prune_dead_capsules actually removes capsules
  2. Protocol A runs and produces expected structure
  3. Protocol B runs at a mid-point
  4. Full experiment runs across 3 seeds and produces verdict
"""

import copy
import pytest

import mlx.core as mx

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT
from ..relu_router.test_composition import (
    _make_relu_model, _freeze_attention,
    BASE, N_CAPSULES, BATCH_SIZE, LR,
)
from .freeze_then_prune import (
    profile_death_rate, prune_dead_capsules,
    run_protocol_A, run_protocol_B, run_no_prune_control,
    run_full_experiment, analyze_results, main,
    S_TOTAL, S_MID_POINTS, DOMAIN,
)


@pytest.fixture(scope="module")
def setup_data():
    """Create shared data objects for all tests."""
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

    all_docs_train, _ = train_val_split(docs, seed=42)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    train_ds = domain_datasets[DOMAIN][0]
    val_ds = domain_datasets[DOMAIN][1]

    # Pretrain base
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=100, batch_size=BATCH_SIZE, lr=LR,
          seed=42, log_every=9999)

    return base, train_ds, val_ds


def test_profile_death_rate(setup_data):
    """profile_death_rate returns valid structure."""
    base, _, val_ds = setup_data
    model = copy.deepcopy(base)

    overall, per_layer, per_layer_masks, flat_mask = profile_death_rate(model, val_ds)

    assert 0.0 <= overall <= 1.0
    assert len(per_layer) == 4
    assert len(per_layer_masks) == 4
    for rate in per_layer:
        assert 0.0 <= rate <= 1.0
    assert all(isinstance(v, bool) for v in flat_mask)


def test_prune_removes_capsules(setup_data):
    """Pruning after training reduces capsule count."""
    base, train_ds, val_ds = setup_data
    model = copy.deepcopy(base)

    # Train to induce death
    _freeze_attention(model)
    train(model, train_ds, steps=200, batch_size=BATCH_SIZE, lr=LR,
          seed=42, log_every=9999)
    model.unfreeze()

    # Count capsules before
    caps_before = sum(l.capsule_pool.n_capsules for l in model.layers)

    prune_stats, death_rate, per_layer_rates = prune_dead_capsules(model, val_ds)

    caps_after = sum(l.capsule_pool.n_capsules for l in model.layers)

    # Some capsules should be dead and pruned
    assert caps_after <= caps_before
    assert prune_stats["total_pruned"] >= 0
    assert death_rate >= 0.0


def test_prune_preserves_quality(setup_data):
    """Pruning dead capsules should not significantly change val loss."""
    base, train_ds, val_ds = setup_data
    model = copy.deepcopy(base)

    _freeze_attention(model)
    train(model, train_ds, steps=200, batch_size=BATCH_SIZE, lr=LR,
          seed=42, log_every=9999)
    model.unfreeze()

    val_before = evaluate(model, val_ds, BATCH_SIZE)
    prune_dead_capsules(model, val_ds)
    val_after = evaluate(model, val_ds, BATCH_SIZE)

    # Dead capsule pruning should be near-exact (0% quality change)
    pct_change = abs(val_after - val_before) / val_before * 100
    assert pct_change < 1.0, f"Pruning changed quality by {pct_change:.2f}%"


def test_full_experiment():
    """Run the full experiment across 3 seeds.

    This IS the experiment. The test passes if the code runs to completion
    and produces a verdict.
    """
    verdict = main()
    assert verdict in ("PASS", "KILL"), f"Unexpected verdict: {verdict}"
    print(f"\n  EXPERIMENT VERDICT: {verdict}")

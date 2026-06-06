"""Tests for post-calibration pruning experiment.

Smoke tests to validate the mechanism works before running full 3-seed.
"""

import copy

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from experiments.models import get_model, MODEL_REGISTRY
from ..relu_router.test_composition import (
    compose_relu_models,
    _make_relu_model, _freeze_attention, full_capsule_calibrate,
    BASE, N_CAPSULES, BATCH_SIZE, LR,
)
from ..dead_capsule_pruning.dead_capsule_pruning import (
    profile_activations,
    identify_dead_capsules,
    prune_model,
)
from .post_calibration_pruning import (
    measure_revival_rate,
    profile_dead_mask,
)


def test_registration():
    """PostCalibrationPruningGPT is registered with correct parent."""
    assert "post_calibration_pruning" in MODEL_REGISTRY
    assert MODEL_REGISTRY["post_calibration_pruning"]["parent"] == "revival_under_composition"


def test_instantiation():
    """Model instantiates and produces correct output shape."""
    model = get_model("post_calibration_pruning", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())
    tokens = mx.zeros((2, 16), dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (2, 16, 28)


def test_measure_revival_rate():
    """Revival rate computation is correct for known masks."""
    # 4 capsules: before=[dead, dead, alive, alive], after=[alive, dead, dead, alive]
    # Capsule 0: dead->alive (revived)
    # Capsule 1: dead->dead (stayed dead)
    # Capsule 2: alive->dead (newly dead)
    # Capsule 3: alive->alive (stayed alive)
    before = [True, True, False, False]
    after = [False, True, True, False]

    rate, n_revived, n_dead_before, n_newly_dead = measure_revival_rate(before, after)

    assert n_dead_before == 2, f"Expected 2 dead before, got {n_dead_before}"
    assert n_revived == 1, f"Expected 1 revived, got {n_revived}"
    assert n_newly_dead == 1, f"Expected 1 newly dead, got {n_newly_dead}"
    assert abs(rate - 0.5) < 1e-6, f"Expected 50% revival rate, got {rate}"


def test_profile_dead_mask():
    """Profile dead mask returns sensible results on trained model."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    all_train, all_val = train_val_split(docs, seed=42)
    train_ds = CharDataset(all_train, tokenizer, BASE["block_size"])
    val_ds = CharDataset(all_val, tokenizer, BASE["block_size"])

    model = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(model, train_ds, steps=100, batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)

    dead_mask, death_rate, n_alive, n_total = profile_dead_mask(model, val_ds, seed=42)

    assert len(dead_mask) == N_CAPSULES * 4  # 4 layers
    assert 0 <= death_rate <= 1
    assert n_alive + sum(dead_mask) == n_total
    assert n_total == N_CAPSULES * 4


def test_post_calibration_prune_pipeline():
    """Smoke test: the post-calibration pruning pipeline produces valid output."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)
    V = tokenizer.vocab_size

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=42)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_train, all_val = train_val_split(docs, seed=42)
    joint_train = CharDataset(all_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_val, tokenizer, BASE["block_size"])

    domain_names = sorted(domain_datasets.keys())

    # Pretrain
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=50, batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)

    # Fine-tune per domain
    domain_models = {}
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=50,
              batch_size=BATCH_SIZE, lr=LR, seed=42, log_every=9999)
        model_d.unfreeze()
        domain_models[d_name] = model_d

    # Post-calibration pipeline: compose -> calibrate -> profile -> prune
    composed = compose_relu_models(base, [domain_models[d] for d in domain_names])

    # Profile before calibration
    mask_pre, dr_pre, _, _ = profile_dead_mask(composed, joint_val, seed=42)

    # Calibrate
    full_capsule_calibrate(composed, joint_train, steps=50, lr=LR * 0.1, seed=42)

    # Profile after calibration
    mask_post, dr_post, _, _ = profile_dead_mask(composed, joint_val, seed=42)

    # Compute revival
    rev_rate, n_rev, n_dead, n_nd = measure_revival_rate(mask_pre, mask_post)

    # Basic sanity: death rates should be reasonable
    assert 0 < dr_pre < 1, f"Death rate {dr_pre} out of range"
    assert 0 < dr_post < 1, f"Death rate {dr_post} out of range"
    assert rev_rate >= 0, f"Revival rate negative: {rev_rate}"

    # Prune and check forward pass still works
    freqs = profile_activations(composed, joint_val, n_batches=5, batch_size=16, seed=42)
    masks = identify_dead_capsules(freqs, threshold=0.0)
    prune_model(composed, masks, verbose=False)

    inputs = mx.ones((2, 32), dtype=mx.int32)
    logits = composed(inputs)
    assert logits.shape == (2, 32, V)

    # Evaluate produces a finite loss
    loss = evaluate(composed, joint_val, batch_size=16)
    assert loss > 0 and loss < 10, f"Bad loss: {loss}"

    print(f"  Smoke test passed: pre-cal death={dr_pre:.1%}, post-cal death={dr_post:.1%}, "
          f"revival={rev_rate:.1%}, final loss={loss:.4f}")

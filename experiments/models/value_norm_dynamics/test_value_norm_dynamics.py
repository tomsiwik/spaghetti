"""Tests for value norm tracking model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


BASE = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=4)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """Value tracking model produces correct output shapes."""
    print("=" * 60)
    print("test_forward_shape")

    model = get_model("value_norm_tracking_moe", **CAP)
    tokens = mx.zeros((2, 32), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (2, 32, 28), f"expected (2,32,28), got {logits.shape}"
    print(f"  shape: {logits.shape}  OK")
    print("  PASSED\n")


def test_tracking_disabled_by_default():
    """Tracking is off by default, get_value_norms returns empty dict."""
    print("=" * 60)
    print("test_tracking_disabled_by_default")

    model = get_model("value_norm_tracking_moe", **CAP)
    tokens = mx.zeros((2, 32), dtype=mx.int32)
    _ = model(tokens)
    mx.eval(model.parameters())

    norms = model.get_value_norms()
    assert norms == {}, f"Expected empty dict, got {norms}"
    print("  norms empty when tracking disabled: OK")
    print("  PASSED\n")


def test_tracking_returns_norms():
    """When enabled, tracking returns per-layer, per-head value norms."""
    print("=" * 60)
    print("test_tracking_returns_norms")

    model = get_model("value_norm_tracking_moe", **CAP)
    model.enable_tracking()
    tokens = mx.zeros((2, 32), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)

    norms = model.get_value_norms()
    # Layers 0, 1, 2 are linear (should have norms), layer 3 is full (no norms)
    assert len(norms) == 3, f"Expected 3 linear layers tracked, got {len(norms)}"
    for layer_idx, layer_norms in norms.items():
        assert len(layer_norms) == 4, f"Layer {layer_idx}: expected 4 heads, got {len(layer_norms)}"
        for head_norm in layer_norms:
            assert head_norm >= 0, f"Layer {layer_idx}: negative norm {head_norm}"
        print(f"  layer {layer_idx}: norms = {[f'{n:.4f}' for n in layer_norms]}")

    model.disable_tracking()
    print("  PASSED\n")


def test_norms_are_positive_after_training():
    """After some training, value norms are positive and finite."""
    print("=" * 60)
    print("test_norms_are_positive_after_training")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    cfg = dict(vocab_size=tok.vocab_size, block_size=32,
               n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    model = get_model("value_norm_tracking_moe", **cfg)
    train(model, ds, steps=50, batch_size=32, lr=3e-3, log_every=50)

    model.enable_tracking()
    inputs, _ = ds.get_batch(4)
    logits = model(inputs)
    mx.eval(logits)

    norms = model.get_value_norms()
    for layer_idx, layer_norms in norms.items():
        for h, n in enumerate(layer_norms):
            assert n > 0, f"Layer {layer_idx} head {h}: norm should be positive, got {n}"
            assert n < 1e6, f"Layer {layer_idx} head {h}: norm too large {n}"
        print(f"  layer {layer_idx}: {[f'{n:.4f}' for n in layer_norms]}")

    model.disable_tracking()
    print("  PASSED\n")


def test_param_count_matches_parent():
    """Value tracking model has same param count as L2 norm parent."""
    print("=" * 60)
    print("test_param_count_matches_parent")

    cfg = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    vt_model = get_model("value_norm_tracking_moe", **cfg)
    l2_model = get_model("l2_norm_hybrid_capsule_moe", **cfg)

    n_vt = sum(v.size for _, v in nn.utils.tree_flatten(vt_model.parameters()))
    n_l2 = sum(v.size for _, v in nn.utils.tree_flatten(l2_model.parameters()))

    print(f"  value_tracking: {n_vt:,} params")
    print(f"  l2_norm:        {n_l2:,} params")
    assert n_vt == n_l2, f"Param counts differ: {n_vt} vs {n_l2}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_tracking_disabled_by_default()
    test_tracking_returns_norms()
    test_param_count_matches_parent()
    test_norms_are_positive_after_training()
    print("All value norm dynamics tests passed!")

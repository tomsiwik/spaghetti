"""Tests for hybrid attention capsule MoE model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


# Small config for fast tests
BASE = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=4)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """Hybrid model produces correct output shapes."""
    print("=" * 60)
    print("test_forward_shape")

    for name in ["hybrid_capsule_moe", "full_attn_capsule_moe"]:
        model = get_model(name, **CAP)
        tokens = mx.zeros((2, 32), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (2, 32, 28), f"{name}: expected (2,32,28), got {logits.shape}"
        print(f"  {name}: {logits.shape}  OK")

    print("  PASSED\n")


def test_layer_types():
    """Verify correct layer type assignment."""
    print("=" * 60)
    print("test_layer_types")

    hybrid = get_model("hybrid_capsule_moe", **CAP)
    assert hybrid.layer_types == ["linear", "linear", "linear", "full"]
    print(f"  hybrid: {hybrid.layer_types}  OK")

    full = get_model("full_attn_capsule_moe", **CAP)
    assert full.layer_types == ["full", "full", "full", "full"]
    print(f"  full: {full.layer_types}  OK")

    # Custom pattern
    custom = get_model("hybrid_capsule_moe", **CAP,
                        layer_types=["linear", "full", "linear", "full"])
    assert custom.layer_types == ["linear", "full", "linear", "full"]
    print(f"  custom: {custom.layer_types}  OK")

    print("  PASSED\n")


def test_causal_masking_linear():
    """Linear attention is causal: future tokens don't influence past."""
    print("=" * 60)
    print("test_causal_masking_linear")

    model = get_model("hybrid_capsule_moe", **CAP)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8] + [0] * 24])
    logits_orig = model(tokens)
    mx.eval(logits_orig)

    # Perturb future token (position 5)
    tokens_perturbed = mx.array([[1, 2, 3, 4, 5, 20, 7, 8] + [0] * 24])
    logits_perturbed = model(tokens_perturbed)
    mx.eval(logits_perturbed)

    for t in range(5):
        diff = mx.max(mx.abs(logits_orig[0, t] - logits_perturbed[0, t])).item()
        assert diff < 1e-4, f"Position {t} changed (max diff={diff}) when future token modified"
        print(f"  position {t}: diff={diff:.6f}  OK")

    # Position 5 should change
    diff_at_5 = mx.max(mx.abs(logits_orig[0, 5] - logits_perturbed[0, 5])).item()
    assert diff_at_5 > 0.0, "Position 5 should change"
    print(f"  position 5: diff={diff_at_5:.6f} (changed as expected)  OK")
    print("  PASSED\n")


def test_aux_loss():
    """aux_loss returns non-negative scalar."""
    print("=" * 60)
    print("test_aux_loss")

    model = get_model("hybrid_capsule_moe", **CAP)
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    model(tokens)
    loss = model.aux_loss()
    mx.eval(loss)
    assert loss.item() >= 0.0, f"aux_loss should be >= 0, got {loss.item()}"
    print(f"  aux_loss = {loss.item():.6f}  OK")
    print("  PASSED\n")


def test_learns_names():
    """Both hybrid and full models can learn the names dataset."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    for name in ["hybrid_capsule_moe", "full_attn_capsule_moe"]:
        cfg = dict(vocab_size=tok.vocab_size, block_size=32,
                   n_embd=64, n_head=4, n_layer=4,
                   n_groups=4, n_capsules_per_group=64, top_k_groups=2)
        model = get_model(name, **cfg)
        result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

        first_loss = result["losses"][0]
        final_loss = result["final_loss"]
        print(f"  {name}: {first_loss:.4f} -> {final_loss:.4f}")
        assert final_loss < first_loss, f"{name}: loss didn't decrease"
        assert final_loss < 3.0, f"{name}: final loss too high: {final_loss:.4f}"

    print("  PASSED\n")


def test_param_count_comparable():
    """Hybrid and full models have similar param counts."""
    print("=" * 60)
    print("test_param_count_comparable")

    cfg = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    hybrid = get_model("hybrid_capsule_moe", **cfg)
    full = get_model("full_attn_capsule_moe", **cfg)

    n_hybrid = sum(v.size for _, v in nn.utils.tree_flatten(hybrid.parameters()))
    n_full = sum(v.size for _, v in nn.utils.tree_flatten(full.parameters()))

    # Linear attention has an extra W_g projection (n_embd -> n_head)
    # per linear layer, but saves nothing (same Q/K/V/O projections).
    # So hybrid should have slightly more params.
    ratio = n_hybrid / n_full
    print(f"  hybrid: {n_hybrid:,} params")
    print(f"  full:   {n_full:,} params")
    print(f"  ratio:  {ratio:.4f}")

    # Should be within 5% of each other
    assert 0.95 < ratio < 1.10, f"Param count ratio {ratio:.4f} out of expected range"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_layer_types()
    test_causal_masking_linear()
    test_aux_loss()
    test_param_count_comparable()
    test_learns_names()
    print("All hybrid attention tests passed!")

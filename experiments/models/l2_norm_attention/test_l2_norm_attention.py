"""Tests for L2-normalized hybrid attention model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


BASE = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=4)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """L2 norm model produces correct output shapes."""
    print("=" * 60)
    print("test_forward_shape")

    model = get_model("l2_norm_hybrid_capsule_moe", **CAP)
    tokens = mx.zeros((2, 32), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (2, 32, 28), f"expected (2,32,28), got {logits.shape}"
    print(f"  shape: {logits.shape}  OK")
    print("  PASSED\n")


def test_layer_types():
    """Default layer types are 3:1 linear:full."""
    print("=" * 60)
    print("test_layer_types")

    model = get_model("l2_norm_hybrid_capsule_moe", **CAP)
    assert model.layer_types == ["linear", "linear", "linear", "full"]
    print(f"  layer_types: {model.layer_types}  OK")
    print("  PASSED\n")


def test_l2_norm_bounds_qk():
    """L2 normalization bounds QK products -- key mechanism test."""
    print("=" * 60)
    print("test_l2_norm_bounds_qk")

    from experiments.models.l2_norm_attention.l2_norm_attention import l2norm

    # Random vectors
    x = mx.random.normal((4, 16))
    x_norm = l2norm(x, dim=-1)
    mx.eval(x_norm)

    # Check each row has unit norm
    norms = mx.sqrt((x_norm * x_norm).sum(axis=-1))
    mx.eval(norms)
    for i in range(4):
        assert abs(norms[i].item() - 1.0) < 1e-4, \
            f"Row {i} norm={norms[i].item()}, expected ~1.0"
    print(f"  All row norms ~1.0: OK")

    # Check QK product bounded
    qk = x_norm @ x_norm.T
    mx.eval(qk)
    max_val = mx.max(mx.abs(qk)).item()
    assert max_val <= 1.0 + 1e-4, f"QK max={max_val}, should be <=1"
    print(f"  QK product max={max_val:.4f} (bounded by 1.0)  OK")
    print("  PASSED\n")


def test_causal_masking():
    """L2-normalized linear attention is causal."""
    print("=" * 60)
    print("test_causal_masking")

    model = get_model("l2_norm_hybrid_capsule_moe", **CAP)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8] + [0] * 24])
    logits_orig = model(tokens)
    mx.eval(logits_orig)

    tokens_perturbed = mx.array([[1, 2, 3, 4, 5, 20, 7, 8] + [0] * 24])
    logits_perturbed = model(tokens_perturbed)
    mx.eval(logits_perturbed)

    for t in range(5):
        diff = mx.max(mx.abs(logits_orig[0, t] - logits_perturbed[0, t])).item()
        assert diff < 1e-4, f"Position {t} changed (diff={diff})"
        print(f"  position {t}: diff={diff:.6f}  OK")

    diff_at_5 = mx.max(mx.abs(logits_orig[0, 5] - logits_perturbed[0, 5])).item()
    assert diff_at_5 > 0.0, "Position 5 should change"
    print(f"  position 5: diff={diff_at_5:.6f} (changed)  OK")
    print("  PASSED\n")


def test_learns_names():
    """L2 norm model can learn the names dataset."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    cfg = dict(vocab_size=tok.vocab_size, block_size=32,
               n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    model = get_model("l2_norm_hybrid_capsule_moe", **cfg)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  loss: {first_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < first_loss, "Loss didn't decrease"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


def test_param_count():
    """L2 norm model has same param count as unnormalized hybrid."""
    print("=" * 60)
    print("test_param_count")

    cfg = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    l2_model = get_model("l2_norm_hybrid_capsule_moe", **cfg)
    hybrid = get_model("hybrid_capsule_moe", **cfg)

    n_l2 = sum(v.size for _, v in nn.utils.tree_flatten(l2_model.parameters()))
    n_hybrid = sum(v.size for _, v in nn.utils.tree_flatten(hybrid.parameters()))

    print(f"  l2_norm:  {n_l2:,} params")
    print(f"  hybrid:   {n_hybrid:,} params")
    # Should be identical (L2 norm adds no learnable params)
    assert n_l2 == n_hybrid, f"Param counts differ: {n_l2} vs {n_hybrid}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_layer_types()
    test_l2_norm_bounds_qk()
    test_causal_masking()
    test_param_count()
    test_learns_names()
    print("All L2 norm attention tests passed!")

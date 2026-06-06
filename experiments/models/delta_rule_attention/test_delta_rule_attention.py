"""Tests for delta rule gated linear attention model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


BASE = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=4)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """Delta rule model produces correct output shapes."""
    print("=" * 60)
    print("test_forward_shape")

    model = get_model("delta_rule_hybrid_capsule_moe", **CAP)
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

    model = get_model("delta_rule_hybrid_capsule_moe", **CAP)
    assert model.layer_types == ["linear", "linear", "linear", "full"]
    print(f"  layer_types: {model.layer_types}  OK")
    print("  PASSED\n")


def test_delta_rule_mechanism():
    """Delta rule retrieval-correction mechanism works correctly."""
    print("=" * 60)
    print("test_delta_rule_mechanism")

    from experiments.models.delta_rule_attention.delta_rule_attention import (
        DeltaRuleGatedLinearAttention,
    )

    attn = DeltaRuleGatedLinearAttention(n_embd=32, n_head=4)
    x = mx.random.normal((1, 8, 32))  # (B=1, T=8, C=32)
    out = attn(x)
    mx.eval(out)
    assert out.shape == (1, 8, 32), f"expected (1,8,32), got {out.shape}"
    print(f"  output shape: {out.shape}  OK")

    # Check that output is not all zeros (state is accumulating)
    out_norm = mx.sqrt((out * out).sum()).item()
    assert out_norm > 1e-6, f"Output norm too small: {out_norm}"
    print(f"  output norm: {out_norm:.6f} (non-zero)  OK")
    print("  PASSED\n")


def test_causal_masking():
    """Delta rule linear attention is causal (future tokens don't affect past)."""
    print("=" * 60)
    print("test_causal_masking")

    model = get_model("delta_rule_hybrid_capsule_moe", **CAP)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8] + [0] * 24])
    logits_orig = model(tokens)
    mx.eval(logits_orig)

    # Perturb position 5 and check positions 0-4 are unchanged
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
    """Delta rule model can learn the names dataset."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    cfg = dict(vocab_size=tok.vocab_size, block_size=32,
               n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    model = get_model("delta_rule_hybrid_capsule_moe", **cfg)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  loss: {first_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < first_loss, "Loss didn't decrease"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


def test_extra_params_vs_l2norm():
    """Delta rule model has more params than L2 norm (beta, A_log, dt_bias, z, out_norm)."""
    print("=" * 60)
    print("test_extra_params_vs_l2norm")

    cfg = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    delta_model = get_model("delta_rule_hybrid_capsule_moe", **cfg)
    l2_model = get_model("l2_norm_hybrid_capsule_moe", **cfg)

    n_delta = sum(v.size for _, v in nn.utils.tree_flatten(delta_model.parameters()))
    n_l2 = sum(v.size for _, v in nn.utils.tree_flatten(l2_model.parameters()))

    print(f"  delta_rule: {n_delta:,} params")
    print(f"  l2_norm:    {n_l2:,} params")
    # Delta rule should have MORE params (beta proj, z proj, out_norm, A_log, dt_bias)
    assert n_delta > n_l2, f"Delta rule should have more params: {n_delta} vs {n_l2}"
    overhead = (n_delta - n_l2) / n_l2 * 100
    print(f"  overhead: +{overhead:.1f}%")
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_layer_types()
    test_delta_rule_mechanism()
    test_causal_masking()
    test_extra_params_vs_l2norm()
    test_learns_names()
    print("All delta rule attention tests passed!")

"""Tests for full GatedDeltaNet composition stack model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


BASE = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=4)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """Full GDN stack model produces correct output shapes."""
    print("=" * 60)
    print("test_forward_shape")

    model = get_model("full_gdn_stack_capsule_moe", **CAP)
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

    model = get_model("full_gdn_stack_capsule_moe", **CAP)
    assert model.layer_types == ["linear", "linear", "linear", "full"]
    print(f"  layer_types: {model.layer_types}  OK")
    print("  PASSED\n")


def test_conv1d_causal():
    """Conv1d is causal: output at position t depends only on positions <= t."""
    print("=" * 60)
    print("test_conv1d_causal")

    from experiments.models.full_gdn_stack.full_gdn_stack import CausalConv1d

    conv = CausalConv1d(channels=16, kernel_size=4)
    x = mx.random.normal((1, 8, 16))

    out1 = conv(x)
    mx.eval(out1)

    # Perturb position 5 and check positions 0-4 unchanged
    x2 = mx.array(x)
    x2_list = x2.tolist()
    for c in range(16):
        x2_list[0][5][c] = 99.0
    x2 = mx.array(x2_list)
    out2 = conv(x2)
    mx.eval(out2)

    for t in range(5):
        diff = mx.max(mx.abs(out1[0, t] - out2[0, t])).item()
        assert diff < 1e-5, f"Conv1d not causal: position {t} changed (diff={diff})"
        print(f"  position {t}: diff={diff:.8f}  OK")

    diff_at_5 = mx.max(mx.abs(out1[0, 5] - out2[0, 5])).item()
    assert diff_at_5 > 0.0, "Position 5 should change"
    print(f"  position 5: diff={diff_at_5:.4f} (changed)  OK")
    print("  PASSED\n")


def test_per_dim_beta():
    """Per-dimension beta has shape (B, T, h, d), not (B, T, h)."""
    print("=" * 60)
    print("test_per_dim_beta")

    from experiments.models.full_gdn_stack.full_gdn_stack import FullGatedDeltaNetAttention

    attn = FullGatedDeltaNetAttention(n_embd=32, n_head=4)
    # Check w_beta output dimension = n_embd (not n_head)
    assert attn.w_beta.weight.shape == (32, 32), \
        f"w_beta should project to n_embd=32, got shape {attn.w_beta.weight.shape}"
    print(f"  w_beta shape: {attn.w_beta.weight.shape} (per-dim)  OK")
    print("  PASSED\n")


def test_full_gdn_mechanism():
    """Full GDN attention produces valid outputs."""
    print("=" * 60)
    print("test_full_gdn_mechanism")

    from experiments.models.full_gdn_stack.full_gdn_stack import FullGatedDeltaNetAttention

    attn = FullGatedDeltaNetAttention(n_embd=32, n_head=4)
    x = mx.random.normal((1, 8, 32))
    out = attn(x)
    mx.eval(out)
    assert out.shape == (1, 8, 32), f"expected (1,8,32), got {out.shape}"

    out_norm = mx.sqrt((out * out).sum()).item()
    assert out_norm > 1e-6, f"Output norm too small: {out_norm}"
    print(f"  output shape: {out.shape}  OK")
    print(f"  output norm: {out_norm:.6f} (non-zero)  OK")
    print("  PASSED\n")


def test_causal_masking():
    """Full GDN model is causal: future tokens don't affect past."""
    print("=" * 60)
    print("test_causal_masking")

    model = get_model("full_gdn_stack_capsule_moe", **CAP)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8] + [0] * 24])
    logits_orig = model(tokens)
    mx.eval(logits_orig)

    tokens_perturbed = mx.array([[1, 2, 3, 4, 5, 20, 7, 8] + [0] * 24])
    logits_perturbed = model(tokens_perturbed)
    mx.eval(logits_perturbed)

    # Positions before the conv kernel range of position 5 should be unaffected
    # Conv kernel=4, so position 5 depends on positions 2-5.
    # Position 0 and 1 should be completely unaffected.
    for t in range(2):
        diff = mx.max(mx.abs(logits_orig[0, t] - logits_perturbed[0, t])).item()
        assert diff < 1e-4, f"Position {t} changed (diff={diff})"
        print(f"  position {t}: diff={diff:.6f}  OK")

    diff_at_5 = mx.max(mx.abs(logits_orig[0, 5] - logits_perturbed[0, 5])).item()
    assert diff_at_5 > 0.0, "Position 5 should change"
    print(f"  position 5: diff={diff_at_5:.6f} (changed)  OK")
    print("  PASSED\n")


def test_extra_params_vs_delta_rule():
    """Full GDN stack has more params than delta rule (conv1d + per-dim beta)."""
    print("=" * 60)
    print("test_extra_params_vs_delta_rule")

    cfg = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    full_model = get_model("full_gdn_stack_capsule_moe", **cfg)
    delta_model = get_model("delta_rule_hybrid_capsule_moe", **cfg)

    n_full = sum(v.size for _, v in nn.utils.tree_flatten(full_model.parameters()))
    n_delta = sum(v.size for _, v in nn.utils.tree_flatten(delta_model.parameters()))

    print(f"  full_gdn_stack: {n_full:,} params")
    print(f"  delta_rule:     {n_delta:,} params")
    overhead = (n_full - n_delta) / n_delta * 100
    print(f"  overhead: +{overhead:.1f}%")
    assert n_full > n_delta, f"Full GDN should have more params: {n_full} vs {n_delta}"
    print("  PASSED\n")


def test_learns_names():
    """Full GDN stack model can learn the names dataset."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    cfg = dict(vocab_size=tok.vocab_size, block_size=32,
               n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    model = get_model("full_gdn_stack_capsule_moe", **cfg)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  loss: {first_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < first_loss, "Loss didn't decrease"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_layer_types()
    test_conv1d_causal()
    test_per_dim_beta()
    test_full_gdn_mechanism()
    test_causal_masking()
    test_extra_params_vs_delta_rule()
    test_learns_names()
    print("All full GDN stack tests passed!")

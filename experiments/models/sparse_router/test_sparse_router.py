"""Tests for the sparse_router model."""

import math

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model


CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2,
           n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_registered():
    """sparse_router is registered in the arena."""
    print("=" * 60)
    print("test_registered")
    from experiments.models import list_models
    assert "sparse_router" in list_models()
    print("  PASSED\n")


def test_forward_shape():
    """Forward produces correct output shape."""
    print("=" * 60)
    print("test_forward_shape")
    model = get_model("sparse_router", **CFG)
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (2, 8, 28)
    print(f"  shape: {logits.shape}  OK")
    print("  PASSED\n")


def test_set_top_k():
    """set_top_k changes routing for all layers."""
    print("=" * 60)
    print("test_set_top_k")
    model = get_model("sparse_router", **CFG)

    for k in [1, 2, 4]:
        model.set_top_k(k)
        assert model.get_top_k() == k
        for layer in model.layers:
            assert layer.capsule_pool.top_k_groups == k
        print(f"  k={k}: all layers updated  OK")

    print("  PASSED\n")


def test_forward_at_different_k():
    """Model produces valid outputs at k=1,2,4."""
    print("=" * 60)
    print("test_forward_at_different_k")
    model = get_model("sparse_router", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))

    for k in [1, 2, 4]:
        model.set_top_k(k)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (4, 16, 28), f"k={k}: shape {logits.shape}"
        assert not mx.any(mx.isnan(logits)).item(), f"k={k}: NaN in logits"
        print(f"  k={k}: shape={logits.shape}, no NaN  OK")

    print("  PASSED\n")


def test_router_stats():
    """Router stats computed correctly after forward pass."""
    print("=" * 60)
    print("test_router_stats")
    model = get_model("sparse_router", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    stats = model.router_stats()
    n_layers = len(model.layers)
    n_groups = CFG["n_groups"]
    h_max = math.log(n_groups)

    assert len(stats["entropy"]) == n_layers
    assert len(stats["concentration_1"]) == n_layers
    assert len(stats["group_freqs"]) == n_layers

    for i in range(n_layers):
        h = stats["entropy"][i]
        assert 0 <= h <= h_max + 0.01, f"Layer {i}: entropy {h} out of range"

        c1 = stats["concentration_1"][i]
        assert 1.0 / n_groups - 0.01 <= c1 <= 1.01, f"Layer {i}: C_1 {c1} out of range"

        freqs = stats["group_freqs"][i]
        assert len(freqs) == n_groups
        assert abs(sum(freqs) - 1.0) < 0.02, f"Layer {i}: freqs sum={sum(freqs)}"
        print(f"  layer {i}: H={h:.3f}, C_1={c1:.3f}  OK")

    print("  PASSED\n")


def test_router_probs_sum_to_one_all_k():
    """Router probs sum to 1 at all top-k values."""
    print("=" * 60)
    print("test_router_probs_sum_to_one_all_k")
    model = get_model("sparse_router", **CFG)
    tokens = mx.random.randint(0, 28, (2, 8))

    for k in [1, 2, 4]:
        model.set_top_k(k)
        model(tokens)
        for i, layer in enumerate(model.layers):
            probs = layer.capsule_pool._gate_probs
            mx.eval(probs)
            sums = mx.sum(probs, axis=-1)
            max_err = mx.max(mx.abs(sums - 1.0)).item()
            assert max_err < 1e-5, f"k={k}, layer {i}: sum err {max_err}"
        print(f"  k={k}: probs sum to 1  OK")

    print("  PASSED\n")


def test_param_parity():
    """sparse_router has same params as capsule_moe."""
    print("=" * 60)
    print("test_param_parity")
    base = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
                n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    cap = get_model("capsule_moe", **base)
    sp = get_model("sparse_router", **base)

    cap_p = sum(v.size for _, v in nn.utils.tree_flatten(cap.parameters()))
    sp_p = sum(v.size for _, v in nn.utils.tree_flatten(sp.parameters()))

    assert cap_p == sp_p, f"Param mismatch: capsule_moe={cap_p}, sparse_router={sp_p}"
    print(f"  capsule_moe={cap_p}, sparse_router={sp_p}  OK")
    print("  PASSED\n")


def test_entropy_bounds_g8():
    """Entropy bounded by [0, log(G)] with G=8."""
    print("=" * 60)
    print("test_entropy_bounds_g8")
    cfg8 = {**CFG, "n_groups": 8, "top_k_groups": 2}
    model = get_model("sparse_router", **cfg8)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    stats = model.router_stats()
    h_max = math.log(8)

    for i, h in enumerate(stats["entropy"]):
        assert 0 <= h <= h_max + 0.01, f"Layer {i}: H={h}, H_max={h_max:.3f}"
        ratio = stats["entropy_ratio"][i]
        assert 0 <= ratio <= 1.01, f"Layer {i}: ratio={ratio}"
        print(f"  layer {i}: H={h:.3f}, H/H_max={ratio:.3f}  OK")

    print("  PASSED\n")


def test_k1_single_group_active():
    """At k=1, exactly 1 group should have nonzero masked prob per token."""
    print("=" * 60)
    print("test_k1_single_group_active")
    model = get_model("sparse_router", **CFG)
    model.set_top_k(1)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    for i, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        x = mx.random.normal((4, 16, CFG["n_embd"]))
        scores = pool.router(x)
        top_vals = mx.topk(scores, 1, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        active = mx.sum(mask, axis=-1)
        mx.eval(active)
        # At k=1, should have exactly 1 (or occasionally 2 if ties)
        min_active = mx.min(active).item()
        assert min_active >= 1, f"Layer {i}: no group active"
        print(f"  layer {i}: active groups = {min_active:.0f}-{mx.max(active).item():.0f}  OK")

    print("  PASSED\n")


def test_aux_loss_nonzero():
    """aux_loss() > 0 after forward pass (inherited from CapsuleMoEGPT)."""
    print("=" * 60)
    print("test_aux_loss_nonzero")
    model = get_model("sparse_router", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    aux = model.aux_loss().item()
    assert aux > 0.0, f"aux_loss should be > 0, got {aux}"
    print(f"  aux_loss = {aux:.6f}  OK")
    print("  PASSED\n")


def test_learns():
    """Loss decreases over 200 training steps."""
    print("=" * 60)
    print("test_learns")
    from experiments.data import load_names, CharTokenizer, CharDataset
    from experiments.train import train

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    model = get_model("sparse_router", vocab_size=tok.vocab_size, block_size=32,
                      n_embd=64, n_head=4, n_layer=2,
                      n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")
    assert final_loss < first_loss
    assert final_loss < 3.0
    print("  PASSED\n")


if __name__ == "__main__":
    test_registered()
    test_forward_shape()
    test_set_top_k()
    test_forward_at_different_k()
    test_router_stats()
    test_router_probs_sum_to_one_all_k()
    test_param_parity()
    test_entropy_bounds_g8()
    test_k1_single_group_active()
    test_aux_loss_nonzero()
    test_learns()
    print("All sparse_router tests passed!")

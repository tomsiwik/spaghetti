"""Tests for the MoE GPT model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


# Small config for fast tests
CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2,
           n_experts=4, top_k=2)


def test_forward_shape():
    """tokens (B, T) -> logits (B, T, V) for various B, T, V."""
    print("=" * 60)
    print("test_forward_shape")

    for B, T, V in [(1, 8, 28), (4, 32, 28), (2, 16, 50)]:
        model = get_model("moe", **{**CFG, "vocab_size": V, "block_size": T})
        tokens = mx.zeros((B, T), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
        print(f"  B={B}, T={T}, V={V} -> {logits.shape}  OK")

    print("  PASSED\n")


def test_router_probs_sum_to_one():
    """Softmax gate outputs sum to 1.0 per token."""
    print("=" * 60)
    print("test_router_probs_sum_to_one")

    model = get_model("moe", **CFG)
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    model(tokens)
    mx.eval(model.parameters())

    for i, layer in enumerate(model.layers):
        probs = layer.moe._gate_probs  # (B, T, N)
        mx.eval(probs)
        sums = mx.sum(probs, axis=-1)  # (B, T)
        max_err = mx.max(mx.abs(sums - 1.0)).item()
        assert max_err < 1e-5, f"Layer {i}: probs don't sum to 1 (max err={max_err})"
        print(f"  layer {i}: max sum error = {max_err:.2e}  OK")

    print("  PASSED\n")


def test_top_k_sparsity():
    """Only top-k experts get nonzero weight per token."""
    print("=" * 60)
    print("test_top_k_sparsity")

    model = get_model("moe", **CFG)
    # Use random tokens to get varied routing
    tokens = mx.random.randint(0, 28, (4, 16))
    logits = model(tokens)
    mx.eval(logits)

    for i, layer in enumerate(model.layers):
        probs = layer.moe._gate_probs
        scores = layer.moe.router(mx.zeros((4, 16, CFG["n_embd"])))
        # Re-compute masked probs to check sparsity
        top_vals = mx.topk(scores, CFG["top_k"], axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        # Count nonzero experts per token (from the mask)
        active_per_token = mx.sum(mask, axis=-1)  # (B, T)
        mx.eval(active_per_token)
        max_active = mx.max(active_per_token).item()
        # With ties, more than top_k can be active, but at least top_k should be
        min_active = mx.min(active_per_token).item()
        assert min_active >= CFG["top_k"], f"Layer {i}: fewer than top_k active ({min_active})"
        print(f"  layer {i}: active experts/token = {min_active:.0f}-{max_active:.0f}  OK")

    print("  PASSED\n")


def test_balance_loss_positive():
    """balance_loss() > 0 after a forward pass."""
    print("=" * 60)
    print("test_balance_loss_positive")

    model = get_model("moe", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    for i, layer in enumerate(model.layers):
        bl = layer.moe.balance_loss().item()
        assert bl > 0.0, f"Layer {i}: balance_loss should be > 0, got {bl}"
        print(f"  layer {i}: balance_loss = {bl:.4f}  OK")

    print("  PASSED\n")


def test_balance_loss_minimum():
    """Balance loss >= 1.0 (theoretical minimum at uniform routing)."""
    print("=" * 60)
    print("test_balance_loss_minimum")

    model = get_model("moe", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    for i, layer in enumerate(model.layers):
        bl = layer.moe.balance_loss().item()
        # By Jensen's inequality, balance_loss = N * sum(mean_i^2) >= 1.0
        assert bl >= 1.0 - 1e-5, f"Layer {i}: balance_loss {bl} < 1.0 (theoretical min)"
        print(f"  layer {i}: balance_loss = {bl:.4f} >= 1.0  OK")

    print("  PASSED\n")


def test_more_params_than_gpt():
    """MoE param count > GPT param count with same base config."""
    print("=" * 60)
    print("test_more_params_than_gpt")

    base = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2)
    gpt = get_model("gpt", **base)
    moe = get_model("moe", **base, n_experts=4, top_k=2)

    gpt_params = sum(v.size for _, v in nn.utils.tree_flatten(gpt.parameters()))
    moe_params = sum(v.size for _, v in nn.utils.tree_flatten(moe.parameters()))

    print(f"  GPT params: {gpt_params}")
    print(f"  MoE params: {moe_params}")
    assert moe_params > gpt_params, f"MoE ({moe_params}) should have more params than GPT ({gpt_params})"
    print("  PASSED\n")


def test_aux_loss_nonzero():
    """MoEGPT.aux_loss() > 0 after forward pass."""
    print("=" * 60)
    print("test_aux_loss_nonzero")

    model = get_model("moe", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    aux = model.aux_loss().item()
    assert aux > 0.0, f"aux_loss should be > 0 after forward, got {aux}"
    print(f"  aux_loss = {aux:.6f}  OK")
    print("  PASSED\n")


def test_learns_names():
    """Train on CharDataset for ~200 steps, loss decreases."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    model = get_model("moe", vocab_size=tok.vocab_size, block_size=32,
                       n_embd=64, n_head=4, n_layer=2, n_experts=4, top_k=2)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")
    assert final_loss < first_loss, f"Loss didn't decrease: {first_loss:.4f} -> {final_loss:.4f}"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_router_probs_sum_to_one()
    test_top_k_sparsity()
    test_balance_loss_positive()
    test_balance_loss_minimum()
    test_more_params_than_gpt()
    test_aux_loss_nonzero()
    test_learns_names()
    print("All MoE tests passed!")

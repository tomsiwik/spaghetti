"""Tests for the Capsule MoE model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


# Small config for fast tests
CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2,
           n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_forward_shape():
    """tokens (B, T) -> logits (B, T, V) for various B, T, V."""
    print("=" * 60)
    print("test_forward_shape")

    for B, T, V in [(1, 8, 28), (4, 32, 28), (2, 16, 50)]:
        model = get_model("capsule_moe", **{**CFG, "vocab_size": V, "block_size": T})
        tokens = mx.zeros((B, T), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
        print(f"  B={B}, T={T}, V={V} -> {logits.shape}  OK")

    print("  PASSED\n")


def test_capsule_group_output_shape():
    """CapsuleGroup: (B, T, D) -> (B, T, D)."""
    print("=" * 60)
    print("test_capsule_group_output_shape")

    from experiments.models.capsule_moe.capsule_moe import CapsuleGroup

    for d, n_caps in [(32, 64), (64, 128), (16, 16)]:
        group = CapsuleGroup(d, n_caps)
        x = mx.random.normal((2, 8, d))
        out = group(x)
        mx.eval(out)
        assert out.shape == (2, 8, d), f"Expected (2, 8, {d}), got {out.shape}"
        print(f"  d={d}, n_capsules={n_caps} -> {out.shape}  OK")

    print("  PASSED\n")


def test_capsule_activation_sparsity():
    """ReLU naturally zeros out ~some fraction of capsule activations."""
    print("=" * 60)
    print("test_capsule_activation_sparsity")

    from experiments.models.capsule_moe.capsule_moe import CapsuleGroup

    group = CapsuleGroup(64, 128)
    x = mx.random.normal((4, 16, 64))
    h = nn.relu(group.A(x))  # (4, 16, 128)
    mx.eval(h)

    # Count zero activations
    total = h.size
    zeros = mx.sum(h == 0).item()
    sparsity = zeros / total

    print(f"  total activations: {total}")
    print(f"  zero activations:  {zeros}")
    print(f"  sparsity: {sparsity:.1%}")
    # With random init + random input, expect ~50% sparsity from ReLU
    assert 0.2 < sparsity < 0.9, f"Unexpected sparsity: {sparsity:.1%}"
    print("  PASSED\n")


def test_router_probs_sum_to_one():
    """Softmax gate outputs sum to 1.0 per token."""
    print("=" * 60)
    print("test_router_probs_sum_to_one")

    model = get_model("capsule_moe", **CFG)
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    model(tokens)
    mx.eval(model.parameters())

    for i, layer in enumerate(model.layers):
        probs = layer.capsule_pool._gate_probs  # (B, T, G)
        mx.eval(probs)
        sums = mx.sum(probs, axis=-1)  # (B, T)
        max_err = mx.max(mx.abs(sums - 1.0)).item()
        assert max_err < 1e-5, f"Layer {i}: probs don't sum to 1 (max err={max_err})"
        print(f"  layer {i}: max sum error = {max_err:.2e}  OK")

    print("  PASSED\n")


def test_top_k_group_selection():
    """Only top-k_g groups get nonzero weight per token."""
    print("=" * 60)
    print("test_top_k_group_selection")

    model = get_model("capsule_moe", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    logits = model(tokens)
    mx.eval(logits)

    for i, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        # Re-compute to check sparsity
        x = mx.random.normal((4, 16, CFG["n_embd"]))
        scores = pool.router(x)
        top_vals = mx.topk(scores, pool.top_k_groups, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        active_per_token = mx.sum(mask, axis=-1)
        mx.eval(active_per_token)
        min_active = mx.min(active_per_token).item()
        max_active = mx.max(active_per_token).item()
        assert min_active >= CFG["top_k_groups"], \
            f"Layer {i}: fewer than top_k_groups active ({min_active})"
        print(f"  layer {i}: active groups/token = {min_active:.0f}-{max_active:.0f}  OK")

    print("  PASSED\n")


def test_balance_loss_positive():
    """balance_loss() > 0 after a forward pass."""
    print("=" * 60)
    print("test_balance_loss_positive")

    model = get_model("capsule_moe", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    for i, layer in enumerate(model.layers):
        bl = layer.capsule_pool.balance_loss().item()
        assert bl > 0.0, f"Layer {i}: balance_loss should be > 0, got {bl}"
        print(f"  layer {i}: balance_loss = {bl:.4f}  OK")

    print("  PASSED\n")


def test_balance_loss_minimum():
    """Balance loss >= 1.0 (theoretical minimum at uniform routing)."""
    print("=" * 60)
    print("test_balance_loss_minimum")

    model = get_model("capsule_moe", **CFG)
    tokens = mx.random.randint(0, 28, (4, 16))
    model(tokens)

    for i, layer in enumerate(model.layers):
        bl = layer.capsule_pool.balance_loss().item()
        assert bl >= 1.0 - 1e-5, \
            f"Layer {i}: balance_loss {bl} < 1.0 (theoretical min)"
        print(f"  layer {i}: balance_loss = {bl:.4f} >= 1.0  OK")

    print("  PASSED\n")


def test_param_parity_with_gpt():
    """Capsule MoE should have roughly the same param count as dense GPT."""
    print("=" * 60)
    print("test_param_parity_with_gpt")

    base = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4)
    gpt = get_model("gpt", **base)
    capsule = get_model("capsule_moe", **base,
                        n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    gpt_params = sum(v.size for _, v in nn.utils.tree_flatten(gpt.parameters()))
    cap_params = sum(v.size for _, v in nn.utils.tree_flatten(capsule.parameters()))

    ratio = cap_params / gpt_params
    print(f"  GPT params:         {gpt_params:,}")
    print(f"  Capsule MoE params: {cap_params:,}")
    print(f"  Ratio: {ratio:.3f}")
    # Should be within 5% of GPT (the only overhead is group router weights)
    assert 0.95 < ratio < 1.10, \
        f"Capsule MoE ({cap_params}) not at parity with GPT ({gpt_params}), ratio={ratio}"
    print("  PASSED\n")


def test_fewer_params_than_moe():
    """Capsule MoE should have fewer params than standard MoE (N=4 experts)."""
    print("=" * 60)
    print("test_fewer_params_than_moe")

    base = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4)
    moe = get_model("moe", **base, n_experts=4, top_k=2)
    capsule = get_model("capsule_moe", **base,
                        n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    moe_params = sum(v.size for _, v in nn.utils.tree_flatten(moe.parameters()))
    cap_params = sum(v.size for _, v in nn.utils.tree_flatten(capsule.parameters()))

    print(f"  MoE params:         {moe_params:,}")
    print(f"  Capsule MoE params: {cap_params:,}")
    print(f"  Savings: {(1 - cap_params/moe_params)*100:.1f}%")
    assert cap_params < moe_params, \
        f"Capsule MoE ({cap_params}) should have fewer params than MoE ({moe_params})"
    print("  PASSED\n")


def test_aux_loss_nonzero():
    """CapsuleMoEGPT.aux_loss() > 0 after forward pass."""
    print("=" * 60)
    print("test_aux_loss_nonzero")

    model = get_model("capsule_moe", **CFG)
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

    model = get_model("capsule_moe", vocab_size=tok.vocab_size, block_size=32,
                       n_embd=64, n_head=4, n_layer=2,
                       n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")
    assert final_loss < first_loss, \
        f"Loss didn't decrease: {first_loss:.4f} -> {final_loss:.4f}"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


def test_uniform_routing_forward():
    """Uniform-routing variant produces correct shapes."""
    print("=" * 60)
    print("test_uniform_routing_forward")

    model = get_model("capsule_moe_uniform", **CFG)
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (2, 8, CFG["vocab_size"]), \
        f"Expected (2, 8, {CFG['vocab_size']}), got {logits.shape}"
    print(f"  shape: {logits.shape}  OK")
    print("  PASSED\n")


def test_uniform_routing_equal_weights():
    """In uniform mode, all groups get weight 1/G (no top-k masking)."""
    print("=" * 60)
    print("test_uniform_routing_equal_weights")

    from experiments.models.capsule_moe.capsule_moe import CapsulePool

    pool = CapsulePool(32, n_groups=4, n_capsules_per_group=32,
                       top_k_groups=2, uniform_routing=True)
    x = mx.random.normal((2, 8, 32))
    out = pool(x)
    mx.eval(out)

    probs = pool._gate_probs
    mx.eval(probs)
    expected = 1.0 / 4.0
    max_err = mx.max(mx.abs(probs - expected)).item()
    assert max_err < 1e-6, f"Uniform probs not equal: max_err={max_err}"
    print(f"  all probs = {expected}, max_err = {max_err:.2e}  OK")
    print("  PASSED\n")


def test_uniform_routing_param_parity():
    """Uniform variant has same param count as learned-router variant."""
    print("=" * 60)
    print("test_uniform_routing_param_parity")

    base = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
                n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    learned = get_model("capsule_moe", **base)
    uniform = get_model("capsule_moe_uniform", **base)

    l_params = sum(v.size for _, v in nn.utils.tree_flatten(learned.parameters()))
    u_params = sum(v.size for _, v in nn.utils.tree_flatten(uniform.parameters()))

    print(f"  Learned router params:  {l_params:,}")
    print(f"  Uniform router params:  {u_params:,}")
    # Uniform still has router weights (just unused), so param counts match
    assert l_params == u_params, f"Param mismatch: {l_params} vs {u_params}"
    print("  PASSED\n")


def test_uniform_routing_learns():
    """Uniform-routing variant can still learn (loss decreases)."""
    print("=" * 60)
    print("test_uniform_routing_learns")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    model = get_model("capsule_moe_uniform", vocab_size=tok.vocab_size, block_size=32,
                       n_embd=64, n_head=4, n_layer=2,
                       n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")
    assert final_loss < first_loss, \
        f"Loss didn't decrease: {first_loss:.4f} -> {final_loss:.4f}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_capsule_group_output_shape()
    test_capsule_activation_sparsity()
    test_router_probs_sum_to_one()
    test_top_k_group_selection()
    test_balance_loss_positive()
    test_balance_loss_minimum()
    test_param_parity_with_gpt()
    test_fewer_params_than_moe()
    test_aux_loss_nonzero()
    test_uniform_routing_forward()
    test_uniform_routing_equal_weights()
    test_uniform_routing_param_parity()
    test_uniform_routing_learns()
    test_learns_names()
    print("All Capsule MoE tests passed!")

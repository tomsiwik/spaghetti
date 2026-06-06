"""Tests for the Contrastive Router model."""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train
from experiments.models.contrastive_router.contrastive_router import (
    ContrastiveCapsulePool, extract_hidden_states,
    infonce_loss, routing_accuracy,
)


# Small config for fast tests
CFG = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=2,
           n_groups=4, n_capsules_per_group=32, top_k_groups=2, d_key=8)


def test_forward_shape():
    """tokens (B, T) -> logits (B, T, V) for various B, T, V."""
    print("=" * 60)
    print("test_forward_shape")

    for B, T, V in [(1, 8, 28), (4, 32, 28), (2, 16, 50)]:
        model = get_model("contrastive_router",
                          **{**CFG, "vocab_size": V, "block_size": T})
        tokens = mx.zeros((B, T), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (B, T, V), f"Expected {(B, T, V)}, got {logits.shape}"
        print(f"  B={B}, T={T}, V={V} -> {logits.shape}  OK")

    print("  PASSED\n")


def test_routing_scores_shape():
    """Routing scores have shape (..., N_groups)."""
    print("=" * 60)
    print("test_routing_scores_shape")

    pool = ContrastiveCapsulePool(32, n_groups=4, n_capsules_per_group=32,
                                  top_k_groups=2, d_key=8)

    # 2D input (B, d)
    x2d = mx.random.normal((16, 32))
    s2d = pool.routing_scores(x2d)
    mx.eval(s2d)
    assert s2d.shape == (16, 4), f"Expected (16, 4), got {s2d.shape}"
    print(f"  2D: (16, 32) -> {s2d.shape}  OK")

    # 3D input (B, T, d)
    x3d = mx.random.normal((2, 8, 32))
    s3d = pool.routing_scores(x3d)
    mx.eval(s3d)
    assert s3d.shape == (2, 8, 4), f"Expected (2, 8, 4), got {s3d.shape}"
    print(f"  3D: (2, 8, 32) -> {s3d.shape}  OK")

    print("  PASSED\n")


def test_routing_scores_nonnegative():
    """Squared norms are always >= 0."""
    print("=" * 60)
    print("test_routing_scores_nonnegative")

    pool = ContrastiveCapsulePool(64, n_groups=8, d_key=8)
    x = mx.random.normal((32, 64))
    scores = pool.routing_scores(x)
    mx.eval(scores)
    min_val = mx.min(scores).item()
    assert min_val >= 0.0, f"Scores should be >= 0, got min={min_val}"
    print(f"  min score = {min_val:.6f} >= 0  OK")

    print("  PASSED\n")


def test_infonce_loss_computable():
    """InfoNCE loss computes without error and is positive."""
    print("=" * 60)
    print("test_infonce_loss_computable")

    pool = ContrastiveCapsulePool(32, n_groups=4, d_key=8)
    h = mx.random.normal((64, 32))
    labels = mx.concatenate([mx.zeros(32, dtype=mx.int32),
                              mx.ones(32, dtype=mx.int32)])
    loss = infonce_loss(pool, h, labels, groups_per_domain=2, tau=0.1)
    mx.eval(loss)
    loss_val = loss.item()
    assert loss_val > 0.0, f"Loss should be > 0, got {loss_val}"
    print(f"  loss = {loss_val:.4f} > 0  OK")

    print("  PASSED\n")


def test_infonce_gradient_flows():
    """Gradient of InfoNCE loss w.r.t. keys is nonzero."""
    print("=" * 60)
    print("test_infonce_gradient_flows")

    pool = ContrastiveCapsulePool(32, n_groups=4, d_key=8)
    h = mx.random.normal((64, 32))
    labels = mx.concatenate([mx.zeros(32, dtype=mx.int32),
                              mx.ones(32, dtype=mx.int32)])

    def loss_fn(pool, h, labels):
        return infonce_loss(pool, h, labels, groups_per_domain=2, tau=0.1)

    loss_and_grad = nn.value_and_grad(pool, loss_fn)
    loss, grads = loss_and_grad(pool, h, labels)
    mx.eval(loss, grads)

    # Check that at least some key gradients are nonzero
    total_grad_norm = 0.0
    for _, v in nn.utils.tree_flatten(grads):
        total_grad_norm += mx.sum(v * v).item()
    assert total_grad_norm > 0.0, "Gradients should be nonzero"
    print(f"  total grad norm^2 = {total_grad_norm:.6f} > 0  OK")

    print("  PASSED\n")


def test_routing_accuracy_random():
    """Random keys give ~50% routing accuracy on 2 domains."""
    print("=" * 60)
    print("test_routing_accuracy_random")

    pool = ContrastiveCapsulePool(64, n_groups=8, d_key=8)
    h = mx.random.normal((200, 64))
    labels = mx.concatenate([mx.zeros(100, dtype=mx.int32),
                              mx.ones(100, dtype=mx.int32)])
    acc = routing_accuracy(pool, h, labels, groups_per_domain=4)
    print(f"  random accuracy = {acc:.1%}")
    # With random keys, should be near 50% (allow wide range for randomness)
    assert 0.2 < acc < 0.8, f"Random accuracy {acc:.1%} too far from 50%"

    print("  PASSED\n")


def test_uniform_routing():
    """Uniform routing produces correct shapes and equal probs."""
    print("=" * 60)
    print("test_uniform_routing")

    pool = ContrastiveCapsulePool(32, n_groups=4, n_capsules_per_group=32,
                                  top_k_groups=2, d_key=8)
    pool.uniform_routing = True
    x = mx.random.normal((2, 8, 32))
    out = pool(x)
    mx.eval(out)
    assert out.shape == (2, 8, 32), f"Expected (2, 8, 32), got {out.shape}"

    probs = pool._gate_probs
    mx.eval(probs)
    max_err = mx.max(mx.abs(probs - 0.25)).item()
    assert max_err < 1e-6, f"Uniform probs not 0.25: max_err={max_err}"
    print(f"  shape: {out.shape}, probs max_err = {max_err:.2e}  OK")

    print("  PASSED\n")


def test_extract_hidden_states():
    """extract_hidden_states returns correct shapes."""
    print("=" * 60)
    print("test_extract_hidden_states")

    model = get_model("contrastive_router", **CFG)
    tokens = mx.zeros((2, 16), dtype=mx.int32)
    hiddens = extract_hidden_states(model, tokens)
    mx.eval(hiddens)

    assert len(hiddens) == CFG["n_layer"], \
        f"Expected {CFG['n_layer']} layers, got {len(hiddens)}"
    for i, h in enumerate(hiddens):
        expected = (2, 16, CFG["n_embd"])
        assert h.shape == expected, \
            f"Layer {i}: expected {expected}, got {h.shape}"
        print(f"  layer {i}: {h.shape}  OK")

    print("  PASSED\n")


def test_param_count():
    """Contrastive router has expected param overhead vs capsule_moe."""
    print("=" * 60)
    print("test_param_count")

    base = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4)
    cap = get_model("capsule_moe", **base,
                     n_groups=4, n_capsules_per_group=64, top_k_groups=2)
    cont = get_model("contrastive_router", **base,
                      n_groups=4, n_capsules_per_group=64, top_k_groups=2,
                      d_key=8)

    cap_params = sum(v.size for _, v in nn.utils.tree_flatten(cap.parameters()))
    cont_params = sum(v.size for _, v in nn.utils.tree_flatten(cont.parameters()))

    # Contrastive router replaces linear router (G*d per layer) with
    # contrastive keys (G*d*d_key per layer)
    # Linear router: 4 * 4 * 64 = 1,024 total
    # Contrastive keys: 4 * 4 * 64 * 8 = 8,192 total
    # Net overhead: 8,192 - 1,024 = 7,168
    expected_overhead = 4 * 4 * 64 * 8 - 4 * 4 * 64  # keys - router
    actual_overhead = cont_params - cap_params

    print(f"  Capsule MoE params:       {cap_params:,}")
    print(f"  Contrastive router params: {cont_params:,}")
    print(f"  Overhead: {actual_overhead:,} (expected ~{expected_overhead:,})")
    assert abs(actual_overhead - expected_overhead) < 100, \
        f"Unexpected param overhead: {actual_overhead} vs expected {expected_overhead}"

    print("  PASSED\n")


def test_learns_names():
    """Train on CharDataset, loss decreases."""
    print("=" * 60)
    print("test_learns_names")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    model = get_model("contrastive_router", vocab_size=tok.vocab_size,
                       block_size=32, n_embd=64, n_head=4, n_layer=2,
                       n_groups=4, n_capsules_per_group=64, top_k_groups=2,
                       d_key=8)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")
    assert final_loss < first_loss, \
        f"Loss didn't decrease: {first_loss:.4f} -> {final_loss:.4f}"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"

    print("  PASSED\n")


def test_infonce_training_improves_accuracy():
    """Training keys with InfoNCE on separable data improves routing accuracy."""
    print("=" * 60)
    print("test_infonce_training_improves_accuracy")
    import mlx.optimizers as optim

    # Create clearly separable hidden states
    mx.random.seed(42)
    n_per_domain = 100
    d = 32
    # Domain 0: positive mean, Domain 1: negative mean
    h_0 = mx.random.normal((n_per_domain, d)) + 1.0
    h_1 = mx.random.normal((n_per_domain, d)) - 1.0
    h_all = mx.concatenate([h_0, h_1], axis=0)
    labels = mx.concatenate([mx.zeros(n_per_domain, dtype=mx.int32),
                              mx.ones(n_per_domain, dtype=mx.int32)])

    pool = ContrastiveCapsulePool(d, n_groups=4, n_capsules_per_group=16,
                                  top_k_groups=2, d_key=8)
    mx.eval(pool.parameters())

    # Freeze groups, train only keys
    pool.freeze()
    for key in pool.routing_keys:
        key.unfreeze()

    acc_before = routing_accuracy(pool, h_all, labels, groups_per_domain=2)

    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(pool, lambda p, h, l:
        infonce_loss(p, h, l, groups_per_domain=2, tau=0.1))

    for step in range(100):
        loss, grads = loss_and_grad(pool, h_all, labels)
        optimizer.update(pool, grads)
        mx.eval(pool.parameters(), optimizer.state)

    acc_after = routing_accuracy(pool, h_all, labels, groups_per_domain=2)
    pool.unfreeze()

    print(f"  accuracy before: {acc_before:.1%}")
    print(f"  accuracy after:  {acc_after:.1%}")
    assert acc_after > acc_before, \
        f"Training should improve accuracy: {acc_before:.1%} -> {acc_after:.1%}"
    assert acc_after > 0.85, \
        f"On separable data, accuracy should reach >85%: got {acc_after:.1%}"

    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_shape()
    test_routing_scores_shape()
    test_routing_scores_nonnegative()
    test_infonce_loss_computable()
    test_infonce_gradient_flows()
    test_routing_accuracy_random()
    test_uniform_routing()
    test_extract_hidden_states()
    test_param_count()
    test_infonce_training_improves_accuracy()
    test_learns_names()
    print("All Contrastive Router tests passed!")

"""Tests for KD-Tree Routing model."""

import mlx.core as mx
import mlx.nn as nn


def test_kd_split_node_output_range():
    """Split node output should be in [0, 1] (sigmoid)."""
    from .kd_tree_routing import KDSplitNode
    split = KDSplitNode(64)
    mx.eval(split.parameters())
    x = mx.random.normal((2, 8, 64))
    p = split(x, temperature=1.0)
    mx.eval(p)
    assert p.shape == (2, 8, 1), f"Expected (2,8,1), got {p.shape}"
    p_flat = p.reshape(-1).tolist()
    assert all(0 <= v <= 1 for v in p_flat), "Split outputs outside [0,1]"


def test_kd_split_temperature_sharpens():
    """Higher temperature should produce more extreme probabilities."""
    from .kd_tree_routing import KDSplitNode
    split = KDSplitNode(64)
    mx.eval(split.parameters())
    x = mx.random.normal((2, 8, 64))

    p_soft = split(x, temperature=0.1)
    p_hard = split(x, temperature=10.0)
    mx.eval(p_soft, p_hard)

    # Variance of hard should be higher (more extreme values)
    var_soft = mx.var(p_soft).item()
    var_hard = mx.var(p_hard).item()
    assert var_hard > var_soft, f"Hard variance {var_hard} not > soft variance {var_soft}"


def test_kd_tree_shapes():
    """Tree output should match input shape."""
    from .kd_tree_routing import KDTreeCapsuleRouter
    tree = KDTreeCapsuleRouter(n_embd=64, depth=3,
                                n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    out = tree(x)
    mx.eval(out)
    assert out.shape == (2, 8, 64), f"Expected (2,8,64), got {out.shape}"


def test_leaf_probs_sum_to_one():
    """Leaf probability distribution should sum to ~1 for each token."""
    from .kd_tree_routing import KDTreeCapsuleRouter
    tree = KDTreeCapsuleRouter(n_embd=64, depth=3,
                                n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    _ = tree(x)
    leaf_probs = tree._leaf_probs  # (2, 8, 8)
    sums = mx.sum(leaf_probs, axis=-1)  # (2, 8)
    mx.eval(sums)
    for b in range(2):
        for t in range(8):
            s = sums[b, t].item()
            assert abs(s - 1.0) < 0.01, f"Leaf probs sum to {s}, expected ~1.0"


def test_n_leaves_correct():
    """Number of leaves should be 2^depth."""
    from .kd_tree_routing import KDTreeCapsuleRouter
    for depth in [1, 2, 3, 4]:
        tree = KDTreeCapsuleRouter(n_embd=32, depth=depth,
                                    n_capsules_per_leaf=8, beam_width=2)
        assert tree.n_leaves == 2 ** depth
        assert len(tree.leaves) == 2 ** depth
        assert len(tree.splits) == 2 ** depth - 1


def test_balance_loss_exists():
    """Balance loss should be a positive scalar."""
    from .kd_tree_routing import KDTreeCapsuleRouter
    tree = KDTreeCapsuleRouter(n_embd=64, depth=3,
                                n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    _ = tree(x)
    bl = tree.balance_loss()
    mx.eval(bl)
    assert bl.item() > 0, "Balance loss should be positive"


def test_full_model_forward():
    """Full KDTreeRoutingGPT forward pass should produce valid logits."""
    from .kd_tree_routing import KDTreeRoutingGPT
    model = KDTreeRoutingGPT(vocab_size=28, block_size=16,
                              n_embd=64, n_head=4, n_layer=2,
                              tree_depth=3, n_capsules_per_leaf=32,
                              beam_width=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (1, 8, 28), f"Expected (1,8,28), got {logits.shape}"


def test_param_count_reasonable():
    """Parameter count should be close to hierarchical_tree (~203K)."""
    from .kd_tree_routing import KDTreeRoutingGPT
    model = KDTreeRoutingGPT(vocab_size=28, block_size=32,
                              n_embd=64, n_head=4, n_layer=4,
                              tree_depth=3, n_capsules_per_leaf=32,
                              beam_width=2)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    # KD splits: 7 * (64 + 1) = 455 per layer (proj weight=64, threshold=1)
    # Same as hierarchical_tree: 7 * (64 + 1) = 455
    # Should be ~203K
    assert 195_000 < n_params < 215_000, f"Unexpected param count: {n_params}"


def test_aux_loss():
    """aux_loss should return a valid scalar."""
    from .kd_tree_routing import KDTreeRoutingGPT
    model = KDTreeRoutingGPT(vocab_size=28, block_size=16,
                              n_embd=64, n_head=4, n_layer=2,
                              tree_depth=3, n_capsules_per_leaf=32,
                              beam_width=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(tokens)
    al = model.aux_loss()
    mx.eval(al)
    assert al.item() >= 0, "Aux loss should be non-negative"


def test_temperature_annealing():
    """Temperature should increase over training steps."""
    from .kd_tree_routing import KDTreeRoutingGPT
    model = KDTreeRoutingGPT(vocab_size=28, block_size=16,
                              n_embd=64, n_head=4, n_layer=2,
                              tree_depth=3, n_capsules_per_leaf=32,
                              beam_width=2, init_temperature=1.0,
                              max_temperature=10.0)
    mx.eval(model.parameters())

    # At step 0 (warmup), temperature should be at init
    model.step_temperature(0, 500)
    t0 = model.layers[0].tree.temperature
    assert t0 == 1.0, f"Initial temperature should be 1.0, got {t0}"

    # At step 250 (50% training, past warmup), temperature should increase
    model.step_temperature(250, 500)
    t_mid = model.layers[0].tree.temperature
    assert t_mid > 1.0, f"Mid-training temperature should be >1.0, got {t_mid}"

    # At step 500, temperature should be near max
    model.step_temperature(500, 500)
    t_end = model.layers[0].tree.temperature
    assert t_end >= 9.0, f"End temperature should be near 10.0, got {t_end}"


def test_gradient_flows():
    """Verify gradients flow through KD split nodes and leaf capsules."""
    from .kd_tree_routing import KDTreeRoutingGPT
    model = KDTreeRoutingGPT(vocab_size=28, block_size=16,
                              n_embd=64, n_head=4, n_layer=2,
                              tree_depth=2, n_capsules_per_leaf=16,
                              beam_width=2)
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
        ) + model.aux_loss()

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 0]])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens, targets)
    mx.eval(loss, grads)

    # Check that split node gradients are non-zero
    split_grads = grads["layers"][0]["tree"]["splits"][0]["proj"]["weight"]
    mx.eval(split_grads)
    assert mx.any(split_grads != 0).item(), "Split gradients should be non-zero"

    # Check that leaf gradients are non-zero
    leaf_grads = grads["layers"][0]["tree"]["leaves"][0]["A"]["weight"]
    mx.eval(leaf_grads)
    assert mx.any(leaf_grads != 0).item(), "Leaf gradients should be non-zero"


def test_split_diversity_loss():
    """Split diversity loss should be non-negative."""
    from .kd_tree_routing import KDTreeCapsuleRouter
    tree = KDTreeCapsuleRouter(n_embd=64, depth=3,
                                n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    _ = tree(x)
    dl = tree.split_diversity_loss()
    mx.eval(dl)
    assert dl.item() >= 0, "Split diversity loss should be non-negative"


def test_routing_diagnostics():
    """Routing diagnostics should return expected structure."""
    from .kd_tree_routing import KDTreeRoutingGPT
    model = KDTreeRoutingGPT(vocab_size=28, block_size=16,
                              n_embd=64, n_head=4, n_layer=2,
                              tree_depth=3, n_capsules_per_leaf=32,
                              beam_width=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(tokens)
    mx.eval(model.parameters())  # force eval

    diag = model.get_routing_diagnostics()
    assert "layer_0" in diag
    assert "normalized_entropy" in diag["layer_0"]
    assert "split_directions" in diag["layer_0"]
    assert len(diag["layer_0"]["split_directions"]) == 7  # 2^3 - 1


if __name__ == "__main__":
    tests = [
        test_kd_split_node_output_range,
        test_kd_split_temperature_sharpens,
        test_kd_tree_shapes,
        test_leaf_probs_sum_to_one,
        test_n_leaves_correct,
        test_balance_loss_exists,
        test_full_model_forward,
        test_param_count_reasonable,
        test_aux_loss,
        test_temperature_annealing,
        test_gradient_flows,
        test_split_diversity_loss,
        test_routing_diagnostics,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")

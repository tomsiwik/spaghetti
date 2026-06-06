"""Tests for Hierarchical Capsule Tree model."""

import mlx.core as mx
import mlx.nn as nn


def test_tree_gate_output_range():
    """Gate output should be in [0, 1] (sigmoid)."""
    from .hierarchical_tree import TreeGate
    gate = TreeGate(64)
    mx.eval(gate.parameters())
    x = mx.random.normal((2, 8, 64))
    p = gate(x)
    assert p.shape == (2, 8, 1), f"Expected (2,8,1), got {p.shape}"
    p_val = p.tolist()
    # Flatten nested list
    flat = []
    for b in p_val:
        for t in b:
            flat.extend(t)
    assert all(0 <= v <= 1 for v in flat), "Gate outputs outside [0,1]"


def test_hierarchical_tree_shapes():
    """Tree output should match input shape."""
    from .hierarchical_tree import HierarchicalCapsuleTree
    tree = HierarchicalCapsuleTree(n_embd=64, depth=3,
                                    n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    out = tree(x)
    assert out.shape == (2, 8, 64), f"Expected (2,8,64), got {out.shape}"


def test_leaf_probs_sum_to_one():
    """Leaf probability distribution should sum to ~1 for each token."""
    from .hierarchical_tree import HierarchicalCapsuleTree
    tree = HierarchicalCapsuleTree(n_embd=64, depth=3,
                                    n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    _ = tree(x)
    leaf_probs = tree._leaf_probs  # (2, 8, 8)
    sums = mx.sum(leaf_probs, axis=-1)  # (2, 8)
    mx.eval(sums)
    # Each token's leaf probs should sum to 1 (product of binary decisions)
    for b in range(2):
        for t in range(8):
            s = sums[b, t].item()
            assert abs(s - 1.0) < 0.01, f"Leaf probs sum to {s}, expected ~1.0"


def test_n_leaves_correct():
    """Number of leaves should be 2^depth."""
    from .hierarchical_tree import HierarchicalCapsuleTree
    for depth in [1, 2, 3, 4]:
        tree = HierarchicalCapsuleTree(n_embd=32, depth=depth,
                                        n_capsules_per_leaf=8, beam_width=2)
        assert tree.n_leaves == 2 ** depth
        assert len(tree.leaves) == 2 ** depth
        assert len(tree.gates) == 2 ** depth - 1


def test_balance_loss_exists():
    """Balance loss should be a positive scalar."""
    from .hierarchical_tree import HierarchicalCapsuleTree
    tree = HierarchicalCapsuleTree(n_embd=64, depth=3,
                                    n_capsules_per_leaf=32, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 8, 64))
    _ = tree(x)
    bl = tree.balance_loss()
    mx.eval(bl)
    assert bl.item() > 0, "Balance loss should be positive"


def test_full_model_forward():
    """Full HierarchicalTreeGPT forward pass should produce valid logits."""
    from .hierarchical_tree import HierarchicalTreeGPT
    model = HierarchicalTreeGPT(vocab_size=28, block_size=16,
                                 n_embd=64, n_head=4, n_layer=2,
                                 tree_depth=3, n_capsules_per_leaf=32,
                                 beam_width=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model(tokens)
    assert logits.shape == (1, 8, 28), f"Expected (1,8,28), got {logits.shape}"


def test_param_count_reasonable():
    """Parameter count should be in expected range."""
    from .hierarchical_tree import HierarchicalTreeGPT
    model = HierarchicalTreeGPT(vocab_size=28, block_size=32,
                                 n_embd=64, n_head=4, n_layer=4,
                                 tree_depth=3, n_capsules_per_leaf=32,
                                 beam_width=2)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    # Should be close to capsule_moe params (~203K)
    # 8 leaves * 32 caps/leaf = 256 total capsules = same as flat G=4, P/G=64
    # Plus 7 internal gates * (64+1) = 455 params overhead per layer
    # So ~203K + 4*455 = ~205K
    assert 195_000 < n_params < 215_000, f"Unexpected param count: {n_params}"


def test_aux_loss():
    """aux_loss should return a valid scalar."""
    from .hierarchical_tree import HierarchicalTreeGPT
    model = HierarchicalTreeGPT(vocab_size=28, block_size=16,
                                 n_embd=64, n_head=4, n_layer=2,
                                 tree_depth=3, n_capsules_per_leaf=32,
                                 beam_width=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(tokens)
    al = model.aux_loss()
    mx.eval(al)
    assert al.item() >= 0, "Aux loss should be non-negative"


def test_gradient_flows():
    """Verify gradients flow through tree gates and leaf capsules."""
    from .hierarchical_tree import HierarchicalTreeGPT
    model = HierarchicalTreeGPT(vocab_size=28, block_size=16,
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

    # Check that gate gradients are non-zero
    gate_grads = grads["layers"][0]["tree"]["gates"][0]["proj"]["weight"]
    mx.eval(gate_grads)
    assert mx.any(gate_grads != 0).item(), "Gate gradients should be non-zero"

    # Check that leaf gradients are non-zero
    leaf_grads = grads["layers"][0]["tree"]["leaves"][0]["A"]["weight"]
    mx.eval(leaf_grads)
    assert mx.any(leaf_grads != 0).item(), "Leaf gradients should be non-zero"


if __name__ == "__main__":
    tests = [
        test_tree_gate_output_range,
        test_hierarchical_tree_shapes,
        test_leaf_probs_sum_to_one,
        test_n_leaves_correct,
        test_balance_loss_exists,
        test_full_model_forward,
        test_param_count_reasonable,
        test_aux_loss,
        test_gradient_flows,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")

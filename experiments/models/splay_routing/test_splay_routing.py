"""Tests for splay-tree adaptive routing."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn


def test_forward_shape():
    """Basic forward pass produces correct output shape."""
    from experiments.models.splay_routing.splay_routing import SplayTreeGPT
    mx.random.seed(42)
    model = SplayTreeGPT(vocab_size=28, block_size=32, n_embd=64, n_head=4,
                          n_layer=2, tree_depth=3, beam_width=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]] * 2)  # (2, 8)
    logits = model(tokens)
    assert logits.shape == (2, 8, 28), f"Expected (2, 8, 28), got {logits.shape}"
    print("PASS: forward shape")


def test_leaf_probs_sum_to_one():
    """Leaf probabilities should still sum to 1 even with splay bias."""
    from experiments.models.splay_routing.splay_routing import SplayCapsuleTree
    mx.random.seed(42)
    tree = SplayCapsuleTree(n_embd=64, depth=3, beam_width=2)
    mx.eval(tree.parameters())
    x = mx.random.normal((2, 4, 64))
    _ = tree(x)
    lp = tree._leaf_probs  # (2, 4, 8)
    sums = mx.sum(lp, axis=-1)  # (2, 4)
    max_err = mx.max(mx.abs(sums - 1.0)).item()
    assert max_err < 1e-5, f"Leaf probs don't sum to 1, max error: {max_err}"
    print(f"PASS: leaf probs sum to 1 (max err: {max_err:.2e})")


def test_splay_frequency_update():
    """After forward passes, leaf frequencies should diverge from uniform."""
    from experiments.models.splay_routing.splay_routing import SplayCapsuleTree
    mx.random.seed(42)
    tree = SplayCapsuleTree(n_embd=64, depth=3, beam_width=2, splay_decay=0.9)
    mx.eval(tree.parameters())

    # Initial state: uniform
    initial_freq = list(tree._leaf_freq)
    assert all(abs(f - 1/8) < 1e-6 for f in initial_freq), "Initial should be uniform"

    # Run several forward passes
    x = mx.random.normal((4, 8, 64))
    for _ in range(20):
        _ = tree(x)
        mx.eval(tree.parameters())

    # After training on same input, frequencies should be non-uniform
    final_freq = list(tree._leaf_freq)
    freq_range = max(final_freq) - min(final_freq)
    assert freq_range > 0.01, f"Frequencies should diverge, range={freq_range:.4f}"
    print(f"PASS: splay frequencies diverge (range: {freq_range:.4f})")


def test_splay_biases_nonzero():
    """After forward passes, gate splay biases should be non-zero."""
    from experiments.models.splay_routing.splay_routing import SplayCapsuleTree
    mx.random.seed(42)
    tree = SplayCapsuleTree(n_embd=64, depth=3, beam_width=2, splay_alpha=1.0)
    mx.eval(tree.parameters())

    x = mx.random.normal((4, 8, 64))
    for _ in range(10):
        _ = tree(x)
        mx.eval(tree.parameters())

    biases = [g._splay_bias for g in tree.gates]
    max_bias = max(abs(b) for b in biases)
    assert max_bias > 0.01, f"Splay biases should be non-zero, max={max_bias:.4f}"
    print(f"PASS: splay biases active (max: {max_bias:.4f})")


def test_reset_splay():
    """reset_splay should return state to uniform."""
    from experiments.models.splay_routing.splay_routing import SplayCapsuleTree
    mx.random.seed(42)
    tree = SplayCapsuleTree(n_embd=64, depth=3, beam_width=2)
    mx.eval(tree.parameters())

    x = mx.random.normal((4, 8, 64))
    for _ in range(10):
        _ = tree(x)
        mx.eval(tree.parameters())

    # Now reset
    tree.reset_splay()
    assert all(abs(f - 1/8) < 1e-6 for f in tree._leaf_freq), "Should be uniform after reset"
    assert all(g._splay_bias == 0.0 for g in tree.gates), "Biases should be zero after reset"
    print("PASS: reset_splay works")


def test_domain_switch():
    """on_domain_switch should reset splay state."""
    from experiments.models.splay_routing.splay_routing import SplayTreeGPT
    mx.random.seed(42)
    model = SplayTreeGPT(vocab_size=28, block_size=32, n_embd=64, n_head=4,
                          n_layer=2, tree_depth=3, beam_width=2)
    mx.eval(model.parameters())

    # Train on some data
    tokens = mx.array([[1, 2, 3, 4]] * 4)
    for _ in range(5):
        _ = model(tokens)
        mx.eval(model.parameters())

    # Check splay biases are non-zero
    diag = model.get_splay_diagnostics()
    has_nonzero = any(abs(b) > 0.001 for d in diag for b in d["gate_biases"])
    assert has_nonzero, "Should have non-zero biases before domain switch"

    # Domain switch
    model.on_domain_switch("new_domain")

    # All biases should be reset
    diag = model.get_splay_diagnostics()
    all_zero = all(b == 0.0 for d in diag for b in d["gate_biases"])
    assert all_zero, "All biases should be zero after domain switch"
    print("PASS: domain switch resets splay")


def test_aux_loss():
    """aux_loss should return a scalar."""
    from experiments.models.splay_routing.splay_routing import SplayTreeGPT
    mx.random.seed(42)
    model = SplayTreeGPT(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=2)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4]] * 2)
    _ = model(tokens)
    loss = model.aux_loss()
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0, "aux_loss should be non-negative"
    print(f"PASS: aux_loss = {loss.item():.6f}")


def test_param_count_matches_parent():
    """Splay tree should have identical param count to hierarchical tree
    (splay biases are non-parametric)."""
    from experiments.models.splay_routing.splay_routing import SplayTreeGPT
    from experiments.models.hierarchical_tree.hierarchical_tree import HierarchicalTreeGPT
    mx.random.seed(42)
    splay = SplayTreeGPT(vocab_size=28, n_embd=64, n_head=4, n_layer=4)
    parent = HierarchicalTreeGPT(vocab_size=28, n_embd=64, n_head=4, n_layer=4)
    mx.eval(splay.parameters())
    mx.eval(parent.parameters())

    n_splay = sum(v.size for _, v in nn.utils.tree_flatten(splay.parameters()))
    n_parent = sum(v.size for _, v in nn.utils.tree_flatten(parent.parameters()))
    assert n_splay == n_parent, f"Param count mismatch: splay={n_splay}, parent={n_parent}"
    print(f"PASS: param count matches parent ({n_splay})")


if __name__ == "__main__":
    test_forward_shape()
    test_leaf_probs_sum_to_one()
    test_splay_frequency_update()
    test_splay_biases_nonzero()
    test_reset_splay()
    test_domain_switch()
    test_aux_loss()
    test_param_count_matches_parent()
    print("\nAll tests passed!")

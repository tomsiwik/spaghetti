"""Tests for Huffman-shaped capsule tree."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn

from experiments.models.huffman_tree.huffman_tree import (
    build_huffman_tree,
    get_huffman_codes,
    huffman_expected_depth,
    count_internal_nodes,
    max_depth,
    HuffmanCapsuleTree,
    HuffmanTreeGPT,
)


def test_huffman_construction_uniform():
    """Uniform frequencies should produce a balanced tree."""
    freqs = [0.125] * 8
    root = build_huffman_tree(freqs)
    codes = get_huffman_codes(root)

    assert len(codes) == 8, f"Expected 8 codes, got {len(codes)}"
    # All depths should be 3 for 8 uniform leaves
    depths = [len(codes[i]) for i in range(8)]
    assert all(d == 3 for d in depths), f"Expected all depth 3, got {depths}"

    ed = huffman_expected_depth(freqs, codes)
    assert abs(ed - 3.0) < 1e-6, f"Expected E[depth]=3.0 for uniform, got {ed}"
    print("PASS: uniform frequencies -> balanced tree (depth 3)")


def test_huffman_construction_skewed():
    """Skewed frequencies should produce unbalanced tree with lower E[depth]."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    root = build_huffman_tree(freqs)
    codes = get_huffman_codes(root)

    assert len(codes) == 8

    # Most frequent leaf should have shortest path
    depths = {i: len(codes[i]) for i in range(8)}
    sorted_by_depth = sorted(depths.items(), key=lambda x: x[1])
    # Leaf 0 (freq=0.30) should be among the shallowest
    assert depths[0] <= 3, f"Most frequent leaf at depth {depths[0]}, expected <= 3"

    # Expected depth should be less than 3.0 (balanced)
    ed = huffman_expected_depth(freqs, codes)
    assert ed < 3.0, f"Expected E[depth] < 3.0, got {ed}"
    print(f"PASS: skewed frequencies -> E[depth]={ed:.4f} < 3.0 (balanced)")
    print(f"  Codes: {codes}")
    print(f"  Depths: {depths}")


def test_huffman_construction_extreme():
    """Extremely skewed: one leaf dominates."""
    freqs = [0.90, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
    root = build_huffman_tree(freqs)
    codes = get_huffman_codes(root)

    ed = huffman_expected_depth(freqs, codes)
    assert ed < 2.0, f"Expected E[depth] < 2.0 for extreme skew, got {ed}"
    print(f"PASS: extreme skew -> E[depth]={ed:.4f} (dominant leaf at depth {len(codes[0])})")


def test_leaf_probs_sum_to_one():
    """Leaf probabilities must sum to 1 for any tree shape."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    mx.random.seed(42)
    tree = HuffmanCapsuleTree(n_embd=64, n_leaves=8, n_capsules_per_leaf=4,
                               beam_width=2, frequencies=freqs)
    mx.eval(tree.parameters())

    x = mx.random.normal((2, 4, 64))
    _ = tree(x)

    leaf_probs = tree._leaf_probs  # (2, 4, 8)
    prob_sums = mx.sum(leaf_probs, axis=-1)  # (2, 4)

    assert mx.allclose(prob_sums, mx.ones_like(prob_sums), atol=1e-5).item(), \
        f"Leaf probs don't sum to 1: {prob_sums}"
    print("PASS: leaf probabilities sum to 1.0")


def test_output_shape():
    """Output shape must match input shape."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    mx.random.seed(42)
    tree = HuffmanCapsuleTree(n_embd=64, n_leaves=8, n_capsules_per_leaf=4,
                               beam_width=2, frequencies=freqs)
    mx.eval(tree.parameters())

    x = mx.random.normal((2, 4, 64))
    out = tree(x)

    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    print("PASS: output shape matches input")


def test_full_model_forward():
    """Full HuffmanTreeGPT forward pass."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    mx.random.seed(42)
    model = HuffmanTreeGPT(vocab_size=28, block_size=32,
                            n_embd=64, n_head=4, n_layer=4,
                            n_leaves=8, n_capsules_per_leaf=32,
                            beam_width=2, frequencies=freqs)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]] * 2)
    logits = model(tokens)

    assert logits.shape == (2, 5, 28), f"Expected (2, 5, 28), got {logits.shape}"
    print("PASS: full model forward pass")


def test_full_model_uniform_frequencies():
    """Model with uniform frequencies should work like balanced tree."""
    mx.random.seed(42)
    model = HuffmanTreeGPT(vocab_size=28, block_size=32,
                            n_embd=64, n_head=4, n_layer=4,
                            n_leaves=8, n_capsules_per_leaf=32,
                            beam_width=2, frequencies=None)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]] * 2)
    logits = model(tokens)

    assert logits.shape == (2, 5, 28)
    # With uniform frequencies, all codes should have depth 3
    for layer in model.layers:
        for i in range(8):
            depth = len(layer.tree.codes[i])
            assert depth == 3, f"Expected depth 3 for uniform, got {depth}"
    print("PASS: uniform frequencies produce balanced tree (depth 3)")


def test_avg_routing_depth():
    """Average routing depth should be trackable."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    mx.random.seed(42)
    model = HuffmanTreeGPT(vocab_size=28, block_size=32,
                            n_embd=64, n_head=4, n_layer=4,
                            n_leaves=8, n_capsules_per_leaf=32,
                            beam_width=2, frequencies=freqs)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]] * 2)
    _ = model(tokens)

    avg_depth = model.avg_routing_depth()
    assert avg_depth > 0, f"Expected positive avg depth, got {avg_depth}"
    print(f"PASS: avg routing depth = {avg_depth:.4f}")


def test_aux_loss():
    """Auxiliary loss should be computable."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    mx.random.seed(42)
    model = HuffmanTreeGPT(vocab_size=28, block_size=32, n_embd=64,
                            n_head=4, n_layer=4, n_leaves=8,
                            n_capsules_per_leaf=32, beam_width=2,
                            frequencies=freqs)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5]] * 2)
    _ = model(tokens)
    loss = model.aux_loss()
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    print(f"PASS: aux_loss = {loss.item():.6f}")


def test_param_count():
    """Verify parameter count is in expected range."""
    freqs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
    mx.random.seed(42)
    model = HuffmanTreeGPT(vocab_size=28, block_size=32, n_embd=64,
                            n_head=4, n_layer=4, n_leaves=8,
                            n_capsules_per_leaf=32, beam_width=2,
                            frequencies=freqs)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"PASS: param count = {n_params:,}")
    # Should be close to balanced tree (203,932)
    # Huffman has 7 internal nodes same as balanced D=3
    assert 200_000 < n_params < 210_000, f"Unexpected param count: {n_params}"


def test_internal_node_count():
    """N leaves -> N-1 internal nodes (fundamental binary tree property)."""
    for n in [2, 4, 8, 16]:
        freqs = [1.0 / n] * n
        root = build_huffman_tree(freqs)
        n_internal = count_internal_nodes(root)
        assert n_internal == n - 1, \
            f"Expected {n-1} internal nodes for {n} leaves, got {n_internal}"
    print("PASS: N leaves -> N-1 internal nodes")


if __name__ == "__main__":
    test_huffman_construction_uniform()
    test_huffman_construction_skewed()
    test_huffman_construction_extreme()
    test_internal_node_count()
    test_leaf_probs_sum_to_one()
    test_output_shape()
    test_full_model_forward()
    test_full_model_uniform_frequencies()
    test_avg_routing_depth()
    test_aux_loss()
    test_param_count()
    print("\n=== ALL TESTS PASSED ===")

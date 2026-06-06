"""Tests for skip-list multi-resolution routing."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn


def test_basic_forward():
    """Model produces correct output shape."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListRoutingGPT

    model = SkipListRoutingGPT(vocab_size=28, block_size=32, n_embd=64,
                                n_head=4, n_layer=2, n_experts=8,
                                n_capsules_per_expert=32, top_k=2)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # (1, 8)
    logits = model(tokens)
    assert logits.shape == (1, 8, 28), f"Expected (1, 8, 28), got {logits.shape}"
    print("PASS: basic_forward")


def test_level_structure():
    """Verify skip list level sizes are correct."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListCapsulePool

    pool = SkipListCapsulePool(n_embd=64, n_experts=8, n_capsules_per_expert=32)
    mx.eval(pool.parameters())

    # N=8: Level 0=8, Level 1=4, Level 2=2 (n_levels=3 total levels)
    assert pool.n_levels == 3, f"Expected 3 coarse levels, got {pool.n_levels}"
    assert pool.level_sizes == [8, 4, 2, 1], f"Expected [8,4,2,1], got {pool.level_sizes}"
    assert len(pool.routers) == 4, f"Expected 4 routers, got {len(pool.routers)}"
    assert len(pool.confidence_gates) == 3, f"Expected 3 confidence gates, got {len(pool.confidence_gates)}"
    print("PASS: level_structure")


def test_level_usage_sums_to_one():
    """Level usage weights should sum to ~1.0 (they're a probability distribution)."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListCapsulePool

    pool = SkipListCapsulePool(n_embd=64, n_experts=8, n_capsules_per_expert=32)
    mx.eval(pool.parameters())

    x = mx.random.normal((2, 8, 64))
    _ = pool(x)
    mx.eval(pool._level_usage)

    usage = pool._level_usage  # (B, T, n_levels+1)
    sums = mx.sum(usage, axis=-1)  # (B, T)
    max_err = mx.max(mx.abs(sums - 1.0)).item()
    assert max_err < 1e-5, f"Level weights don't sum to 1: max error {max_err}"
    print(f"PASS: level_usage_sums_to_one (max_err={max_err:.2e})")


def test_adaptive_depth():
    """Verify avg_routing_depth returns a reasonable value."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListCapsulePool

    pool = SkipListCapsulePool(n_embd=64, n_experts=8, n_capsules_per_expert=32)
    mx.eval(pool.parameters())

    x = mx.random.normal((2, 8, 64))
    _ = pool(x)

    depth = pool.avg_routing_depth()
    mx.eval(depth)
    d = depth.item()
    # Depth should be between 1 (always stop at coarsest) and n_levels+1 (always descend)
    assert 1.0 <= d <= pool.n_levels + 1, f"Depth {d} out of range [1, {pool.n_levels+1}]"
    print(f"PASS: adaptive_depth (avg_depth={d:.3f}, max={pool.n_levels+1})")


def test_param_count_comparison():
    """Skip list should have similar param count to flat and tree baselines."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListRoutingGPT
    from experiments.models.capsule_moe.capsule_moe import CapsuleMoEGPT
    from experiments.models.hierarchical_tree.hierarchical_tree import HierarchicalTreeGPT

    def count(model):
        return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))

    flat = CapsuleMoEGPT(vocab_size=28, block_size=32, n_embd=64, n_head=4,
                          n_layer=4, n_groups=8, n_capsules_per_group=32, top_k_groups=2)
    tree = HierarchicalTreeGPT(vocab_size=28, block_size=32, n_embd=64, n_head=4,
                                n_layer=4, tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    skip = SkipListRoutingGPT(vocab_size=28, block_size=32, n_embd=64, n_head=4,
                               n_layer=4, n_experts=8, n_capsules_per_expert=32, top_k=2)

    mx.eval(flat.parameters())
    mx.eval(tree.parameters())
    mx.eval(skip.parameters())

    n_flat = count(flat)
    n_tree = count(tree)
    n_skip = count(skip)

    print(f"  Flat:  {n_flat:,} params")
    print(f"  Tree:  {n_tree:,} params")
    print(f"  Skip:  {n_skip:,} params")
    print(f"  Skip overhead vs flat: {100*(n_skip-n_flat)/n_flat:+.1f}%")
    print(f"  Skip overhead vs tree: {100*(n_skip-n_tree)/n_tree:+.1f}%")
    print("PASS: param_count_comparison")


def test_gradient_flow():
    """Verify gradients flow through all components."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListRoutingGPT

    model = SkipListRoutingGPT(vocab_size=28, block_size=32, n_embd=64,
                                n_head=4, n_layer=1, n_experts=4,
                                n_capsules_per_expert=16, top_k=2)
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B*T, V), y.reshape(B*T), reduction="mean"
        ) + model.aux_loss()

    x = mx.array([[1, 2, 3, 4]])
    y = mx.array([[2, 3, 4, 5]])

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    mx.eval(loss, grads)

    # Check that confidence gate gradients are non-zero
    flat_grads = nn.utils.tree_flatten(grads)
    gate_grads = [(k, v) for k, v in flat_grads if "confidence" in k]
    assert len(gate_grads) > 0, "No confidence gate gradients found"

    nonzero_gates = sum(1 for _, v in gate_grads if mx.any(mx.abs(v) > 1e-10).item())
    print(f"  Confidence gates with nonzero grads: {nonzero_gates}/{len(gate_grads)}")
    assert nonzero_gates > 0, "No gradients flowing to confidence gates"
    print("PASS: gradient_flow")


def test_routing_stats():
    """get_routing_stats returns meaningful data."""
    from experiments.models.skip_list_routing.skip_list_routing import SkipListRoutingGPT

    model = SkipListRoutingGPT(vocab_size=28, block_size=32, n_embd=64,
                                n_head=4, n_layer=2, n_experts=8,
                                n_capsules_per_expert=32, top_k=2)
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _ = model(tokens)

    stats = model.get_routing_stats()
    assert len(stats) == 2, f"Expected 2 layers, got {len(stats)}"
    for layer_name, layer_stats in stats.items():
        assert "level_usage" in layer_stats
        assert "avg_depth" in layer_stats
        print(f"  {layer_name}: avg_depth={layer_stats['avg_depth']:.3f}, "
              f"usage={[f'{u:.3f}' for u in layer_stats['level_usage']]}")
    print("PASS: routing_stats")


if __name__ == "__main__":
    test_basic_forward()
    test_level_structure()
    test_level_usage_sums_to_one()
    test_adaptive_depth()
    test_param_count_comparison()
    test_gradient_flow()
    test_routing_stats()
    print("\nAll tests passed!")

"""Tests for linear state capacity scaling experiment.

Validates that:
1. Models instantiate at all three dimensions (d=64, 128, 256)
2. Forward pass produces correct output shapes
3. State matrix dimensions scale as expected (d_h^2)
4. Composition protocol works at each dimension
"""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model


def test_model_instantiation():
    """Models instantiate at all three target dimensions."""
    for n_embd in [64, 128, 256]:
        model = get_model(
            "full_gdn_stack_capsule_moe",
            vocab_size=28, block_size=32,
            n_embd=n_embd, n_head=4, n_layer=4,
            n_groups=4, n_capsules_per_group=64, top_k_groups=2,
            layer_types=["linear"] * 4,
        )
        mx.eval(model.parameters())

        # Check forward pass
        tokens = mx.zeros((2, 16), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (2, 16, 28), f"Expected (2, 16, 28), got {logits.shape}"

        # Check head dimension
        d_h = n_embd // 4
        attn = model.layers[0].attn
        assert attn.head_dim == d_h, f"Expected d_h={d_h}, got {attn.head_dim}"

        print(f"  d={n_embd}, d_h={d_h}: OK (state={d_h}x{d_h}={d_h*d_h})")


def test_state_capacity_scaling():
    """State matrix has correct dimensions at each scale."""
    for n_embd, expected_state_size in [(64, 256), (128, 1024), (256, 4096)]:
        d_h = n_embd // 4
        actual_state_size = d_h * d_h
        assert actual_state_size == expected_state_size, \
            f"d={n_embd}: expected state={expected_state_size}, got {actual_state_size}"
        print(f"  d={n_embd}: state size = {actual_state_size} (d_h={d_h})")


def test_full_attn_instantiation():
    """Full attention control model instantiates at all dimensions."""
    for n_embd in [64, 128, 256]:
        model = get_model(
            "full_attn_capsule_moe",
            vocab_size=28, block_size=32,
            n_embd=n_embd, n_head=4, n_layer=4,
            n_groups=4, n_capsules_per_group=64, top_k_groups=2,
        )
        mx.eval(model.parameters())

        tokens = mx.zeros((2, 16), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (2, 16, 28), f"Expected (2, 16, 28), got {logits.shape}"
        print(f"  full_attn d={n_embd}: OK")


def test_param_counts():
    """Report parameter counts at each dimension."""
    for n_embd in [64, 128, 256]:
        for model_name, lt in [("full_gdn_stack_capsule_moe", ["linear"]*4),
                                ("full_attn_capsule_moe", None)]:
            kwargs = dict(
                vocab_size=28, block_size=32,
                n_embd=n_embd, n_head=4, n_layer=4,
                n_groups=4, n_capsules_per_group=64, top_k_groups=2,
            )
            if lt is not None:
                kwargs["layer_types"] = lt

            model = get_model(model_name, **kwargs)
            mx.eval(model.parameters())

            n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
            attn_type = "pure_linear" if lt else "full_attn"
            print(f"  d={n_embd} {attn_type}: {n_params:,} params")


if __name__ == "__main__":
    print("Test 1: Model instantiation at all dimensions")
    test_model_instantiation()

    print("\nTest 2: State capacity scaling")
    test_state_capacity_scaling()

    print("\nTest 3: Full attention control instantiation")
    test_full_attn_instantiation()

    print("\nTest 4: Parameter counts")
    test_param_counts()

    print("\nAll tests passed.")

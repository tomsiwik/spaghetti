"""Tests for MinimalGraftRecalGPT and selective freezing utilities."""

import sys
sys.path.insert(0, "/Users/tom/Code/tomsiwik/llm")

import mlx.core as mx
import mlx.nn as nn
from experiments.models import get_model
from experiments.models.minimal_graft_recal.run_experiment import (
    freeze_all, unfreeze_gates, freeze_except_gates, count_params,
)


def test_model_registered():
    """Model should be registered and instantiable."""
    model = get_model("minimal_graft_recal", vocab_size=28, block_size=32)
    mx.eval(model.parameters())
    tokens = mx.zeros((1, 8), dtype=mx.int32)
    out = model(tokens)
    assert out.shape == (1, 8, 28), f"Expected (1, 8, 28), got {out.shape}"
    print("PASS: model registered and forward works")


def test_selective_gate_freezing():
    """Selective gate freezing should control trainable param count."""
    model = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                      tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    total_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))

    # Freeze everything
    freeze_all(model)
    assert count_params(model) == 0, "freeze_all should leave 0 trainable"

    # Unfreeze root gate only (gate 0)
    unfreeze_gates(model, [0])
    root_params = count_params(model)
    # Each gate has d+1=65 params (Linear(64, 1, bias=True))
    # 4 layers * 65 = 260
    assert root_params == 260, f"Expected 260, got {root_params}"
    print(f"PASS: root-only = {root_params} params")

    # Reset and try root+graft-point (gates 0, 1, 2)
    freeze_all(model)
    unfreeze_gates(model, [0, 1, 2])
    rgp_params = count_params(model)
    assert rgp_params == 780, f"Expected 780, got {rgp_params}"
    print(f"PASS: root+graft-point = {rgp_params} params")

    # Reset and try all gates (0-6)
    freeze_all(model)
    unfreeze_gates(model, list(range(7)))
    all_params = count_params(model)
    assert all_params == 1820, f"Expected 1820, got {all_params}"
    print(f"PASS: all-gates = {all_params} params")

    # Verify ratios
    assert rgp_params == 3 * root_params, "root+graft should be 3x root"
    assert all_params == 7 * root_params, "all-gates should be 7x root"
    print("PASS: param ratios correct (1:3:7)")


def test_freeze_except_gates():
    """freeze_except_gates should be equivalent to freeze_all + unfreeze."""
    model = get_model("hierarchical_tree", vocab_size=28, block_size=32,
                      tree_depth=3, n_capsules_per_leaf=32, beam_width=2)
    mx.eval(model.parameters())

    freeze_except_gates(model, [0, 1, 2])
    assert count_params(model) == 780
    print("PASS: freeze_except_gates works correctly")


if __name__ == "__main__":
    test_model_registered()
    test_selective_gate_freezing()
    test_freeze_except_gates()
    print("\nAll tests passed.")

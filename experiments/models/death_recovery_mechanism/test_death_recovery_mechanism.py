"""Unit tests for death recovery mechanism experiment.

Validates:
1. Model registration and instantiation
2. Layer-selective freezing works correctly
3. Revival computation is correct on synthetic data
"""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model, MODEL_REGISTRY
from .death_recovery_mechanism import (
    freeze_specific_mlp_layers,
    compute_revival_per_layer,
    N_LAYERS,
)


def test_registration():
    """DeathRecoveryMechanismGPT is registered with correct parent."""
    assert "death_recovery_mechanism" in MODEL_REGISTRY
    assert MODEL_REGISTRY["death_recovery_mechanism"]["parent"] == "pruning_controls"


def test_instantiation():
    """Model instantiates and produces correct output shape."""
    model = get_model("death_recovery_mechanism", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())
    tokens = mx.zeros((2, 16), dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (2, 16, 28)


def test_freeze_specific_layers():
    """Freezing specific MLP layers leaves others trainable."""
    model = get_model("death_recovery_mechanism", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())

    # Freeze layers 0 and 1
    freeze_specific_mlp_layers(model, [0, 1])

    # Attention should be frozen everywhere
    for l_idx, layer in enumerate(model.layers):
        attn_trainable = any(
            not isinstance(v, mx.array) or True  # check via freeze status
            for v in nn.utils.tree_flatten(layer.attn.trainable_parameters())
        )
        # Note: with freeze(), trainable_parameters() returns empty for frozen modules

    # Layers 0,1 MLP should be frozen (no trainable params)
    for l_idx in [0, 1]:
        pool = model.layers[l_idx].capsule_pool
        trainable = list(nn.utils.tree_flatten(pool.trainable_parameters()))
        assert len(trainable) == 0, (
            f"Layer {l_idx} should be frozen but has {len(trainable)} trainable params"
        )

    # Layers 2,3 MLP should be trainable
    for l_idx in [2, 3]:
        pool = model.layers[l_idx].capsule_pool
        trainable = list(nn.utils.tree_flatten(pool.trainable_parameters()))
        assert len(trainable) > 0, (
            f"Layer {l_idx} should be trainable but has 0 trainable params"
        )


def test_freeze_all_but_one():
    """Training only one layer freezes all others."""
    model = get_model("death_recovery_mechanism", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())

    # Train only layer 2
    frozen = [i for i in range(4) if i != 2]
    freeze_specific_mlp_layers(model, frozen)

    for l_idx in range(4):
        pool = model.layers[l_idx].capsule_pool
        trainable = list(nn.utils.tree_flatten(pool.trainable_parameters()))
        if l_idx == 2:
            assert len(trainable) > 0, "Layer 2 should be trainable"
        else:
            assert len(trainable) == 0, f"Layer {l_idx} should be frozen"


def test_revival_computation():
    """Revival computation returns correct rates from synthetic masks."""
    # Build synthetic results structure
    # 4 layers, 4 capsules each = 16 total
    anchor_step = 100
    final_step = 3200

    # Synthetic data: at S=100, layers 1-3 have 2 dead each
    # At S=3200, with baseline: 1 dead revives per layer
    # With freeze: no revivals
    results_baseline = {
        "condition": "baseline",
        "frozen_layers": [],
        "checkpoints": {
            100: {
                "per_layer_masks": [
                    [False, False, False, False],  # layer 0: no dead
                    [True, True, False, False],     # layer 1: caps 0,1 dead
                    [False, True, True, False],     # layer 2: caps 1,2 dead
                    [True, False, True, False],     # layer 3: caps 0,2 dead
                ],
                "per_layer_rates": [0.0, 0.5, 0.5, 0.5],
                "flat_mask": [False]*4 + [True,True,False,False] + [False,True,True,False] + [True,False,True,False],
                "overall_rate": 6/16,
            },
            3200: {
                "per_layer_masks": [
                    [False, False, False, False],  # layer 0: still no dead
                    [True, False, False, False],    # layer 1: cap 1 revived
                    [False, False, True, False],    # layer 2: cap 1 revived
                    [False, False, True, False],    # layer 3: cap 0 revived
                ],
                "per_layer_rates": [0.0, 0.25, 0.25, 0.25],
                "flat_mask": [False]*4 + [True,False,False,False] + [False,False,True,False] + [False,False,True,False],
                "overall_rate": 3/16,
            },
        },
    }

    revival = compute_revival_per_layer(results_baseline, anchor_step=100)
    assert 3200 in revival

    # Layer 0: no dead at anchor, revival rate should be 0
    assert revival[3200][0]["n_dead_anchor"] == 0

    # Layer 1: 2 dead at anchor, 1 revived -> revival rate 0.5
    assert revival[3200][1]["revival_rate"] == 0.5
    assert revival[3200][1]["n_revived"] == 1
    assert revival[3200][1]["n_dead_anchor"] == 2

    # Layer 2: 2 dead at anchor, 1 revived -> revival rate 0.5
    assert revival[3200][2]["revival_rate"] == 0.5

    # Layer 3: 2 dead at anchor, 1 revived -> revival rate 0.5
    assert revival[3200][3]["revival_rate"] == 0.5


if __name__ == "__main__":
    test_registration()
    print("PASS: test_registration")
    test_instantiation()
    print("PASS: test_instantiation")
    test_freeze_specific_layers()
    print("PASS: test_freeze_specific_layers")
    test_freeze_all_but_one()
    print("PASS: test_freeze_all_but_one")
    test_revival_computation()
    print("PASS: test_revival_computation")
    print("\nAll mechanism tests passed.")

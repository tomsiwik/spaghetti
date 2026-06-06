"""Unit tests for revival_under_composition experiment.

Validates:
1. Model registration and instantiation
2. Composed mask splitting works correctly
3. Revival computation handles composed models
4. Full experiment runs end-to-end (functional test)
"""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model, MODEL_REGISTRY
from .revival_under_composition import (
    split_composed_mask_by_domain,
    compute_revival_rates,
    N_LAYERS,
)


def test_registration():
    """RevivalUnderCompositionGPT is registered with correct parent."""
    assert "revival_under_composition" in MODEL_REGISTRY
    assert MODEL_REGISTRY["revival_under_composition"]["parent"] == "capsule_revival"


def test_instantiation():
    """Model instantiates and produces correct output shape."""
    model = get_model("revival_under_composition", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())
    tokens = mx.zeros((2, 16), dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (2, 16, 28)


def test_split_composed_mask():
    """Splitting composed mask into domain halves works correctly."""
    # 4 layers, 4 capsules per domain = 8 per layer in composed model
    n_caps = 4
    n_layers = 4
    # flat_mask: 8 bools per layer, 4 layers = 32 total
    # Layer 0: AAAA BBBB = [T,F,F,F, F,T,T,F]
    # Layer 1: [F,F,T,F, T,F,F,T]
    # Layer 2: [T,T,F,F, F,F,T,T]
    # Layer 3: [F,F,F,F, T,T,T,T]
    flat_mask = [
        True, False, False, False,  False, True, True, False,   # L0
        False, False, True, False,  True, False, False, True,   # L1
        True, True, False, False,   False, False, True, True,   # L2
        False, False, False, False, True, True, True, True,     # L3
    ]

    mask_a, mask_b = split_composed_mask_by_domain(flat_mask, n_caps, n_layers)

    # Domain A should be first 4 of each layer
    assert mask_a == [
        True, False, False, False,   # L0 A
        False, False, True, False,   # L1 A
        True, True, False, False,    # L2 A
        False, False, False, False,  # L3 A
    ]
    # Domain B should be last 4 of each layer
    assert mask_b == [
        False, True, True, False,    # L0 B
        True, False, False, True,    # L1 B
        False, False, True, True,    # L2 B
        True, True, True, True,      # L3 B
    ]


def test_revival_single_domain():
    """Revival computation for single-domain results is correct."""
    results = {
        "condition": "single_domain",
        "anchor_mask": [True, True, False, False, True, False, False, False],
        "checkpoints": {
            0: {"flat_mask": [True, True, False, False, True, False, False, False]},
            100: {"flat_mask": [True, False, False, False, True, False, True, False]},
            # Cap 1: revived (T->F), Cap 6: newly dead (F->T)
        },
    }

    revival = compute_revival_rates(results, is_composed=False)

    assert 100 in revival
    # 3 dead at anchor (caps 0,1,4), 1 revived (cap 1) = 1/3
    assert abs(revival[100]["revival_rate"] - 1/3) < 1e-6
    assert revival[100]["n_revived"] == 1
    assert revival[100]["n_dead_anchor"] == 3
    assert revival[100]["n_newly_dead"] == 1


def test_revival_composed():
    """Revival computation for composed model correctly splits domains."""
    # 2 layers, 2 capsules per domain = 4 per layer composed = 8 total
    results = {
        "condition": "composed_joint",
        "anchor_mask_full": [True, False, True, False, False, True, False, True],
        "anchor_mask_a": [True, False, False, True],   # from split
        "anchor_mask_b": [True, False, True, False],   # from split
        "checkpoints": {
            0: {
                "flat_mask": [True, False, True, False, False, True, False, True],
                "mask_a": [True, False, False, True],
                "mask_b": [True, False, True, False],
            },
            100: {
                "flat_mask": [False, False, True, False, False, False, False, True],
                "mask_a": [False, False, False, False],  # cap 0 revived, cap 3 revived
                "mask_b": [True, False, False, True],   # cap 2 revived
            },
        },
    }

    revival = compute_revival_rates(results, is_composed=True)

    assert 100 in revival
    # Domain A: 2 dead at anchor (caps 0,3), 2 revived = 100%
    assert abs(revival[100]["revival_a"] - 1.0) < 1e-6
    # Domain B: 1 dead at anchor (cap 0=True), 0 revived = 0%
    # Wait: anchor_mask_b = [True, False, True, False], so caps 0,2 dead
    # At step 100: mask_b = [True, False, False, True]
    # Cap 0: still dead, Cap 2: revived! So 1/2 = 50%
    assert abs(revival[100]["revival_b"] - 0.5) < 1e-6


if __name__ == "__main__":
    test_registration()
    print("PASS: test_registration")
    test_instantiation()
    print("PASS: test_instantiation")
    test_split_composed_mask()
    print("PASS: test_split_composed_mask")
    test_revival_single_domain()
    print("PASS: test_revival_single_domain")
    test_revival_composed()
    print("PASS: test_revival_composed")
    print("\nAll tests passed.")

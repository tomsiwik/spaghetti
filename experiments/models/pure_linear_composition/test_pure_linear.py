"""Tests for pure-linear composition (4:0 all-linear layers).

Validates that the full_gdn_stack model works correctly when all 4 layers
use GatedDeltaNet linear attention (no full attention scaffolding).
"""

import mlx.core as mx
import mlx.nn as nn

from experiments.models import get_model
from experiments.data import load_names, CharTokenizer, CharDataset
from experiments.train import train


# Small config for fast tests
BASE = dict(vocab_size=28, block_size=32, n_embd=32, n_head=4, n_layer=4)
CAP = dict(**BASE, n_groups=4, n_capsules_per_group=32, top_k_groups=2)


def test_pure_linear_forward():
    """All-linear model produces correct output shapes."""
    print("=" * 60)
    print("test_pure_linear_forward")

    model = get_model("full_gdn_stack_capsule_moe",
                      **CAP, layer_types=["linear"] * 4)
    assert model.layer_types == ["linear", "linear", "linear", "linear"]

    tokens = mx.zeros((2, 32), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (2, 32, 28), f"Expected (2,32,28), got {logits.shape}"
    print(f"  shape: {logits.shape}  OK")
    print(f"  layer_types: {model.layer_types}")
    print("  PASSED\n")


def test_pure_linear_causal():
    """All-linear model is causal: future tokens don't influence past."""
    print("=" * 60)
    print("test_pure_linear_causal")

    model = get_model("full_gdn_stack_capsule_moe",
                      **CAP, layer_types=["linear"] * 4)

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8] + [0] * 24])
    logits_orig = model(tokens)
    mx.eval(logits_orig)

    tokens_perturbed = mx.array([[1, 2, 3, 4, 5, 20, 7, 8] + [0] * 24])
    logits_perturbed = model(tokens_perturbed)
    mx.eval(logits_perturbed)

    for t in range(5):
        diff = mx.max(mx.abs(logits_orig[0, t] - logits_perturbed[0, t])).item()
        assert diff < 1e-3, f"Position {t} changed (max diff={diff}) when future token modified"
        print(f"  position {t}: diff={diff:.6f}  OK")

    diff_at_5 = mx.max(mx.abs(logits_orig[0, 5] - logits_perturbed[0, 5])).item()
    assert diff_at_5 > 0.0, "Position 5 should change"
    print(f"  position 5: diff={diff_at_5:.6f} (changed as expected)  OK")
    print("  PASSED\n")


def test_pure_linear_learns():
    """All-linear model can learn the names dataset."""
    print("=" * 60)
    print("test_pure_linear_learns")

    docs = load_names()
    tok = CharTokenizer(docs)
    ds = CharDataset(docs[:500], tok, block_size=32)

    cfg = dict(vocab_size=tok.vocab_size, block_size=32,
               n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2,
               layer_types=["linear"] * 4)
    model = get_model("full_gdn_stack_capsule_moe", **cfg)
    result = train(model, ds, steps=200, batch_size=32, lr=3e-3, log_every=100)

    first_loss = result["losses"][0]
    final_loss = result["final_loss"]
    print(f"  pure_linear: {first_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < first_loss, "Loss didn't decrease"
    assert final_loss < 3.0, f"Final loss too high: {final_loss:.4f}"
    print("  PASSED\n")


def test_param_count():
    """Pure-linear and hybrid have similar param counts."""
    print("=" * 60)
    print("test_param_count")

    cfg = dict(vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4,
               n_groups=4, n_capsules_per_group=64, top_k_groups=2)

    hybrid = get_model("full_gdn_stack_capsule_moe", **cfg,
                       layer_types=["linear", "linear", "linear", "full"])
    pure = get_model("full_gdn_stack_capsule_moe", **cfg,
                     layer_types=["linear"] * 4)

    n_hybrid = sum(v.size for _, v in nn.utils.tree_flatten(hybrid.parameters()))
    n_pure = sum(v.size for _, v in nn.utils.tree_flatten(pure.parameters()))

    print(f"  hybrid (3:1): {n_hybrid:,} params")
    print(f"  pure (4:0):   {n_pure:,} params")
    print(f"  ratio (pure/hybrid): {n_pure / n_hybrid:.4f}")

    # Pure-linear has one extra GDN layer instead of CausalSelfAttention.
    # GDN has more params (conv1d, beta, z projections), so pure > hybrid.
    assert 0.90 < n_pure / n_hybrid < 1.20, f"Unexpected param ratio: {n_pure / n_hybrid:.4f}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_pure_linear_forward()
    test_pure_linear_causal()
    test_param_count()
    test_pure_linear_learns()
    print("All pure-linear composition tests passed!")

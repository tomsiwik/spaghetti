"""Tests for Exp 4: N=5 expert scaling."""

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, domain_split, train_val_split, CharDataset
from experiments.models import get_model
from experiments.train import evaluate


def test_quintary_split():
    """domain_split('quintary') produces 5 non-empty, non-overlapping domains."""
    docs = load_names()
    splits = domain_split(docs, method="quintary")

    assert len(splits) == 5
    expected_keys = {"a_e", "f_j", "k_o", "p_t", "u_z"}
    assert set(splits.keys()) == expected_keys

    # Non-empty
    for name, ddocs in splits.items():
        assert len(ddocs) > 0, f"Domain {name} is empty"

    # Non-overlapping and complete
    total = sum(len(d) for d in splits.values())
    assert total == len(docs), f"Splits don't cover all docs: {total} vs {len(docs)}"

    # Check letter ranges
    for doc in splits["a_e"]:
        assert "a" <= doc[0].lower() <= "e"
    for doc in splits["u_z"]:
        assert "u" <= doc[0].lower() <= "z"


def test_composed_model_shape():
    """Composed model with G=20 produces correct output shapes."""
    model = get_model("capsule_moe", vocab_size=28,
                      n_groups=20, n_capsules_per_group=64,
                      top_k_groups=10, **dict(n_embd=64, n_head=4, n_layer=4, block_size=32))
    mx.eval(model.parameters())

    tokens = mx.array([[0, 1, 2, 3, 4]])  # (1, 5)
    logits = model(tokens)
    assert logits.shape == (1, 5, 28)


def test_composition_preserves_weights():
    """compose_from_shared_base_n correctly slots domain weights."""
    from .run_experiment import compose_from_shared_base_n, G_PER_DOMAIN

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    base = get_model("capsule_moe", vocab_size=V,
                     n_groups=G_PER_DOMAIN, n_capsules_per_group=64,
                     top_k_groups=2, n_embd=64, n_head=4, n_layer=4, block_size=32)
    mx.eval(base.parameters())

    # Create 2 "domain" models with known different weights
    d1 = get_model("capsule_moe", vocab_size=V,
                   n_groups=G_PER_DOMAIN, n_capsules_per_group=64,
                   top_k_groups=2, n_embd=64, n_head=4, n_layer=4, block_size=32)
    d2 = get_model("capsule_moe", vocab_size=V,
                   n_groups=G_PER_DOMAIN, n_capsules_per_group=64,
                   top_k_groups=2, n_embd=64, n_head=4, n_layer=4, block_size=32)
    mx.eval(d1.parameters())
    mx.eval(d2.parameters())

    # Extract groups
    groups = []
    for d_model in [d1, d2]:
        d_groups = []
        for l_idx in range(4):
            d_groups.append(d_model.layers[l_idx].capsule_pool.groups)
        groups.append(d_groups)

    composed = compose_from_shared_base_n(base, groups, V, 32)

    # Check that composed groups match domain groups
    for l_idx in range(4):
        pool = composed.layers[l_idx].capsule_pool
        assert pool.n_groups == 8
        # First 4 groups from d1
        for g in range(4):
            diff_a = mx.sum(mx.abs(
                pool.groups[g].A.weight - d1.layers[l_idx].capsule_pool.groups[g].A.weight
            )).item()
            assert diff_a < 1e-6, f"Layer {l_idx} group {g} A weight mismatch"
        # Next 4 groups from d2
        for g in range(4):
            diff_a = mx.sum(mx.abs(
                pool.groups[4 + g].A.weight - d2.layers[l_idx].capsule_pool.groups[g].A.weight
            )).item()
            assert diff_a < 1e-6, f"Layer {l_idx} group {4+g} A weight mismatch"

    # Check attention from base
    for l_idx in range(4):
        diff = mx.sum(mx.abs(
            composed.layers[l_idx].attn.wq.weight - base.layers[l_idx].attn.wq.weight
        )).item()
        assert diff < 1e-6, f"Layer {l_idx} attention not from base"


def test_orthogonality_computation():
    """compute_delta_orthogonality returns valid cosine similarities."""
    from .run_experiment import compute_delta_orthogonality

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    base = get_model("capsule_moe", vocab_size=V,
                     n_groups=4, n_capsules_per_group=64,
                     top_k_groups=2, n_embd=64, n_head=4, n_layer=4, block_size=32)
    mx.eval(base.parameters())

    # Create 3 slightly different domain models
    domain_models = []
    for i in range(3):
        mx.random.seed(i * 100)
        dm = get_model("capsule_moe", vocab_size=V,
                       n_groups=4, n_capsules_per_group=64,
                       top_k_groups=2, n_embd=64, n_head=4, n_layer=4, block_size=32)
        mx.eval(dm.parameters())
        domain_models.append(dm)

    ortho = compute_delta_orthogonality(base, domain_models)

    assert "mean" in ortho
    assert "max" in ortho
    assert "per_layer" in ortho
    assert len(ortho["per_layer"]) == 4  # n_layers
    # 3 domains -> 3 pairs per layer
    assert len(ortho["per_layer"][0]) == 3
    # Cosine sims should be in [-1, 1]
    for sim in ortho["all_sims"]:
        assert -1.01 <= sim <= 1.01, f"Invalid cosine similarity: {sim}"


if __name__ == "__main__":
    test_quintary_split()
    print("PASS: test_quintary_split")
    test_composed_model_shape()
    print("PASS: test_composed_model_shape")
    test_composition_preserves_weights()
    print("PASS: test_composition_preserves_weights")
    test_orthogonality_computation()
    print("PASS: test_orthogonality_computation")
    print("\nAll tests passed!")

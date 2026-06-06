"""Tests for Flat MoE N=8 Boundary experiment (revised).

Test 1: Smoke test -- verify octonary split produces 8 non-empty domains
Test 2: Verify N=8 domain composition works and has correct capsule count
Test 3: Full experiment -- run across 3 seeds at N=2, N=5, N=8
"""

import mlx.core as mx

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from ..relu_router.test_composition import (
    _make_relu_model, BASE, N_CAPSULES,
)
from ..n5_identity_scaling.n5_identity_scaling import compose_n_domains
from .flat_moe_n8_boundary import main, run_single_seed, run_post_cal_jaccard


def test_octonary_split():
    """Verify octonary split produces 8 non-empty domains."""
    docs = load_names()
    splits = domain_split(docs, method="octonary")
    assert len(splits) == 8, f"Expected 8 domains, got {len(splits)}"
    total = 0
    for name, docs_list in splits.items():
        assert len(docs_list) > 0, f"Empty domain: {name}"
        total += len(docs_list)
        print(f"  {name}: {len(docs_list)} names ({len(docs_list)/len(docs)*100:.1f}%)")
    assert total == len(docs), f"Domain split lost names: {total} vs {len(docs)}"
    print(f"  Total: {total} (all accounted for)")


def test_compose_n8():
    """Verify N=8 composition produces correct capsule count."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    models = [_make_relu_model(V, n_capsules=N_CAPSULES) for _ in range(8)]
    base = models[0]

    composed = compose_n_domains(base, models)
    expected_caps = N_CAPSULES * 8
    actual_caps = composed.layers[0].capsule_pool.n_capsules
    assert actual_caps == expected_caps, \
        f"Expected {expected_caps} capsules, got {actual_caps}"
    print(f"  N=8: {actual_caps} capsules (OK)")


def test_full_experiment():
    """Run the full revised experiment: 3 seeds at N=2, N=5, N=8."""
    main()


if __name__ == "__main__":
    print("=== test_octonary_split ===")
    test_octonary_split()
    print("\n=== test_compose_n8 ===")
    test_compose_n8()
    print("\n=== test_full_experiment ===")
    test_full_experiment()

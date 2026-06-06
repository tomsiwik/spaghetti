"""Tests for N=5 Identity Scaling experiment.

Test 1: Smoke test -- verify the experiment runs and produces expected structure
Test 2: Full experiment -- run across 3 seeds and report
"""

import mlx.core as mx

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from ..relu_router.test_composition import (
    _make_relu_model, BASE, N_CAPSULES,
)
from ..capsule_identity.capsule_identity import jaccard_similarity, overlap_coefficient
from .n5_identity_scaling import (
    compose_n_domains,
    decompose_n_domain,
    run_n5_experiment,
    main,
)


def test_quintary_split():
    """Verify quintary split produces 5 non-empty domains."""
    docs = load_names()
    splits = domain_split(docs, method="quintary")
    assert len(splits) == 5, f"Expected 5 domains, got {len(splits)}"
    for name, docs_list in splits.items():
        assert len(docs_list) > 0, f"Empty domain: {name}"
        print(f"  {name}: {len(docs_list)} names")


def test_compose_n_domains():
    """Verify that N-domain composition produces correct capsule count."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    models = [_make_relu_model(V, n_capsules=N_CAPSULES) for _ in range(5)]
    base = models[0]

    for n in [2, 3, 5]:
        composed = compose_n_domains(base, models[:n])
        expected_caps = N_CAPSULES * n
        actual_caps = composed.layers[0].capsule_pool.n_capsules
        assert actual_caps == expected_caps, \
            f"N={n}: expected {expected_caps} capsules, got {actual_caps}"
        print(f"  N={n}: {actual_caps} capsules (OK)")


def test_decompose_n_domain():
    """Test N-domain decomposition with known sets."""
    # 3 domains, P=4, 1 layer
    single_dead = {
        "d0": {(0, 0), (0, 1)},          # capsules 0,1 dead in d0
        "d1": {(0, 2), (0, 3)},          # capsules 2,3 dead in d1
        "d2": {(0, 0), (0, 3)},          # capsules 0,3 dead in d2
    }
    # Composed: d0 at [0..3], d1 at [4..7], d2 at [8..11]
    # All single-domain dead capsules also dead in composed, plus some extras
    composed_dead = {
        (0, 0), (0, 1),   # d0's 0,1
        (0, 6), (0, 7),   # d1's 2,3 (offset by 4)
        (0, 8), (0, 11),  # d2's 0,3 (offset by 8)
        (0, 3),           # extra: d0's capsule 3 killed by composition
    }

    result = decompose_n_domain(single_dead, composed_dead, 4, ["d0", "d1", "d2"])

    assert result["per_domain"]["d0"]["jaccard"] == jaccard_similarity(
        {(0, 0), (0, 1)}, {(0, 0), (0, 1), (0, 3)}
    )
    assert result["per_domain"]["d1"]["jaccard"] == 1.0  # identical sets
    assert result["per_domain"]["d2"]["jaccard"] == 1.0  # identical sets
    print(f"  d0 Jaccard: {result['per_domain']['d0']['jaccard']:.3f}")
    print(f"  d1 Jaccard: {result['per_domain']['d1']['jaccard']:.3f}")
    print(f"  d2 Jaccard: {result['per_domain']['d2']['jaccard']:.3f}")
    print(f"  Combined: {result['combined_jaccard']:.3f}")


def test_full_experiment():
    """Run the full 3-seed experiment and report results."""
    main()


if __name__ == "__main__":
    print("=== test_quintary_split ===")
    test_quintary_split()
    print("\n=== test_compose_n_domains ===")
    test_compose_n_domains()
    print("\n=== test_decompose_n_domain ===")
    test_decompose_n_domain()
    print("\n=== test_full_experiment ===")
    test_full_experiment()

"""Tests for N=8 Identity Boundary experiment.

Test 1: Smoke test -- verify octonary split produces 8 non-empty domains
Test 2: Verify N-domain composition works for N=8
Test 3: Full experiment -- run across 3 seeds and report
"""

import mlx.core as mx

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from ..relu_router.test_composition import (
    _make_relu_model, BASE, N_CAPSULES,
)
from ..capsule_identity.capsule_identity import jaccard_similarity, overlap_coefficient
from ..n5_identity_scaling.n5_identity_scaling import (
    compose_n_domains,
    decompose_n_domain,
)
from .n8_identity_boundary import (
    run_n8_experiment,
    main,
)


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


def test_compose_n8_domains():
    """Verify that 8-domain composition produces correct capsule count."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    models = [_make_relu_model(V, n_capsules=N_CAPSULES) for _ in range(8)]
    base = models[0]

    for n in [2, 4, 8]:
        composed = compose_n_domains(base, models[:n])
        expected_caps = N_CAPSULES * n
        actual_caps = composed.layers[0].capsule_pool.n_capsules
        assert actual_caps == expected_caps, \
            f"N={n}: expected {expected_caps} capsules, got {actual_caps}"
        print(f"  N={n}: {actual_caps} capsules (OK)")


def test_decompose_n8_domain():
    """Test N=8 domain decomposition with known sets."""
    # 4 domains for quick test, P=4, 1 layer
    single_dead = {
        "d0": {(0, 0), (0, 1)},
        "d1": {(0, 2), (0, 3)},
        "d2": {(0, 0), (0, 3)},
        "d3": {(0, 1), (0, 2)},
    }
    # Composed: d0 at [0..3], d1 at [4..7], d2 at [8..11], d3 at [12..15]
    composed_dead = {
        (0, 0), (0, 1),    # d0
        (0, 6), (0, 7),    # d1's 2,3
        (0, 8), (0, 11),   # d2's 0,3
        (0, 13), (0, 14),  # d3's 1,2
        (0, 3),            # extra: d0's capsule 3 killed
    }

    result = decompose_n_domain(single_dead, composed_dead, 4, ["d0", "d1", "d2", "d3"])

    assert result["per_domain"]["d1"]["jaccard"] == 1.0
    assert result["per_domain"]["d2"]["jaccard"] == 1.0
    assert result["per_domain"]["d3"]["jaccard"] == 1.0
    # d0 has extra kill (capsule 3)
    assert result["per_domain"]["d0"]["jaccard"] < 1.0

    print(f"  d0 Jaccard: {result['per_domain']['d0']['jaccard']:.3f}")
    print(f"  d1 Jaccard: {result['per_domain']['d1']['jaccard']:.3f}")
    print(f"  d2 Jaccard: {result['per_domain']['d2']['jaccard']:.3f}")
    print(f"  d3 Jaccard: {result['per_domain']['d3']['jaccard']:.3f}")
    print(f"  Combined: {result['combined_jaccard']:.3f}")


def test_full_experiment():
    """Run the full 3-seed experiment and report results."""
    main()


if __name__ == "__main__":
    print("=== test_octonary_split ===")
    test_octonary_split()
    print("\n=== test_compose_n8_domains ===")
    test_compose_n8_domains()
    print("\n=== test_decompose_n8_domain ===")
    test_decompose_n8_domain()
    print("\n=== test_full_experiment ===")
    test_full_experiment()

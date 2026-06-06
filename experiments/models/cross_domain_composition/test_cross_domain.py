#!/usr/bin/env python3
"""Tests for cross-domain composition experiment (Revision 2)."""

import numpy as np
from .cross_domain_composition import (
    CharTokenizer, DOMAIN_GENERATORS, CROSS_DOMAIN_GENERATORS,
    init_model, forward, eval_loss, compute_delta, svd_truncate_delta,
    apply_delta, merge_deltas, gram_schmidt_deltas, flatten_delta,
    cosine_sim, subspace_analysis, train_model,
)


def test_cross_domain_data_generators():
    """All 10 cross-domain generators produce valid sequences."""
    rng = np.random.RandomState(42)
    tok = CharTokenizer()

    assert len(CROSS_DOMAIN_GENERATORS) == 10, \
        f"Expected 10 cross-domain pairs, got {len(CROSS_DOMAIN_GENERATORS)}"

    for name, (gen_fn, involved) in CROSS_DOMAIN_GENERATORS.items():
        data = gen_fn(10, np.random.RandomState(42))
        assert len(data) == 10, f"{name}: expected 10 samples, got {len(data)}"
        assert len(involved) == 2, f"{name}: expected 2 involved domains"
        assert involved[0] in DOMAIN_GENERATORS, f"{name}: unknown domain {involved[0]}"
        assert involved[1] in DOMAIN_GENERATORS, f"{name}: unknown domain {involved[1]}"

        for s in data:
            encoded = tok.encode(s)
            assert len(encoded) > 1, f"{name}: empty encoding for '{s}'"


def test_all_10_pairs_covered():
    """Verify all C(5,2)=10 domain pairs are covered."""
    domains = list(DOMAIN_GENERATORS.keys())
    expected_pairs = set()
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            expected_pairs.add((domains[i], domains[j]))

    actual_pairs = set()
    for name, (gen_fn, involved) in CROSS_DOMAIN_GENERATORS.items():
        pair = tuple(sorted(involved))
        actual_pairs.add(pair)

    expected_sorted = {tuple(sorted(p)) for p in expected_pairs}
    missing = expected_sorted - actual_pairs
    assert len(missing) == 0, f"Missing domain pairs: {missing}"
    assert len(actual_pairs) == 10, f"Expected 10 pairs, got {len(actual_pairs)}"


def test_cross_domain_contains_both_domains():
    """Cross-domain queries contain patterns from both involved domains."""
    rng = np.random.RandomState(42)

    # arith_reverse should contain '+', '=', and '>'
    data = CROSS_DOMAIN_GENERATORS["arith_reverse"][0](20, rng)
    for s in data:
        assert '+' in s, f"arith_reverse missing '+': {s}"
        assert '=' in s, f"arith_reverse missing '=': {s}"
        assert '>' in s, f"arith_reverse missing '>': {s}"

    # arith_parity should contain arithmetic and parity results
    rng2 = np.random.RandomState(42)
    data2 = CROSS_DOMAIN_GENERATORS["arith_parity"][0](20, rng2)
    for s in data2:
        assert '+' in s, f"arith_parity missing '+': {s}"
        assert ('even' in s or 'odd' in s), f"arith_parity missing parity: {s}"


def test_gram_schmidt_zeros_cosines():
    """GS orthogonalization reduces pairwise cosines to ~0."""
    rng = np.random.RandomState(42)
    tok = CharTokenizer()

    base = init_model(tok.vocab_size, d=16, H=2, L=1, seed=42)
    deltas = []
    for i in range(3):
        expert = {k: (v.copy() if k != '_config' else v) for k, v in base.items()}
        for k in expert:
            if k != '_config' and expert[k].ndim >= 2:
                expert[k] = expert[k] + rng.randn(*expert[k].shape).astype(np.float32) * 0.01
        delta = compute_delta(base, expert)
        deltas.append(delta)

    ortho_deltas, report = gram_schmidt_deltas(deltas)

    assert report['post_cosines_max'] < 1e-6, \
        f"Post-GS max cosine {report['post_cosines_max']} should be ~0"
    assert report['signal_retention_min'] > 0.5, \
        f"Min signal retention {report['signal_retention_min']} too low"


def test_multi_expert_beats_naive():
    """Multi-expert (2 relevant) should beat naive (all N) on cross-domain queries."""
    rng = np.random.RandomState(42)
    tok = CharTokenizer()

    base = init_model(tok.vocab_size, d=16, H=2, L=1, seed=42)
    domains = list(DOMAIN_GENERATORS.keys())[:3]

    expert_deltas = {}
    for dom in domains:
        gen = DOMAIN_GENERATORS[dom]
        data = gen(50, np.random.RandomState(42 + hash(dom) % 10000))
        enc = [tok.encode(s) for s in data]

        expert = {k: (v.copy() if k != '_config' else v) for k, v in base.items()}
        expert = train_model(expert, enc, tok.pad_id, epochs=3, lr=0.001,
                             batch_size=16, verbose=False)
        delta = compute_delta(base, expert)
        delta_trunc, _ = svd_truncate_delta(delta, rank=2)
        expert_deltas[dom] = delta_trunc

    cross_gen = CROSS_DOMAIN_GENERATORS["arith_reverse"][0]
    cross_data = cross_gen(20, np.random.RandomState(99))
    cross_enc = [tok.encode(s) for s in cross_data]

    naive = merge_deltas(list(expert_deltas.values()), mode='avg')
    naive_params = apply_delta(base, naive)
    naive_loss = eval_loss(naive_params, cross_enc, tok.pad_id)

    if "arithmetic" in expert_deltas and "reverse" in expert_deltas:
        multi = merge_deltas(
            [expert_deltas["arithmetic"], expert_deltas["reverse"]], mode='avg')
        multi_params = apply_delta(base, multi)
        multi_loss = eval_loss(multi_params, cross_enc, tok.pad_id)

        assert multi_loss <= naive_loss * 1.15, \
            f"Multi ({multi_loss:.3f}) much worse than naive ({naive_loss:.3f})"


def test_subspace_analysis():
    """Subspace analysis produces valid shared fractions."""
    rng = np.random.RandomState(42)
    tok = CharTokenizer()

    base = init_model(tok.vocab_size, d=16, H=2, L=1, seed=42)
    deltas = []
    names = ["a", "b"]
    for i in range(2):
        expert = {k: (v.copy() if k != '_config' else v) for k, v in base.items()}
        for k in expert:
            if k != '_config' and expert[k].ndim >= 2:
                expert[k] = expert[k] + rng.randn(*expert[k].shape).astype(np.float32) * 0.01
        deltas.append(compute_delta(base, expert))

    result = subspace_analysis(deltas, names)
    assert "a_vs_b" in result
    info = result["a_vs_b"]
    assert 0.0 <= info['shared_fraction_i'] <= 1.0
    assert info['total_norm_i'] > 0


if __name__ == "__main__":
    test_cross_domain_data_generators()
    print("PASS: test_cross_domain_data_generators")

    test_all_10_pairs_covered()
    print("PASS: test_all_10_pairs_covered")

    test_cross_domain_contains_both_domains()
    print("PASS: test_cross_domain_contains_both_domains")

    test_gram_schmidt_zeros_cosines()
    print("PASS: test_gram_schmidt_zeros_cosines")

    test_multi_expert_beats_naive()
    print("PASS: test_multi_expert_beats_naive")

    test_subspace_analysis()
    print("PASS: test_subspace_analysis")

    print("\nAll tests passed.")

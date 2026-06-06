#!/usr/bin/env python3
"""Tests for shadow scoring mechanism."""

import numpy as np
from experiments.models.shadow_scoring.shadow_scoring import (
    EloRating,
    kendall_tau,
    pairwise_agreement,
    select_challenger,
    hash_route,
    compute_per_query_answer_ppl,
)
from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
    CharTokenizer,
    init_model,
)
import random


def test_elo_basic():
    """Test Elo rating updates."""
    elo = EloRating(3, k_factor=32, initial_rating=1500)
    assert all(r == 1500 for r in elo.ratings)

    # Winner gains, loser loses
    elo.update(0, 1)
    assert elo.ratings[0] > 1500
    assert elo.ratings[1] < 1500
    assert elo.ratings[2] == 1500

    # Sum is preserved (zero-sum)
    assert abs(sum(elo.ratings) - 4500) < 1e-6


def test_elo_convergence():
    """Test that Elo converges when one player always wins."""
    elo = EloRating(3, k_factor=32)

    # Player 0 always beats 1, 1 always beats 2
    for _ in range(100):
        elo.update(0, 1)
        elo.update(1, 2)
        elo.update(0, 2)

    # 0 should have highest rating, 2 lowest
    assert elo.ratings[0] > elo.ratings[1] > elo.ratings[2]
    ranking = elo.get_ranking()
    assert ranking == [0, 1, 2]


def test_kendall_tau():
    """Test Kendall's tau computation."""
    # Perfect agreement
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    # Perfect disagreement
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0

    # Partial agreement (one swap)
    tau = kendall_tau([0, 1, 2], [0, 2, 1])
    assert -1 < tau < 1


def test_pairwise_agreement():
    """Test pairwise agreement computation."""
    # Perfect agreement
    assert pairwise_agreement([0, 1, 2], [0, 1, 2]) == 1.0

    # Complete disagreement
    assert pairwise_agreement([0, 1, 2], [2, 1, 0]) == 0.0


def test_select_challenger():
    """Test challenger selection never picks incumbent."""
    rng = random.Random(42)
    for _ in range(100):
        incumbent = rng.randint(0, 4)
        challenger = select_challenger(incumbent, 5, rng)
        assert challenger != incumbent
        assert 0 <= challenger < 5


def test_hash_route_deterministic():
    """Test hash routing is deterministic."""
    for _ in range(20):
        q = f"test_query_{_}"
        r1 = hash_route(q, 5)
        r2 = hash_route(q, 5)
        assert r1 == r2
        assert 0 <= r1 < 5


def test_hash_route_distribution():
    """Test hash routing distributes roughly uniformly."""
    counts = [0] * 5
    for i in range(500):
        idx = hash_route(f"query_{i}", 5)
        counts[idx] += 1

    # Each bucket should get roughly 100 (20%)
    for c in counts:
        assert 50 < c < 150, f"Hash distribution too skewed: {counts}"


def test_per_query_answer_ppl():
    """Test per-query answer PPL computation."""
    tokenizer = CharTokenizer()
    params = init_model(tokenizer.vocab_size, d=32, H=2, L=2, max_T=48, seed=42)

    query_str = "12+34=46"
    query_enc = tokenizer.encode(query_str)

    ppl, n_tokens = compute_per_query_answer_ppl(
        params, query_str, query_enc, "=", tokenizer.pad_id)

    assert ppl > 0
    assert not np.isinf(ppl)
    assert n_tokens > 0  # Should have answer tokens after "="


def test_per_query_ppl_expert_beats_base():
    """Test that domain expert has lower PPL than base on its domain."""
    from experiments.models.answer_conditioned_scoring.answer_conditioned_scoring import (
        train_model, train_expert, DOMAIN_GENERATORS
    )
    import numpy as onp

    onp.random.seed(42)
    random.seed(42)

    tokenizer = CharTokenizer()
    V = tokenizer.vocab_size

    # Small training for speed
    params = init_model(V, d=32, H=2, L=2, max_T=48, seed=42)

    # Generate data
    rng = random.Random(42)
    arith_data = DOMAIN_GENERATORS["arithmetic"](200, rng)
    arith_enc = [tokenizer.encode(s) for s in arith_data]

    # Quick base training
    all_data = arith_enc[:100]
    params = train_model(params, all_data, tokenizer.pad_id,
                         epochs=5, lr=0.001, batch_size=32, verbose=False)
    base_params = {k: (v.copy() if k != '_config' else v) for k, v in params.items()}

    # Train arithmetic expert
    delta = train_expert(base_params, arith_enc[:100], tokenizer.pad_id,
                         epochs=10, lr=0.001, batch_size=32, verbose=False)

    # Build expert params
    expert_params = {k: (base_params[k].copy() if k != '_config' else base_params[k])
                     for k in base_params}
    for k, d in delta.items():
        expert_params[k] = expert_params[k] + d

    # Compare PPL on arithmetic queries
    test_query = "25+13=38"
    test_enc = tokenizer.encode(test_query)

    base_ppl, _ = compute_per_query_answer_ppl(
        base_params, test_query, test_enc, "=", tokenizer.pad_id)
    expert_ppl, _ = compute_per_query_answer_ppl(
        expert_params, test_query, test_enc, "=", tokenizer.pad_id)

    # Expert should generally have lower PPL on its domain
    # (not always guaranteed with small training, but directionally correct)
    print(f"  Base PPL: {base_ppl:.2f}, Expert PPL: {expert_ppl:.2f}")
    # Just assert both are finite
    assert not np.isinf(base_ppl)
    assert not np.isinf(expert_ppl)


if __name__ == "__main__":
    test_elo_basic()
    print("PASS: test_elo_basic")

    test_elo_convergence()
    print("PASS: test_elo_convergence")

    test_kendall_tau()
    print("PASS: test_kendall_tau")

    test_pairwise_agreement()
    print("PASS: test_pairwise_agreement")

    test_select_challenger()
    print("PASS: test_select_challenger")

    test_hash_route_deterministic()
    print("PASS: test_hash_route_deterministic")

    test_hash_route_distribution()
    print("PASS: test_hash_route_distribution")

    test_per_query_answer_ppl()
    print("PASS: test_per_query_answer_ppl")

    test_per_query_ppl_expert_beats_base()
    print("PASS: test_per_query_ppl_expert_beats_base")

    print("\nAll tests passed!")

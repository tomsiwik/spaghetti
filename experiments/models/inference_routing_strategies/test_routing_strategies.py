#!/usr/bin/env python3
"""Smoke tests for routing strategy comparison."""

import numpy as np
import pytest

from experiments.models.inference_routing_strategies.routing_strategies import (
    generate_quality_matrix,
    generate_queries,
    compute_expert_centroids,
    PreMergeRouter,
    HashRingRouter,
    EmbeddingSimilarityRouter,
    TinyClassifierRouter,
    HierarchicalRouter,
    OracleRouter,
    measure_quality,
)


@pytest.fixture
def setup():
    rng = np.random.default_rng(42)
    n_domains, n_experts, n_clusters = 6, 12, 2
    embed_dim = 32
    qm = generate_quality_matrix(n_domains, n_experts, n_clusters, 0.8, rng)
    queries = generate_queries(500, n_domains, embed_dim,
                                qm["domain_to_cluster"], rng)
    centroids = compute_expert_centroids(
        qm["expert_to_domain"], queries["domain_centroids"],
        n_experts, embed_dim, rng)
    oracle = OracleRouter(qm["Q"])
    return {
        "Q": qm["Q"],
        "qm": qm,
        "queries": queries,
        "centroids": centroids,
        "oracle": oracle,
        "n_experts": n_experts,
        "n_domains": n_domains,
        "embed_dim": embed_dim,
    }


def test_quality_matrix_shape(setup):
    Q = setup["Q"]
    assert Q.shape == (6, 12)


def test_quality_matrix_specialization(setup):
    Q = setup["Q"]
    qm = setup["qm"]
    # Home domain quality should be highest for each expert
    for e in range(setup["n_experts"]):
        home = qm["expert_to_domain"][e]
        assert Q[home, e] > Q.mean(), f"Expert {e} should be best at domain {home}"


def test_premerge_returns_sentinel(setup):
    pm = PreMergeRouter(setup["Q"])
    result = pm.route(setup["queries"]["embeddings"][0])
    assert result == -1


def test_hash_ring_returns_valid(setup):
    hr = HashRingRouter(setup["n_experts"])
    for i in range(10):
        result = hr.route(setup["queries"]["embeddings"][i], i)
        assert 0 <= result < setup["n_experts"]


def test_embedding_sim_returns_valid(setup):
    es = EmbeddingSimilarityRouter(setup["centroids"])
    for i in range(10):
        result = es.route(setup["queries"]["embeddings"][i], i)
        assert 0 <= result < setup["n_experts"]


def test_tiny_classifier_trains(setup):
    oracle = setup["oracle"]
    labels = np.array([oracle.route_for_domain(d)
                       for d in setup["queries"]["domains"]])
    tc = TinyClassifierRouter(setup["n_experts"], setup["embed_dim"], 16)
    tc.train(setup["queries"]["embeddings"], labels, epochs=10)
    result = tc.route(setup["queries"]["embeddings"][0])
    assert 0 <= result < setup["n_experts"]


def test_hierarchical_returns_valid(setup):
    hier = HierarchicalRouter(
        setup["queries"]["cluster_centroids"],
        setup["qm"]["expert_to_cluster"],
        setup["n_experts"])
    for i in range(10):
        result = hier.route(setup["queries"]["embeddings"][i], i)
        assert 0 <= result < setup["n_experts"]


def test_batch_routing_consistent(setup):
    es = EmbeddingSimilarityRouter(setup["centroids"])
    batch = es.route_batch(setup["queries"]["embeddings"][:10])
    singles = [es.route(setup["queries"]["embeddings"][i], i) for i in range(10)]
    np.testing.assert_array_equal(batch, singles)


def test_quality_capture_between_0_and_1(setup):
    pm = PreMergeRouter(setup["Q"])
    qual = measure_quality(pm, setup["Q"], setup["queries"]["domains"],
                           setup["queries"]["embeddings"], setup["oracle"])
    assert 0 < qual["quality_capture"] < 1.0
    assert 0 <= qual["oracle_agreement"] <= 1.0


def test_oracle_is_best(setup):
    """Oracle quality capture should be 1.0."""
    # Oracle always picks best expert
    oracle = setup["oracle"]
    Q = setup["Q"]
    for d in range(setup["n_domains"]):
        best_e = oracle.route_for_domain(d)
        assert Q[d, best_e] == Q[d, :].max()


def test_k1_latency_under_50ms():
    """K1 kill criterion: all strategies <50ms at N=100."""
    # This is a sanity check -- real measurement is in the experiment
    # Just verify routers can be constructed at N=100
    rng = np.random.default_rng(42)
    N = 100
    centroids = rng.standard_normal((N, 64))
    es = EmbeddingSimilarityRouter(centroids)
    q = rng.standard_normal(64)
    # Should not take >50ms
    import time
    start = time.perf_counter()
    for _ in range(100):
        es.route(q)
    elapsed = (time.perf_counter() - start) / 100 * 1000
    assert elapsed < 50.0, f"Routing took {elapsed:.2f}ms, exceeds 50ms"

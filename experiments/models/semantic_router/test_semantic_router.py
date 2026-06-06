#!/usr/bin/env python3
"""Tests for semantic router experiment."""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.models.semantic_router.semantic_router import (
    compute_ngram_features,
    project_features,
    make_projection_matrix,
    HashRingRouter,
    KeywordRouter,
    CosineRouter,
    LSHRouter,
    UtteranceRouter,
    ALL_DOMAINS,
    DOMAIN_TO_CLUSTER,
    D_EMBED,
    VOCAB_SIZE,
    CONTEXT_LEN,
)


def test_ngram_features_shape():
    """N-gram features have expected shape."""
    x = np.random.randint(0, VOCAB_SIZE, (10, CONTEXT_LEN))
    feats = compute_ngram_features(x)
    assert feats.shape == (10, 224), f"Expected (10, 224), got {feats.shape}"
    # Unigram part should sum to ~1 (normalized frequencies)
    uni = feats[:, :VOCAB_SIZE]
    assert np.allclose(uni.sum(axis=1), 1.0, atol=1e-6)


def test_projection_normalization():
    """Projected features are L2-normalized."""
    rng = np.random.RandomState(42)
    x = np.random.randint(0, VOCAB_SIZE, (20, CONTEXT_LEN))
    feats = compute_ngram_features(x)
    proj = make_projection_matrix(224, D_EMBED, rng)
    embs = project_features(feats, proj)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"Norms not 1: {norms}"


def test_hash_ring_covers_all_experts():
    """Hash ring distributes across all experts for enough queries."""
    router = HashRingRouter(ALL_DOMAINS, seed=42)
    x = np.random.randint(0, VOCAB_SIZE, (1000, CONTEXT_LEN))
    routes = router.route_batch(x)
    unique = set(routes)
    assert len(unique) == len(ALL_DOMAINS), \
        f"Hash ring covered {len(unique)}/{len(ALL_DOMAINS)} experts"


def test_cosine_router_returns_valid_domains():
    """Cosine router returns valid domain names."""
    rng = np.random.RandomState(42)
    domain_embs = {}
    for d in ALL_DOMAINS:
        embs = rng.randn(50, D_EMBED)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        domain_embs[d] = embs

    router = CosineRouter(ALL_DOMAINS, domain_embs)
    q = rng.randn(10, D_EMBED)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    routes = router.route_batch(q)
    assert all(r in ALL_DOMAINS for r in routes)


def test_lsh_binary_codes():
    """LSH produces binary hash codes of correct dimension."""
    rng = np.random.RandomState(42)
    domain_embs = {}
    for d in ALL_DOMAINS:
        embs = rng.randn(50, D_EMBED)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        domain_embs[d] = embs

    router = LSHRouter(ALL_DOMAINS, domain_embs, n_planes=32, rng=rng)
    assert router.planes.shape == (D_EMBED, 32)
    for d in ALL_DOMAINS:
        codes = router.domain_codes[d]
        assert codes.shape == (50, 32)
        assert set(np.unique(codes)).issubset({0.0, 1.0})


def test_utterance_router_1nn():
    """Utterance 1-NN router returns nearest exemplar's domain."""
    rng = np.random.RandomState(42)
    domain_embs = {}
    for d in ALL_DOMAINS:
        embs = rng.randn(50, D_EMBED)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        domain_embs[d] = embs

    router = UtteranceRouter(ALL_DOMAINS, domain_embs, n_exemplars=50)
    # Query very close to first domain's first exemplar
    q = domain_embs[ALL_DOMAINS[0]][0:1] + rng.randn(1, D_EMBED) * 0.01
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    routes = router.route_batch(q)
    assert routes[0] == ALL_DOMAINS[0], \
        f"Expected {ALL_DOMAINS[0]}, got {routes[0]}"


def test_cluster_accuracy_above_random():
    """All semantic routers achieve above-random cluster accuracy on trivial data."""
    from experiments.models.semantic_router.semantic_router import (
        generate_cluster_prototypes, generate_domain_data
    )
    rng = np.random.RandomState(42)
    prototypes = generate_cluster_prototypes(rng, cross_cluster_distance=3.0)

    domain_data = {}
    for d in ALL_DOMAINS:
        c = DOMAIN_TO_CLUSTER[d]
        x, y = generate_domain_data(rng, prototypes, d, c, n_sequences=100)
        domain_data[d] = (x, y)

    # Build embeddings
    proj = make_projection_matrix(224, D_EMBED, rng)
    domain_embs = {}
    for d in ALL_DOMAINS:
        feats = compute_ngram_features(domain_data[d][0])
        domain_embs[d] = project_features(feats, proj)

    # Test data
    test_x = np.concatenate([domain_data[d][0][:20] for d in ALL_DOMAINS])
    test_feats = compute_ngram_features(test_x)
    test_embs = project_features(test_feats, proj)
    test_domains = []
    for d in ALL_DOMAINS:
        test_domains.extend([d] * 20)

    # Cosine router
    cosine = CosineRouter(ALL_DOMAINS, domain_embs)
    routes = cosine.route_batch(test_embs)
    cluster_acc = sum(
        1 for i in range(len(test_domains))
        if DOMAIN_TO_CLUSTER.get(routes[i], '') == DOMAIN_TO_CLUSTER[test_domains[i]]
    ) / len(test_domains)
    assert cluster_acc > 0.5, f"Cosine cluster acc {cluster_acc} should be >0.5"


if __name__ == '__main__':
    tests = [
        test_ngram_features_shape,
        test_projection_normalization,
        test_hash_ring_covers_all_experts,
        test_cosine_router_returns_valid_domains,
        test_lsh_binary_codes,
        test_utterance_router_1nn,
        test_cluster_accuracy_above_random,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {t.__name__}: {e}")
    print("\nAll tests complete.")

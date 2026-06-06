#!/usr/bin/env python3
"""Tests for content_aware_routing experiment."""

import numpy as np


def test_hash_ring_deterministic():
    """Hash ring produces the same result for the same input."""
    from experiments.models.content_aware_routing.content_aware_routing import (
        HashRingRouter, ALL_DOMAINS,
    )
    router = HashRingRouter(ALL_DOMAINS, seed=42)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                  dtype=np.int32)
    r1 = router.route(x)
    r2 = router.route(x)
    assert r1 == r2, "Hash ring should be deterministic"
    assert r1 in ALL_DOMAINS, f"Routed to unknown domain: {r1}"


def test_hash_ring_coverage():
    """Hash ring distributes queries across multiple experts."""
    from experiments.models.content_aware_routing.content_aware_routing import (
        HashRingRouter, ALL_DOMAINS, VOCAB_SIZE, CONTEXT_LEN,
    )
    router = HashRingRouter(ALL_DOMAINS, seed=42)
    rng = np.random.RandomState(0)
    routed = set()
    for _ in range(500):
        x = rng.randint(0, VOCAB_SIZE, size=CONTEXT_LEN).astype(np.int32)
        routed.add(router.route(x))
    # Should hit at least 10 of 15 experts with 500 random queries
    assert len(routed) >= 10, f"Only routed to {len(routed)} experts"


def test_cosine_router_returns_valid():
    """Cosine router returns a valid domain name."""
    from experiments.models.content_aware_routing.content_aware_routing import (
        CosineRouter, MicroMLP, ALL_DOMAINS, DOMAIN_TO_CLUSTER,
        generate_cluster_prototypes, generate_domain_data,
        CONTEXT_LEN, VOCAB_SIZE,
    )
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    prototypes = generate_cluster_prototypes(rng)
    domain_data = {}
    for domain in ALL_DOMAINS:
        cluster = DOMAIN_TO_CLUSTER[domain]
        x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                     n_sequences=50)
        domain_data[domain] = (x, y)
    router = CosineRouter(model, ALL_DOMAINS, domain_data)
    x = rng.randint(0, VOCAB_SIZE, size=CONTEXT_LEN).astype(np.int32)
    result = router.route(x, model)
    assert result in ALL_DOMAINS


def test_keyword_router_returns_valid():
    """Keyword router returns a valid domain name."""
    from experiments.models.content_aware_routing.content_aware_routing import (
        KeywordRouter, ALL_DOMAINS, DOMAIN_TO_CLUSTER,
        generate_cluster_prototypes, generate_domain_data,
        CONTEXT_LEN, VOCAB_SIZE,
    )
    rng = np.random.RandomState(42)
    prototypes = generate_cluster_prototypes(rng)
    domain_data = {}
    for domain in ALL_DOMAINS:
        cluster = DOMAIN_TO_CLUSTER[domain]
        x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                     n_sequences=50)
        domain_data[domain] = (x, y)
    router = KeywordRouter(ALL_DOMAINS, domain_data)
    x = rng.randint(0, VOCAB_SIZE, size=CONTEXT_LEN).astype(np.int32)
    result = router.route(x)
    assert result in ALL_DOMAINS


def test_merge_loras():
    """Merging LoRA adapters produces valid shapes."""
    from experiments.models.content_aware_routing.content_aware_routing import (
        init_lora, merge_loras, N_LAYERS, D_MODEL, LORA_RANK, D_FF,
    )
    rng = np.random.RandomState(42)
    loras = [init_lora(rng) for _ in range(3)]
    merged = merge_loras(loras)
    for l in range(N_LAYERS):
        assert merged['A1'][l].shape == (D_MODEL, LORA_RANK)
        assert merged['B1'][l].shape == (LORA_RANK, D_FF)
        assert merged['A2'][l].shape == (D_FF, LORA_RANK)
        assert merged['B2'][l].shape == (LORA_RANK, D_MODEL)


def test_ntp_loss_finite():
    """NTP loss is finite for random inputs."""
    from experiments.models.content_aware_routing.content_aware_routing import (
        MicroMLP, evaluate_ntp_loss, VOCAB_SIZE, CONTEXT_LEN,
    )
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    x = rng.randint(0, VOCAB_SIZE, size=(10, CONTEXT_LEN)).astype(np.int32)
    y = rng.randint(0, VOCAB_SIZE, size=10).astype(np.int32)
    loss = evaluate_ntp_loss(model, x, y)
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss > 0, f"Loss should be positive: {loss}"


if __name__ == '__main__':
    test_hash_ring_deterministic()
    print("PASS: test_hash_ring_deterministic")
    test_hash_ring_coverage()
    print("PASS: test_hash_ring_coverage")
    test_cosine_router_returns_valid()
    print("PASS: test_cosine_router_returns_valid")
    test_keyword_router_returns_valid()
    print("PASS: test_keyword_router_returns_valid")
    test_merge_loras()
    print("PASS: test_merge_loras")
    test_ntp_loss_finite()
    print("PASS: test_ntp_loss_finite")
    print("\nAll tests passed.")

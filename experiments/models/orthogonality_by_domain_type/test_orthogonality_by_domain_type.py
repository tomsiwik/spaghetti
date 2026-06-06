#!/usr/bin/env python3
"""Tests for Orthogonality by Domain Type experiment."""

import sys
from pathlib import Path

import numpy as np

# Direct import to avoid MLX-dependent micro.models.__init__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "orthogonality_by_domain_type",
    str(Path(__file__).parent / "orthogonality_by_domain_type.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

MicroMLP = mod.MicroMLP
init_lora = mod.init_lora
lora_to_delta_vector = mod.lora_to_delta_vector
train_lora = mod.train_lora
generate_cluster_prototypes = mod.generate_cluster_prototypes
generate_domain_data = mod.generate_domain_data
cosine_sim = mod.cosine_sim
compute_cosine_matrix = mod.compute_cosine_matrix
analyze_clustering = mod.analyze_clustering
permutation_test = mod.permutation_test
softmax = mod.softmax
ALL_DOMAINS = mod.ALL_DOMAINS
CLUSTERS = mod.CLUSTERS
DOMAIN_TO_CLUSTER = mod.DOMAIN_TO_CLUSTER
VOCAB_SIZE = mod.VOCAB_SIZE
D_MODEL = mod.D_MODEL
D_FF = mod.D_FF
N_LAYERS = mod.N_LAYERS
LORA_RANK = mod.LORA_RANK


def test_model_forward():
    """Model forward pass produces valid logits."""
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    x = rng.randint(0, VOCAB_SIZE, size=(4, 16))
    logits, intermediates, h = model.forward(x)
    assert logits.shape == (4, VOCAB_SIZE), f"Expected (4, {VOCAB_SIZE}), got {logits.shape}"
    assert len(intermediates) == N_LAYERS
    # Logits should not be all the same
    assert np.std(logits) > 0, "Logits are constant"
    print("PASS: model_forward")


def test_model_forward_with_lora():
    """Model forward with LoRA produces different logits."""
    rng = np.random.RandomState(42)
    model = MicroMLP(rng)
    lora = init_lora(rng)
    x = rng.randint(0, VOCAB_SIZE, size=(4, 16))

    logits_base, _, _ = model.forward(x)
    logits_lora, _, _ = model.forward(x, lora)

    # With zero-initialized B, LoRA output should equal base
    diff = np.abs(logits_base - logits_lora).max()
    assert diff < 1e-10, f"Zero-B LoRA should match base, got diff={diff}"
    print("PASS: model_forward_with_lora")


def test_lora_delta_vector_shape():
    """Delta vector has expected dimension."""
    rng = np.random.RandomState(42)
    lora = init_lora(rng)
    vec = lora_to_delta_vector(lora)
    # Each layer: (d, dff) + (dff, d) = 2 * d * dff
    expected_dim = N_LAYERS * 2 * D_MODEL * D_FF
    assert vec.shape == (expected_dim,), f"Expected ({expected_dim},), got {vec.shape}"
    print("PASS: lora_delta_vector_shape")


def test_zero_lora_has_zero_delta():
    """Zero-initialized LoRA (B=0) produces zero delta vector."""
    rng = np.random.RandomState(42)
    lora = init_lora(rng)
    vec = lora_to_delta_vector(lora)
    assert np.all(vec == 0), "Zero-B LoRA should give zero delta"
    print("PASS: zero_lora_has_zero_delta")


def test_cluster_prototypes_are_different():
    """Different clusters should have different prototype transition matrices."""
    rng = np.random.RandomState(42)
    prototypes = generate_cluster_prototypes(rng)

    # Each prototype should be a valid probability distribution
    for name, proto in prototypes.items():
        assert proto.shape == (VOCAB_SIZE, VOCAB_SIZE)
        row_sums = proto.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"Rows of {name} don't sum to 1"

    # Prototypes should be different from each other
    codes = ['code', 'reasoning', 'knowledge']
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            diff = np.abs(prototypes[codes[i]] - prototypes[codes[j]]).mean()
            assert diff > 0.01, f"{codes[i]} vs {codes[j]} diff={diff} too small"

    print("PASS: cluster_prototypes_are_different")


def test_domain_data_generation():
    """Generated data has valid shapes and token ranges."""
    rng = np.random.RandomState(42)
    prototypes = generate_cluster_prototypes(rng)
    x, y = generate_domain_data(rng, prototypes, 'python', 'code', n_sequences=50)
    assert x.shape == (50, 16), f"x shape {x.shape}"
    assert y.shape == (50,), f"y shape {y.shape}"
    assert x.min() >= 0 and x.max() < VOCAB_SIZE
    assert y.min() >= 0 and y.max() < VOCAB_SIZE
    print("PASS: domain_data_generation")


def test_cosine_sim():
    """Cosine similarity basic properties."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_sim(a, b)) < 1e-10, "Orthogonal vectors should have cos=0"
    assert abs(cosine_sim(a, c) - 1.0) < 1e-10, "Same direction should have cos=1"
    assert abs(cosine_sim(a, -a) + 1.0) < 1e-10, "Opposite should have cos=-1"
    print("PASS: cosine_sim")


def test_permutation_test_trivial():
    """Permutation test with clearly separated groups."""
    within = [0.5, 0.6, 0.7, 0.8, 0.9]
    cross = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = permutation_test(within, cross, n_permutations=1000)
    assert result['p_value'] < 0.01, f"p={result['p_value']} should be <0.01"
    assert result['significant_at_001'], "Should be significant"
    print("PASS: permutation_test_trivial")


def test_domains_cover_all_clusters():
    """All 15 domains are in correct clusters."""
    assert len(ALL_DOMAINS) == 15
    assert len(DOMAIN_TO_CLUSTER) == 15
    for cluster, domains in CLUSTERS.items():
        assert len(domains) == 5, f"Cluster {cluster} has {len(domains)} domains"
        for d in domains:
            assert DOMAIN_TO_CLUSTER[d] == cluster
    print("PASS: domains_cover_all_clusters")


def test_analyze_clustering_pair_counts():
    """Analysis produces correct number of within/cross pairs."""
    # 15 domains, C(15,2)=105 pairs
    # 3 clusters of 5: within = 3 * C(5,2) = 30
    # cross = 105 - 30 = 75
    rng = np.random.RandomState(42)
    vecs = {d: rng.randn(100) for d in ALL_DOMAINS}
    matrix, domains = compute_cosine_matrix(vecs)
    analysis = analyze_clustering(matrix, domains)
    assert analysis['within_cluster']['n_pairs'] == 30
    assert analysis['cross_cluster']['n_pairs'] == 75
    print("PASS: analyze_clustering_pair_counts")


def test_results_exist():
    """Check that results.json was generated with expected structure."""
    results_file = Path(__file__).parent / "results.json"
    if not results_file.exists():
        print("SKIP: results.json not found (run experiment first)")
        return

    import json
    with open(results_file) as f:
        results = json.load(f)

    # Check structure
    assert 'config' in results
    assert 'aggregate' in results
    assert 'kill_criteria' in results
    assert 'seed_results' in results

    # Check aggregate values
    agg = results['aggregate']
    assert agg['within_mean'] > 0
    assert agg['cross_mean'] > 0

    # Check kill criteria were evaluated
    kc = results['kill_criteria']
    assert 'k1_within_gt_cross' in kc
    assert 'k2_pattern_significant' in kc
    assert 'overall_kill' in kc

    print(f"PASS: results_exist (within={agg['within_mean']:.4f}, "
          f"cross={agg['cross_mean']:.4f}, ratio={agg['mean_ratio']:.2f}x)")


if __name__ == '__main__':
    test_model_forward()
    test_model_forward_with_lora()
    test_lora_delta_vector_shape()
    test_zero_lora_has_zero_delta()
    test_cluster_prototypes_are_different()
    test_domain_data_generation()
    test_cosine_sim()
    test_permutation_test_trivial()
    test_domains_cover_all_clusters()
    test_analyze_clustering_pair_counts()
    test_results_exist()
    print("\nAll tests passed.")

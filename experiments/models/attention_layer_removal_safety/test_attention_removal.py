"""Quick smoke tests for attention_layer_removal_safety."""

import numpy as np
from experiments.models.attention_layer_removal_safety.run_experiment import (
    generate_expert_set_controlled_cos,
    gram_schmidt,
    cosine_sim,
    reconstruction_error,
    naive_removal,
    gs_recompute,
)


def test_controlled_cosine_generation():
    """Verify that generate_expert_set_controlled_cos produces correct cosines."""
    D = 10000
    N = 10
    for target_cos in [0.1, 0.5, 0.85, 0.95]:
        deltas = generate_expert_set_controlled_cos(N, D, target_cos, seed=42)
        cos_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                cos_pairs.append(abs(cosine_sim(deltas[i], deltas[j])))
        mean_cos = np.mean(cos_pairs)
        assert abs(mean_cos - target_cos) < 0.02, \
            f"Expected cos~{target_cos}, got {mean_cos:.4f}"


def test_gs_reduces_cosine():
    """Verify GS orthogonalization reduces pairwise cosine."""
    D = 10000
    N = 10
    deltas = generate_expert_set_controlled_cos(N, D, target_cos=0.85, seed=42)
    ortho = gram_schmidt(deltas)
    cos_before = np.mean([abs(cosine_sim(deltas[i], deltas[j]))
                          for i in range(N) for j in range(i+1, N)])
    cos_after = np.mean([abs(cosine_sim(ortho[i], ortho[j]))
                         for i in range(N) for j in range(i+1, N)])
    assert cos_after < cos_before, \
        f"GS should reduce cosine: before={cos_before:.4f}, after={cos_after:.4f}"


def test_naive_subtraction_fails_at_high_cosine():
    """K1: naive subtraction error >3% at cos=0.85."""
    D = 50000
    N = 20
    deltas = generate_expert_set_controlled_cos(N, D, target_cos=0.85, seed=42)
    ortho = gram_schmidt(deltas)
    merged = sum(ortho)
    w_naive = naive_removal(ortho, merged, N // 2)
    w_gt = gs_recompute(deltas, N // 2)
    err = reconstruction_error(w_naive, w_gt)
    assert err > 3.0, f"Expected >3% error at cos=0.85, got {err:.4f}%"


def test_naive_subtraction_ok_at_low_cosine():
    """Naive subtraction should work at cos=0.001."""
    D = 50000
    N = 20
    deltas = generate_expert_set_controlled_cos(N, D, target_cos=0.001, seed=42)
    ortho = gram_schmidt(deltas)
    merged = sum(ortho)
    w_naive = naive_removal(ortho, merged, N // 2)
    w_gt = gs_recompute(deltas, N // 2)
    err = reconstruction_error(w_naive, w_gt)
    assert err < 1.0, f"Expected <1% error at cos=0.001, got {err:.4f}%"


def test_hybrid_strategy():
    """Hybrid (GS attn + naive MLP) should have low error."""
    D_attn = 20000
    D_mlp = 60000
    N = 20
    seed = 42
    remove_idx = N // 2

    attn = generate_expert_set_controlled_cos(N, D_attn, 0.85, seed)
    mlp = generate_expert_set_controlled_cos(N, D_mlp, 0.001, seed + 5000)

    # Ground truth per-component
    gt_attn = gs_recompute(attn, remove_idx)
    gt_mlp = gs_recompute(mlp, remove_idx)
    gt = np.concatenate([gt_attn, gt_mlp])

    # Hybrid: GS for attn, naive for MLP
    ortho_mlp = gram_schmidt(mlp)
    merged_mlp = sum(ortho_mlp)
    hybrid_attn = gs_recompute(attn, remove_idx)
    hybrid_mlp = naive_removal(ortho_mlp, merged_mlp, remove_idx)
    hybrid = np.concatenate([hybrid_attn, hybrid_mlp])

    err = reconstruction_error(hybrid, gt)
    assert err < 1.0, f"Hybrid error should be <1%, got {err:.4f}%"


if __name__ == "__main__":
    test_controlled_cosine_generation()
    print("PASS: test_controlled_cosine_generation")
    test_gs_reduces_cosine()
    print("PASS: test_gs_reduces_cosine")
    test_naive_subtraction_fails_at_high_cosine()
    print("PASS: test_naive_subtraction_fails_at_high_cosine")
    test_naive_subtraction_ok_at_low_cosine()
    print("PASS: test_naive_subtraction_ok_at_low_cosine")
    test_hybrid_strategy()
    print("PASS: test_hybrid_strategy")
    print("\nAll tests passed.")

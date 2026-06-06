"""Smoke test for LTE rank accumulation quality experiment."""

import time


def test_single_seed_quick():
    """Run a single seed at reduced scale to verify the experiment works."""
    from .lte_rank_accumulation import run_experiment

    t0 = time.time()
    r = run_experiment(
        n_embd=128,  # Smaller than full experiment for speed
        n_head=4,
        n_layer=2,
        block_size=32,
        lora_rank=8,
        lora_alpha=1.0,
        pretrain_steps=100,
        adapt_steps=100,
        merge_every_par=25,
        merge_every_seq=50,
        n_par_heads=4,
        expert_steps=50,
        n_experts=2,
        batch_size=16,
        lr=3e-3,
        seed=42,
    )
    elapsed = time.time() - t0

    # Basic sanity checks
    assert r.n_embd == 128
    assert r.lora_rank == 8
    assert r.parallel_loss > 0
    assert r.sequential_loss > 0
    assert r.continued_loss > 0
    assert r.parallel_rank > 0
    assert r.sequential_rank > 0
    assert r.parallel_mean_cos >= 0
    assert r.sequential_mean_cos >= 0
    assert len(r.parallel_expert_losses) == 2
    assert len(r.sequential_expert_losses) == 2
    assert r.rank_capacity["rank_ratio"] == 4.0
    assert r.rank_capacity["parallel_coverage_pct"] == 25.0  # 32/128
    assert r.rank_capacity["sequential_coverage_pct"] == 6.25  # 8/128

    print(f"\nSmoke test passed in {elapsed:.1f}s")
    print(f"  Parallel base: {r.parallel_loss:.4f}")
    print(f"  Sequential base: {r.sequential_loss:.4f}")
    print(f"  par/seq base: {r.par_vs_seq_base:.4f}")
    print(f"  par/seq cos: {r.par_vs_seq_cos:.4f}")
    print(f"  Verdict: {r.verdict}")


def test_rank_capacity_computation():
    """Test the theoretical rank capacity computation."""
    from .lte_rank_accumulation import compute_rank_per_interval_advantage

    # d=64, r=8, K=4
    cap64 = compute_rank_per_interval_advantage(64, 8, 4)
    assert cap64["parallel_rank_per_interval"] == 32
    assert cap64["sequential_rank_per_interval"] == 8
    assert cap64["rank_ratio"] == 4.0
    assert abs(cap64["parallel_coverage_pct"] - 50.0) < 0.01
    assert abs(cap64["sequential_coverage_pct"] - 12.5) < 0.01

    # d=256, r=8, K=4
    cap256 = compute_rank_per_interval_advantage(256, 8, 4)
    assert cap256["parallel_rank_per_interval"] == 32
    assert cap256["sequential_rank_per_interval"] == 8
    assert cap256["rank_ratio"] == 4.0
    assert abs(cap256["parallel_coverage_pct"] - 12.5) < 0.01
    assert abs(cap256["sequential_coverage_pct"] - 3.125) < 0.01

    # d=4096, r=16, K=4
    cap4096 = compute_rank_per_interval_advantage(4096, 16, 4)
    assert cap4096["parallel_rank_per_interval"] == 64
    assert cap4096["sequential_rank_per_interval"] == 16
    assert cap4096["rank_ratio"] == 4.0
    assert abs(cap4096["parallel_coverage_pct"] - 1.5625) < 0.01

    print("Rank capacity computation test passed")


if __name__ == "__main__":
    test_rank_capacity_computation()
    test_single_seed_quick()

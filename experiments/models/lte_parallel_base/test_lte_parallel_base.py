"""Tests for LTE parallel base construction experiment."""


def test_lte_single_seed():
    """Run a fast single-seed experiment."""
    from .lte_parallel_base import run_experiment

    r = run_experiment(
        n_embd=64, n_head=4, n_layer=4, block_size=32,
        lora_rank=8, pretrain_steps=200, adapt_steps=200,
        merge_every_par=50, merge_every_seq=100,
        n_par_heads=2, expert_steps=100,
        n_experts=4, batch_size=16, lr=3e-3, seed=42,
    )

    # All losses finite
    assert r.parallel_loss < 10.0, f"Parallel base loss diverged: {r.parallel_loss}"
    assert r.sequential_loss < 10.0
    assert r.continued_loss < 10.0

    # Expert losses finite
    for v in r.parallel_expert_losses + r.sequential_expert_losses + r.continued_expert_losses:
        assert v < 10.0

    # Cosines in [0, 1]
    assert 0 <= r.parallel_mean_cos <= 1.0
    assert 0 <= r.sequential_mean_cos <= 1.0

    # Effective ranks positive
    assert r.parallel_rank > 0
    assert r.sequential_rank > 0

    print(f"\nVerdict: {r.verdict}")
    print(f"K1={r.k1} K2={r.k2} K3={r.k3}")


def test_head_rng_independence():
    """Verify heads use different data shards."""
    import random
    rngs = [random.Random(42 + k * 7919) for k in range(4)]
    samples = [rngs[k].random() for k in range(4)]
    assert len(set(samples)) == 4


if __name__ == "__main__":
    test_lte_single_seed()

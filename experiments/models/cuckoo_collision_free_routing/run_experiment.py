"""Experiment: Cuckoo collision-free routing vs softmax baseline.

Tests two kill criteria:
1. Cuckoo routing >2% worse than softmax at same effective k
2. Eviction chain length >3 (routing instability)

Protocol:
1. Train cuckoo model with N=8 experts, k=2 (500 steps)
2. Train softmax baseline with same config (500 steps)
3. Compare val loss (kill criterion 1: >2% worse = killed)
4. Measure eviction chain depth distribution (kill criterion 2: >3 = killed)
5. Track routing diagnostics: eviction rate, tau evolution, entropy
6. Repeat for 3 seeds

Additional measurements:
- Compare with consistent hash routing baseline
- Measure routing collision rate (softmax score ties)
- Track learned tau value and eviction rate over training
"""

import json
import time
import random
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.data import load_names, CharTokenizer, CharDataset, train_val_split
from experiments.train import train, evaluate
from experiments.models.cuckoo_collision_free_routing.cuckoo_collision_free_routing import (
    CuckooCollisionFreeRoutingGPT,
)
from experiments.models.capsule_moe.capsule_moe import CapsuleMoEGPT


def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def measure_softmax_collision_rate(model, val_ds, n_batches=5):
    """Measure how often softmax assigns similar scores to top-2 experts.

    A 'collision' is when the gap between the top-1 and top-2 softmax
    probabilities is < 0.05 (near-tie). This wastes compute because
    the router is uncertain which expert is best.
    """
    rng = random.Random(0)
    total_tokens = 0
    collision_count = 0
    gap_threshold = 0.05

    for _ in range(n_batches):
        inputs, _ = val_ds.get_batch(32, rng)
        B, T = inputs.shape
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for layer in model.layers:
            h = x + layer.attn(layer.norm1(x))
            h_normed = layer.norm2(h)

            scores = layer.capsule_pool.router(h_normed)
            probs = mx.softmax(scores, axis=-1)
            mx.eval(probs)

            # Get top-2 probabilities
            sorted_probs = mx.sort(probs, axis=-1)
            mx.eval(sorted_probs)
            top1 = sorted_probs[..., -1]
            top2 = sorted_probs[..., -2]
            gap = top1 - top2
            mx.eval(gap)

            collisions = mx.sum((gap < gap_threshold).astype(mx.float32)).item()
            collision_count += collisions
            total_tokens += B * T

            x = layer(x)

    return collision_count / total_tokens if total_tokens > 0 else 0


def measure_cuckoo_diagnostics(model, val_ds, n_batches=5):
    """Measure cuckoo routing diagnostics over validation set."""
    rng = random.Random(0)
    all_chain_depths = []
    all_eviction_rates = []
    all_taus = []

    for _ in range(n_batches):
        inputs, _ = val_ds.get_batch(32, rng)
        _ = model(inputs)
        mx.eval(model.parameters())

        diag = model.get_routing_diagnostics()
        for li in range(len(model.layers)):
            d = diag[f"layer_{li}"]
            all_chain_depths.append(d["mean_chain_depth"])
            all_eviction_rates.append(d["eviction_rate"])

    # Get tau values
    for layer in model.layers:
        tau_val = layer.capsule_pool.router.tau.item()
        all_taus.append(tau_val)

    return {
        "mean_chain_depth": sum(all_chain_depths) / len(all_chain_depths),
        "max_chain_depth": max(all_chain_depths),
        "mean_eviction_rate": sum(all_eviction_rates) / len(all_eviction_rates),
        "tau_values": all_taus,
        "mean_tau": sum(all_taus) / len(all_taus),
    }


def run_seed(seed: int, steps: int = 500):
    """Run full experiment for one seed."""
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    # Data setup
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    docs_train, docs_val = train_val_split(docs, seed=seed)
    train_ds = CharDataset(docs_train, tokenizer, block_size=32)
    val_ds = CharDataset(docs_val, tokenizer, block_size=32)

    vocab_size = tokenizer.vocab_size
    N = 8  # expert groups

    # ----- Train cuckoo model -----
    print(f"\n--- Cuckoo Collision-Free Routing (N={N}, k=2) ---")
    ck_model = CuckooCollisionFreeRoutingGPT(
        vocab_size=vocab_size, block_size=32,
        n_embd=64, n_head=4, n_layer=4,
        n_groups=N, n_capsules_per_group=32, top_k=2,
    )
    mx.eval(ck_model.parameters())
    ck_params = count_params(ck_model)
    print(f"  Params: {ck_params:,}")

    ck_result = train(ck_model, train_ds, val_ds, steps=steps,
                      batch_size=32, lr=3e-3, seed=seed, log_every=100)
    ck_val = ck_result["val_loss"]
    print(f"  Val loss: {ck_val:.4f}")

    # Cuckoo diagnostics
    ck_diag = measure_cuckoo_diagnostics(ck_model, val_ds)
    print(f"  Mean chain depth: {ck_diag['mean_chain_depth']:.3f}")
    print(f"  Max chain depth: {ck_diag['max_chain_depth']:.1f}")
    print(f"  Eviction rate: {ck_diag['mean_eviction_rate']:.3f}")
    print(f"  Tau values: {[f'{t:.3f}' for t in ck_diag['tau_values']]}")

    # ----- Train softmax baseline -----
    print(f"\n--- Softmax Routing Baseline (N={N}, k=2) ---")
    sm_model = CapsuleMoEGPT(
        vocab_size=vocab_size, block_size=32,
        n_embd=64, n_head=4, n_layer=4,
        n_groups=N, n_capsules_per_group=32, top_k_groups=2,
    )
    mx.eval(sm_model.parameters())
    sm_params = count_params(sm_model)
    print(f"  Params: {sm_params:,}")

    sm_result = train(sm_model, train_ds, val_ds, steps=steps,
                      batch_size=32, lr=3e-3, seed=seed, log_every=100)
    sm_val = sm_result["val_loss"]
    print(f"  Val loss: {sm_val:.4f}")

    # Softmax collision rate
    sm_collision_rate = measure_softmax_collision_rate(sm_model, val_ds)
    print(f"  Softmax collision rate (gap<0.05): {sm_collision_rate:.3f}")

    # ----- Comparison -----
    pct_diff = (ck_val - sm_val) / sm_val * 100
    print(f"\n  Cuckoo vs Softmax: {pct_diff:+.2f}%")

    return {
        "seed": seed,
        "cuckoo": {
            "params": ck_params,
            "val_loss": ck_val,
            "tokens_per_sec": ck_result["tokens_per_sec"],
            "mean_chain_depth": ck_diag["mean_chain_depth"],
            "max_chain_depth": ck_diag["max_chain_depth"],
            "eviction_rate": ck_diag["mean_eviction_rate"],
            "tau_values": ck_diag["tau_values"],
            "mean_tau": ck_diag["mean_tau"],
        },
        "softmax": {
            "params": sm_params,
            "val_loss": sm_val,
            "tokens_per_sec": sm_result["tokens_per_sec"],
            "collision_rate": sm_collision_rate,
        },
        "pct_diff": pct_diff,
    }


def main():
    seeds = [42, 123, 777]
    results = []

    for seed in seeds:
        r = run_seed(seed, steps=500)
        results.append(r)

    # Aggregate
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS (3 seeds)")
    print(f"{'='*60}")

    ck_vals = [r["cuckoo"]["val_loss"] for r in results]
    sm_vals = [r["softmax"]["val_loss"] for r in results]
    pct_diffs = [r["pct_diff"] for r in results]
    chain_depths = [r["cuckoo"]["mean_chain_depth"] for r in results]
    max_depths = [r["cuckoo"]["max_chain_depth"] for r in results]
    eviction_rates = [r["cuckoo"]["eviction_rate"] for r in results]
    taus = [r["cuckoo"]["mean_tau"] for r in results]
    collision_rates = [r["softmax"]["collision_rate"] for r in results]

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return (sum((x - m)**2 for x in xs) / len(xs)) ** 0.5

    print(f"\n{'Method':<25} {'Params':>8} {'Val Loss':>10} {'Diff%':>8}")
    print("-" * 55)
    print(f"{'Softmax (baseline)':<25} {results[0]['softmax']['params']:>8,} {mean(sm_vals):>10.4f} {'---':>8}")
    print(f"{'Cuckoo':<25} {results[0]['cuckoo']['params']:>8,} {mean(ck_vals):>10.4f} {mean(pct_diffs):>+8.2f}%")

    print(f"\n--- Kill Criteria ---")
    mean_pct_diff = mean(pct_diffs)
    mean_max_depth = mean(max_depths)
    print(f"  KC1: Quality diff: {mean_pct_diff:+.2f}% (threshold: >+2% kills)")
    print(f"  KC1 verdict: {'KILLED' if mean_pct_diff > 2.0 else 'PASSES'}")
    print(f"  KC2: Max chain depth: {mean_max_depth:.1f} (threshold: >3 kills)")
    print(f"  KC2 verdict: {'KILLED' if mean_max_depth > 3.0 else 'PASSES'}")

    print(f"\n--- Routing Diagnostics ---")
    print(f"  Mean eviction rate: {mean(eviction_rates):.3f}")
    print(f"  Mean chain depth: {mean(chain_depths):.3f}")
    print(f"  Mean learned tau: {mean(taus):.3f}")
    print(f"  Softmax collision rate (gap<0.05): {mean(collision_rates):.3f}")

    print(f"\n--- Per-Seed Breakdown ---")
    print(f"{'Seed':>6} | {'CK val':>8} {'SM val':>8} {'Diff%':>8} | {'Chain':>6} {'MaxD':>5} {'Evict%':>7} {'Tau':>5} | {'SM Coll':>8}")
    for r in results:
        ck = r["cuckoo"]
        sm = r["softmax"]
        print(f"{r['seed']:>6} | {ck['val_loss']:>8.4f} {sm['val_loss']:>8.4f} {r['pct_diff']:>+8.2f} | "
              f"{ck['mean_chain_depth']:>6.3f} {ck['max_chain_depth']:>5.1f} {ck['eviction_rate']:>7.3f} {ck['mean_tau']:>5.3f} | "
              f"{sm['collision_rate']:>8.3f}")

    # Throughput comparison
    ck_tps = mean([r["cuckoo"]["tokens_per_sec"] for r in results])
    sm_tps = mean([r["softmax"]["tokens_per_sec"] for r in results])
    print(f"\n--- Throughput ---")
    print(f"  Softmax: {sm_tps:,.0f} tok/s")
    print(f"  Cuckoo: {ck_tps:,.0f} tok/s ({(ck_tps/sm_tps - 1)*100:+.1f}%)")

    # Save results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

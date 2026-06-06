"""Ablation: learned routing vs uniform routing in Capsule MoE.

Answers: Does the Level 1 group router contribute beyond simple averaging?
If uniform routing matches learned routing, the router adds no value at micro scale.

Run: python -m micro.models.capsule_moe.ablation_uniform_routing
"""

import mlx.core as mx
import mlx.nn as nn

from experiments.arena import run_single, run_multidomain, leaderboard


def run_ablation():
    """Compare capsule_moe (learned routing) vs capsule_moe_uniform across 3 seeds."""

    seeds = [42, 123, 7]

    # --- Single domain ---
    print("\n" + "=" * 70)
    print("ABLATION: Learned Router vs Uniform Routing — Single Domain")
    print("=" * 70)

    single_results = []
    for seed in seeds:
        for model_name in ["gpt", "capsule_moe", "capsule_moe_uniform"]:
            r = run_single(model_name, steps=500, seed=seed)
            r.model_name = f"{model_name}_s{seed}"
            single_results.append(r)

    print("\n" + leaderboard(single_results))

    # Aggregate by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in single_results:
        base_name = r.model_name.rsplit("_s", 1)[0]
        by_model[base_name].append(r.val_loss)

    print("\n--- Aggregated (3 seeds) ---")
    for name in ["gpt", "capsule_moe", "capsule_moe_uniform"]:
        vals = by_model[name]
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        print(f"  {name:<22s}: {mean:.4f} +/- {spread/2:.4f}")

    # --- Multi-domain ---
    print("\n" + "=" * 70)
    print("ABLATION: Learned Router vs Uniform Routing — Multi-Domain")
    print("=" * 70)

    multi_results = []
    for seed in seeds:
        for model_name in ["gpt", "capsule_moe", "capsule_moe_uniform"]:
            r = run_multidomain(model_name, steps_per_domain=300, seed=seed)
            r.model_name = f"{model_name}_s{seed}"
            multi_results.append(r)

    by_model_md = defaultdict(list)
    for r in multi_results:
        base_name = r.model_name.rsplit("_s", 1)[0]
        by_model_md[base_name].append(r.val_loss)

    print("\n--- Multi-Domain Aggregated (3 seeds) ---")
    for name in ["gpt", "capsule_moe", "capsule_moe_uniform"]:
        vals = by_model_md[name]
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        print(f"  {name:<22s}: {mean:.4f} +/- {spread/2:.4f}")


if __name__ == "__main__":
    run_ablation()

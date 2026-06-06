"""Shared Layer 0 at N=5: Full Experiment.

Tests whether sharing a single Layer 0 capsule pool across 5 domains
degrades quality by less than 2% vs per-domain Layer 0 pools, and
whether Layer 0 cross-domain Jaccard remains above 0.40.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen), 5 domains
  3. Compose with standard full concatenation (control)
  4. Compose with shared Layer 0 (three strategies: base, average, first)
  5. Also test: joint training baseline, weight averaging
  6. Profile Layer 0 cross-domain Jaccard at N=5
  7. Compare param savings at N=5 vs N=2
  8. Repeat for 3 seeds

Kill criteria:
  1. shared Layer 0 pool degrades quality >2% vs full concat at N=5
  2. Layer 0 cross-domain Jaccard drops below 0.40 at N=5
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from .shared_layer0_n5 import (
    compose_shared_layer0_n5,
    compose_full_concat_n5,
    profile_layer0_cross_domain_jaccard,
    profile_layer0_coactivation_jaccard,
    count_params,
)


# Shared experiment config (matches shared_layer0_pool / relu_router)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128       # per domain
STEPS_PRETRAIN = 300
STEPS_FINETUNE = 200
BATCH_SIZE = 32
LR = 3e-3


def _make_relu_model(vocab_size, n_capsules=N_CAPSULES):
    """Create a ReLURouterGPT model."""
    model = ReLURouterGPT(
        vocab_size=vocab_size, n_capsules=n_capsules, **BASE,
    )
    mx.eval(model.parameters())
    return model


def _freeze_attention(model):
    """Freeze everything EXCEPT capsule pool / MLP weights."""
    model.freeze()
    for layer in model.layers:
        layer.capsule_pool.unfreeze()


def _eval_domains(model, domain_datasets, batch_size=BATCH_SIZE):
    """Evaluate model on all domains, return dict with per-domain and avg loss."""
    result = {}
    for d_name in domain_datasets:
        result[d_name] = evaluate(model, domain_datasets[d_name][1], batch_size)
    result["avg"] = sum(v for k, v in result.items() if k != "avg") / len(domain_datasets)
    return result


def run_experiment(seed=42):
    """Run the full shared Layer 0 at N=5 experiment for one seed."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs, method="quintary")

    domain_datasets = {}
    for d_name, d_docs in splits.items():
        d_train, d_val = train_val_split(d_docs, seed=seed)
        domain_datasets[d_name] = (
            CharDataset(d_train, tokenizer, BASE["block_size"]),
            CharDataset(d_val, tokenizer, BASE["block_size"]),
        )

    all_docs_train, all_docs_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_docs_train, tokenizer, BASE["block_size"])
    joint_val = CharDataset(all_docs_val, tokenizer, BASE["block_size"])

    V = tokenizer.vocab_size
    domain_names = list(domain_datasets.keys())
    n_domains = len(domain_names)
    assert n_domains == 5, f"Expected 5 domains, got {n_domains}"
    results = {}

    # ============================================================
    # 1. Joint training baseline (N=5 sized capsule pool)
    # ============================================================
    print("  [1/7] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * n_domains)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)
    results["joint_params"] = count_params(model_joint)
    print(f"    joint avg: {results['joint']['avg']:.4f}")

    # ============================================================
    # 2. Pretrain base + domain fine-tune (5 domains)
    # ============================================================
    print("  [2/7] Pretrain base + fine-tune per 5 domains...")
    base = _make_relu_model(V, n_capsules=N_CAPSULES)
    train(base, joint_train, steps=STEPS_PRETRAIN,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)

    domain_models = []
    for d_name in domain_names:
        model_d = copy.deepcopy(base)
        _freeze_attention(model_d)
        train(model_d, domain_datasets[d_name][0], steps=STEPS_FINETUNE,
              batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
        domain_models.append(model_d)

    # ============================================================
    # 3. Full concatenation composition (CONTROL)
    # ============================================================
    print("  [3/7] Full concatenation composition (control)...")
    composed_full = compose_full_concat_n5(base, domain_models)
    results["full_concat"] = _eval_domains(composed_full, domain_datasets)
    results["full_concat_params"] = count_params(composed_full)
    print(f"    full_concat avg: {results['full_concat']['avg']:.4f}")

    # Weight averaging baseline
    averaged = _make_relu_model(V, n_capsules=N_CAPSULES)
    base_params = dict(nn.utils.tree_flatten(base.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items() if "capsule_pool" not in k]
    averaged.load_weights(shared_weights, strict=False)
    for layer_idx in range(len(averaged.layers)):
        A_avg = sum(dm.layers[layer_idx].capsule_pool.A.weight for dm in domain_models) / n_domains
        B_avg = sum(dm.layers[layer_idx].capsule_pool.B.weight for dm in domain_models) / n_domains
        averaged.layers[layer_idx].capsule_pool.A.load_weights([("weight", A_avg)])
        averaged.layers[layer_idx].capsule_pool.B.load_weights([("weight", B_avg)])
    mx.eval(averaged.parameters())
    results["weight_avg"] = _eval_domains(averaged, domain_datasets)
    results["weight_avg_params"] = count_params(averaged)
    print(f"    weight_avg avg: {results['weight_avg']['avg']:.4f}")

    # ============================================================
    # 4. Shared Layer 0 composition (THREE STRATEGIES)
    # ============================================================
    strategies = ["base", "average", "first"]
    for strategy in strategies:
        key = f"shared_L0_{strategy}"
        print(f"  [4/7] Shared Layer 0 ({strategy})...")
        composed_shared = compose_shared_layer0_n5(base, domain_models, strategy=strategy)
        results[key] = _eval_domains(composed_shared, domain_datasets)
        results[f"{key}_params"] = count_params(composed_shared)
        print(f"    {key} avg: {results[key]['avg']:.4f}")

    # ============================================================
    # 5. Layer 0 cross-domain Jaccard (pairwise between domain models)
    # ============================================================
    print("  [5/7] Layer 0 cross-domain Jaccard (pairwise)...")

    l0_pairwise = profile_layer0_cross_domain_jaccard(
        domain_models, base, joint_val, domain_names,
        n_batches=20, batch_size=32, seed=seed,
    )
    results["l0_pairwise_jaccard_mean"] = l0_pairwise["mean_jaccard"]
    results["l0_pairwise_jaccard_min"] = l0_pairwise["min_jaccard"]
    results["l0_pairwise_jaccard_max"] = l0_pairwise["max_jaccard"]
    results["l0_pairwise_jaccards"] = l0_pairwise["pairwise_jaccards"]
    print(f"    Layer 0 pairwise Jaccard: mean={l0_pairwise['mean_jaccard']:.4f}, "
          f"min={l0_pairwise['min_jaccard']:.4f}, max={l0_pairwise['max_jaccard']:.4f}")

    # ============================================================
    # 6. Layer 0 co-activation Jaccard in full concat (cross-pool)
    # ============================================================
    print("  [6/7] Layer 0 co-activation Jaccard in composed model...")

    l0_coact = profile_layer0_coactivation_jaccard(
        composed_full, joint_val,
        n_capsules_per_domain=N_CAPSULES,
        n_domains=n_domains,
        n_batches=20, batch_size=32, seed=seed,
    )
    results["l0_coactivation_mean"] = l0_coact["mean"]
    results["l0_coactivation_p50"] = l0_coact["p50"]
    results["l0_coactivation_p90"] = l0_coact["p90"]
    results["l0_coactivation_max"] = l0_coact["max"]
    print(f"    Layer 0 co-activation Jaccard: mean={l0_coact['mean']:.4f}, "
          f"p50={l0_coact['p50']:.4f}, p90={l0_coact['p90']:.4f}")

    # ============================================================
    # 7. Parameter analysis
    # ============================================================
    print("  [7/7] Parameter analysis...")
    n_embd = BASE["n_embd"]
    # At N=5: full concat has 5*P capsules at Layer 0, shared has P
    # Savings: (D-1)*2*P*d = 4*2*128*64 = 65,536 params
    l0_saving = (n_domains - 1) * 2 * N_CAPSULES * n_embd
    results["l0_param_saving"] = l0_saving
    results["l0_param_saving_pct"] = l0_saving / results["full_concat_params"] * 100

    # For comparison: at N=2 the saving was (2-1)*2*128*64 = 16,384
    n2_saving = 1 * 2 * N_CAPSULES * n_embd
    results["n2_l0_param_saving"] = n2_saving

    print(f"    Layer 0 param saving at N=5: {l0_saving:,} ({results['l0_param_saving_pct']:.1f}%)")
    print(f"    Layer 0 param saving at N=2: {n2_saving:,} (for comparison)")

    return results


def main():
    """Run across 3 seeds and report aggregate results."""
    seeds = [42, 123, 7]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  Seed {seed}")
        print(f"{'='*70}")
        r = run_experiment(seed=seed)
        all_results.append(r)

    # ============================================================
    # Aggregate quality results
    # ============================================================
    print(f"\n{'='*70}")
    print("  3-Seed Aggregate Results (N=5 Domains)")
    print(f"{'='*70}")

    quality_methods = [
        "joint", "full_concat", "weight_avg",
        "shared_L0_base", "shared_L0_average", "shared_L0_first",
    ]

    full_concat_mean = statistics.mean([r["full_concat"]["avg"] for r in all_results])
    joint_mean = statistics.mean([r["joint"]["avg"] for r in all_results])

    print(f"\n  {'Method':<25} {'avg':>8} {'std':>8} {'vs joint':>10} {'vs full_concat':>14}")
    print("  " + "-" * 75)
    for method in quality_methods:
        avgs = [r[method]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_joint = ((mean - joint_mean) / joint_mean) * 100
        vs_full = ((mean - full_concat_mean) / full_concat_mean) * 100
        print(f"  {method:<25} {mean:>8.4f} {std:>8.4f} {vs_joint:>+9.1f}% {vs_full:>+13.1f}%")

    # ============================================================
    # Per-domain breakdown
    # ============================================================
    print(f"\n  Per-Domain Breakdown (3-seed mean)")
    print("  " + "-" * 75)
    domain_names = [k for k in all_results[0]["joint"].keys() if k != "avg"]
    for method in quality_methods:
        for d_name in domain_names:
            vals = [r[method][d_name] for r in all_results]
            mean = statistics.mean(vals)
            full_d = statistics.mean([r["full_concat"][d_name] for r in all_results])
            vs_full = ((mean - full_d) / full_d) * 100
            print(f"  {method:<25} {d_name}: {mean:.4f} (vs full_concat: {vs_full:+.1f}%)")

    # ============================================================
    # Layer 0 Jaccard analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("  Layer 0 Cross-Domain Jaccard at N=5")
    print(f"{'='*70}")

    # Pairwise Jaccard between domain models
    pairwise_means = [r["l0_pairwise_jaccard_mean"] for r in all_results]
    pairwise_mins = [r["l0_pairwise_jaccard_min"] for r in all_results]
    pairwise_maxes = [r["l0_pairwise_jaccard_max"] for r in all_results]
    print(f"\n  Pairwise Jaccard (domain model Layer 0, same data):")
    print(f"    Mean:  {statistics.mean(pairwise_means):.4f} +/- {statistics.stdev(pairwise_means):.4f}")
    print(f"    Min:   {statistics.mean(pairwise_mins):.4f}")
    print(f"    Max:   {statistics.mean(pairwise_maxes):.4f}")

    # Co-activation Jaccard in composed model
    coact_means = [r["l0_coactivation_mean"] for r in all_results]
    coact_p50s = [r["l0_coactivation_p50"] for r in all_results]
    coact_p90s = [r["l0_coactivation_p90"] for r in all_results]
    coact_maxes = [r["l0_coactivation_max"] for r in all_results]
    print(f"\n  Co-activation Jaccard (composed model Layer 0, cross-pool):")
    print(f"    Mean:  {statistics.mean(coact_means):.4f} +/- {statistics.stdev(coact_means):.4f}")
    print(f"    P50:   {statistics.mean(coact_p50s):.4f}")
    print(f"    P90:   {statistics.mean(coact_p90s):.4f}")
    print(f"    Max:   {statistics.mean(coact_maxes):.4f}")

    # Print per-pair breakdown from first seed
    print(f"\n  Per-pair Jaccard (seed {seeds[0]}):")
    for (d_i, d_j), jac in sorted(all_results[0]["l0_pairwise_jaccards"].items()):
        print(f"    {d_i} vs {d_j}: {jac:.4f}")

    # ============================================================
    # Parameter savings comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("  Parameter Savings: N=5 vs N=2")
    print(f"{'='*70}")

    full_params = statistics.mean([r["full_concat_params"] for r in all_results])
    shared_params = statistics.mean([r["shared_L0_average_params"] for r in all_results])
    saving_n5 = statistics.mean([r["l0_param_saving"] for r in all_results])
    saving_pct = statistics.mean([r["l0_param_saving_pct"] for r in all_results])
    saving_n2 = statistics.mean([r["n2_l0_param_saving"] for r in all_results])

    print(f"\n  Full concat params (N=5):     {full_params:,.0f}")
    print(f"  Shared L0 params (N=5):       {shared_params:,.0f}")
    print(f"  Layer 0 saving at N=5:        {saving_n5:,.0f} ({saving_pct:.1f}%)")
    print(f"  Layer 0 saving at N=2:        {saving_n2:,.0f} (parent experiment)")
    print(f"  Saving ratio N=5/N=2:         {saving_n5/saving_n2:.1f}x")

    # ============================================================
    # Kill criterion analysis
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  KILL CRITERION ANALYSIS")
    print(f"{'='*70}")

    # Kill criterion 1: quality degradation >2%
    print(f"\n  Kill criterion 1: shared Layer 0 pool degrades quality >2%")
    print(f"  vs per-domain Layer 0 pools (full concatenation) at N=5")

    for strategy in ["base", "average", "first"]:
        key = f"shared_L0_{strategy}"
        avgs = [r[key]["avg"] for r in all_results]
        mean = statistics.mean(avgs)
        std = statistics.stdev(avgs) if len(avgs) > 1 else 0
        vs_full = ((mean - full_concat_mean) / full_concat_mean) * 100
        status = "KILL" if vs_full > 2.0 else "PASS"
        print(f"\n  Strategy '{strategy}':")
        print(f"    Mean: {mean:.4f} +/- {std:.4f}")
        print(f"    vs full_concat: {vs_full:+.2f}%")
        print(f"    Per-seed: {', '.join(f'{v:.4f}' for v in avgs)}")
        for i, seed in enumerate(seeds):
            per_seed_vs = ((avgs[i] - all_results[i]["full_concat"]["avg"]) / all_results[i]["full_concat"]["avg"]) * 100
            print(f"      Seed {seed}: {avgs[i]:.4f} vs full_concat {all_results[i]['full_concat']['avg']:.4f} ({per_seed_vs:+.2f}%)")
        print(f"    Verdict: [{status}]")

    # Kill criterion 2: Layer 0 cross-domain Jaccard <0.40
    print(f"\n  Kill criterion 2: Layer 0 cross-domain Jaccard drops below 0.40 at N=5")
    mean_pairwise_jac = statistics.mean(pairwise_means)
    min_pairwise_jac = min(
        min(r["l0_pairwise_jaccards"].values()) for r in all_results
    )
    print(f"\n    Mean pairwise Jaccard: {mean_pairwise_jac:.4f}")
    print(f"    Worst-case pair (any seed): {min_pairwise_jac:.4f}")
    jac_status = "KILL" if mean_pairwise_jac < 0.40 else "PASS"
    print(f"    Verdict: [{jac_status}]")

    # Best strategy
    best_strategy = min(
        ["base", "average", "first"],
        key=lambda s: statistics.mean([r[f"shared_L0_{s}"]["avg"] for r in all_results]),
    )
    best_mean = statistics.mean([r[f"shared_L0_{best_strategy}"]["avg"] for r in all_results])
    best_vs_full = ((best_mean - full_concat_mean) / full_concat_mean) * 100

    print(f"\n  Best strategy: '{best_strategy}'")
    print(f"    {best_mean:.4f} ({best_vs_full:+.2f}% vs full_concat)")

    # Overall verdict
    quality_pass = any(
        ((statistics.mean([r[f"shared_L0_{s}"]["avg"] for r in all_results]) - full_concat_mean)
         / full_concat_mean * 100) <= 2.0
        for s in ["base", "average", "first"]
    )
    jaccard_pass = mean_pairwise_jac >= 0.40

    if quality_pass and jaccard_pass:
        print(f"\n  OVERALL VERDICT: PASS")
        print(f"  Both kill criteria survived at N=5.")
    elif not quality_pass:
        print(f"\n  OVERALL VERDICT: KILL (quality)")
        print(f"  All sharing strategies degrade quality >2% vs full concatenation at N=5.")
    elif not jaccard_pass:
        print(f"\n  OVERALL VERDICT: KILL (jaccard)")
        print(f"  Layer 0 cross-domain Jaccard dropped below 0.40 at N=5.")
    else:
        print(f"\n  OVERALL VERDICT: KILL (both)")
        print(f"  Both quality and Jaccard kill criteria triggered.")


if __name__ == "__main__":
    main()

"""Shared Layer 0 Capsule Pool Experiment.

Tests whether sharing a single Layer 0 capsule pool across domains
degrades quality by less than 2% vs per-domain Layer 0 pools.

Protocol:
  1. Pretrain base model on ALL data (shared attention + embeddings)
  2. Fine-tune only MLP weights per domain (attention frozen)
  3. Compose with standard full concatenation (control)
  4. Compose with shared Layer 0 (three strategies: base, average, first)
  5. Also test: joint training baseline, weight averaging
  6. Compare: shared Layer 0 quality vs full concatenation (the kill criterion)

Kill criterion: shared Layer 0 pool degrades quality >2% vs per-domain
Layer 0 pools (full concatenation).

Additional analysis:
  - Parameter savings from shared Layer 0
  - Behavioral profiling of Layer 0 co-activation (confirm behavioral_dedup finding)
  - Per-domain quality impact (is degradation symmetric?)
"""

import copy
import random
import statistics

import mlx.core as mx
import mlx.nn as nn

from experiments.data import load_names, CharTokenizer, CharDataset, domain_split, train_val_split
from experiments.train import train, evaluate
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from .shared_layer0_pool import (
    compose_shared_layer0,
    compose_full_concat,
    count_params,
)


# Shared experiment config (matches behavioral_dedup / relu_router)
BASE = dict(n_embd=64, n_head=4, n_layer=4, block_size=32)
N_CAPSULES = 128       # per domain (composed = 256)
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


def profile_layer0_jaccard(model, dataset, n_batches=20, batch_size=32, seed=0):
    """Compute mean cross-pool Jaccard for Layer 0 (confirm behavioral_dedup).

    Returns mean Jaccard similarity for Layer 0 cross-pool capsule pairs.
    This is a lightweight version of the full behavioral profiling.
    """
    rng = random.Random(seed)
    layer = model.layers[0]
    P = layer.capsule_pool.n_capsules

    co_fire = mx.zeros((P, P))
    fire_count = mx.zeros((P,))
    total_positions = 0

    for batch_idx in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B_size, T = inputs.shape

        # Forward to Layer 0 capsule input
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)
        x_norm1 = layer.norm1(x)
        x = x + layer.attn(x_norm1)
        x_norm2 = layer.norm2(x)

        pool = layer.capsule_pool
        h = nn.relu(pool.A(x_norm2))  # (B, T, P)
        fired = (h > 0).astype(mx.float32)  # (B, T, P)
        fired_flat = fired.reshape(-1, P)  # (N, P)

        co_fire = co_fire + (fired_flat.T @ fired_flat)
        fire_count = fire_count + mx.sum(fired_flat, axis=0)
        total_positions += B_size * T

        if batch_idx % 5 == 0:
            mx.eval(co_fire, fire_count)

    mx.eval(co_fire, fire_count)

    # Compute Jaccard matrix
    fc_sum = fire_count[:, None] + fire_count[None, :]
    union = fc_sum - co_fire
    J = co_fire / (union + 1e-8)
    mx.eval(J)

    return {
        "jaccard_matrix": J,
        "fire_count": fire_count,
        "total_positions": total_positions,
        "P": P,
    }


def run_experiment(seed=42):
    """Run the full shared Layer 0 experiment for one seed."""
    docs = load_names()
    tokenizer = CharTokenizer(docs)
    splits = domain_split(docs)

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
    results = {}

    # ============================================================
    # 1. Joint training baseline
    # ============================================================
    print("  [1/6] Joint training baseline...")
    model_joint = _make_relu_model(V, n_capsules=N_CAPSULES * 2)
    train(model_joint, joint_train, steps=STEPS_PRETRAIN + STEPS_FINETUNE,
          batch_size=BATCH_SIZE, lr=LR, seed=seed, log_every=9999)
    results["joint"] = _eval_domains(model_joint, domain_datasets)
    results["joint_params"] = count_params(model_joint)
    print(f"    joint avg: {results['joint']['avg']:.4f}")

    # ============================================================
    # 2. Pretrain base + domain fine-tune
    # ============================================================
    print("  [2/6] Pretrain base + fine-tune per domain...")
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
    print("  [3/6] Full concatenation composition (control)...")
    composed_full = compose_full_concat(base, domain_models)
    results["full_concat"] = _eval_domains(composed_full, domain_datasets)
    results["full_concat_params"] = count_params(composed_full)
    print(f"    full_concat avg: {results['full_concat']['avg']:.4f}")

    # Weight averaging baseline
    n_domains = len(domain_models)
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
        print(f"  [4/6] Shared Layer 0 ({strategy})...")
        composed_shared = compose_shared_layer0(base, domain_models, strategy=strategy)
        results[key] = _eval_domains(composed_shared, domain_datasets)
        results[f"{key}_params"] = count_params(composed_shared)
        print(f"    {key} avg: {results[key]['avg']:.4f}")

    # ============================================================
    # 5. Layer 0 co-activation analysis (confirm behavioral_dedup)
    # ============================================================
    print("  [5/6] Layer 0 co-activation analysis...")

    # Profile the full-concat composed model's Layer 0
    l0_stats = profile_layer0_jaccard(composed_full, joint_val, seed=seed)
    J = l0_stats["jaccard_matrix"]
    P = l0_stats["P"]
    mx.eval(J)
    J_np = J.tolist()

    # Compute cross-pool mean Jaccard (first N_CAPSULES vs second N_CAPSULES)
    cross_jacs = []
    for i in range(N_CAPSULES):
        for j in range(N_CAPSULES, P):
            cross_jacs.append(J_np[i][j])

    if cross_jacs:
        results["l0_cross_jaccard_mean"] = sum(cross_jacs) / len(cross_jacs)
        cross_jacs.sort()
        n = len(cross_jacs)
        results["l0_cross_jaccard_p50"] = cross_jacs[n // 2]
        results["l0_cross_jaccard_p90"] = cross_jacs[int(n * 0.90)]
        results["l0_cross_jaccard_max"] = cross_jacs[-1]
    else:
        results["l0_cross_jaccard_mean"] = 0.0

    print(f"    Layer 0 cross-pool Jaccard: mean={results['l0_cross_jaccard_mean']:.4f}")

    # ============================================================
    # 6. Parameter analysis
    # ============================================================
    print("  [6/6] Parameter analysis...")
    # Full concat: Layer 0 has 2*N_CAPSULES capsules (concatenated)
    # Shared L0: Layer 0 has N_CAPSULES capsules (shared)
    # Savings in Layer 0: N_CAPSULES * d (A) + d * N_CAPSULES (B) = 2 * N_CAPSULES * d
    n_embd = BASE["n_embd"]
    l0_saving = 2 * N_CAPSULES * n_embd  # A + B params saved
    results["l0_param_saving"] = l0_saving
    results["l0_param_saving_pct"] = l0_saving / results["full_concat_params"] * 100
    print(f"    Layer 0 param saving: {l0_saving:,} ({results['l0_param_saving_pct']:.1f}% of composed model)")

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
    print("  3-Seed Aggregate Results")
    print(f"{'='*70}")

    quality_methods = [
        "joint", "full_concat", "weight_avg",
        "shared_L0_base", "shared_L0_average", "shared_L0_first",
    ]

    # Reference: full_concat is the control for the kill criterion
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
    # Per-domain breakdown (check asymmetry)
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
    # Layer 0 co-activation confirmation
    # ============================================================
    print(f"\n  Layer 0 Cross-Pool Jaccard (3-seed)")
    print("  " + "-" * 75)
    jac_means = [r["l0_cross_jaccard_mean"] for r in all_results]
    jac_p50s = [r.get("l0_cross_jaccard_p50", 0) for r in all_results]
    jac_p90s = [r.get("l0_cross_jaccard_p90", 0) for r in all_results]
    jac_maxes = [r.get("l0_cross_jaccard_max", 0) for r in all_results]
    print(f"  Mean: {statistics.mean(jac_means):.4f} (per seed: {', '.join(f'{v:.4f}' for v in jac_means)})")
    print(f"  P50:  {statistics.mean(jac_p50s):.4f}")
    print(f"  P90:  {statistics.mean(jac_p90s):.4f}")
    print(f"  Max:  {statistics.mean(jac_maxes):.4f}")

    # ============================================================
    # Parameter savings
    # ============================================================
    print(f"\n  Parameter Analysis")
    print("  " + "-" * 75)
    full_params = statistics.mean([r["full_concat_params"] for r in all_results])
    shared_params = statistics.mean([r["shared_L0_average_params"] for r in all_results])
    saving = statistics.mean([r["l0_param_saving"] for r in all_results])
    saving_pct = statistics.mean([r["l0_param_saving_pct"] for r in all_results])
    print(f"  Full concat params:    {full_params:,.0f}")
    print(f"  Shared L0 params:      {shared_params:,.0f}")
    print(f"  Layer 0 saving:        {saving:,.0f} ({saving_pct:.1f}%)")

    # ============================================================
    # Kill criterion analysis
    # ============================================================
    print(f"\n  {'='*70}")
    print(f"  KILL CRITERION ANALYSIS")
    print(f"  {'='*70}")
    print(f"\n  Kill criterion: shared Layer 0 pool degrades quality >2%")
    print(f"  vs per-domain Layer 0 pools (full concatenation)")

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

    # Best strategy
    best_strategy = min(
        ["base", "average", "first"],
        key=lambda s: statistics.mean([r[f"shared_L0_{s}"]["avg"] for r in all_results]),
    )
    best_mean = statistics.mean([r[f"shared_L0_{best_strategy}"]["avg"] for r in all_results])
    best_vs_full = ((best_mean - full_concat_mean) / full_concat_mean) * 100
    best_status = "KILL" if best_vs_full > 2.0 else "PASS"

    print(f"\n  Best strategy: '{best_strategy}'")
    print(f"    {best_mean:.4f} ({best_vs_full:+.2f}% vs full_concat) [{best_status}]")

    # Overall verdict
    any_pass = any(
        ((statistics.mean([r[f"shared_L0_{s}"]["avg"] for r in all_results]) - full_concat_mean)
         / full_concat_mean * 100) <= 2.0
        for s in ["base", "average", "first"]
    )

    if any_pass:
        print(f"\n  OVERALL VERDICT: PASS")
        print(f"  At least one sharing strategy stays within 2% of full concatenation.")
    else:
        print(f"\n  OVERALL VERDICT: KILL")
        print(f"  All sharing strategies degrade quality >2% vs full concatenation.")


if __name__ == "__main__":
    main()
